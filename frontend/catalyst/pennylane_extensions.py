# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains various functions for enabling Catalyst functionality
(such as mid-circuit measurements and advanced control flow) from PennyLane
while using :func:`~.qjit`.
"""

import functools
import numbers
import uuid
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.lax.control_flow import (
    _initial_style_jaxpr,
    _initial_style_jaxprs_with_common_consts,
)
from jax._src.lax.lax import _abstractify
from jax.core import ShapedArray
from jax.linear_util import wrap_init
from jax.tree_util import tree_flatten, tree_structure, tree_unflatten, treedef_is_leaf
from pennylane import QNode
from pennylane.measurements import MidMeasureMP
from pennylane.operation import AnyWires, Operation, Operator, Wires
from pennylane.queuing import QueuingManager

import catalyst
import catalyst.jax_primitives as jprim
from catalyst.jax_primitives import GradParams, expval_p, probs_p
from catalyst.jax_tape import JaxTape
from catalyst.jax_tracer import get_traceable_fn, insert_to_qreg, trace_quantum_tape
from catalyst.jax_tracer2 import (
    MidCircuitMeasure,
    adjoint,
    cond,
    for_loop,
    measure,
    trace_quantum_function,
    while_loop,
)
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.tracing import TracingContext, EvaluationMode

# pylint: disable=too-many-lines


def _trace_quantum_tape(
    cargs, ckwargs, qargs, _callee: Callable, _allow_quantum_measurements: bool = True
) -> Tuple[Any, Any]:
    """Jax-trace the ``_callee`` function accepting positional and keyword arguments and containig
    quantum calls by running it under the PennyLane's quantum tape recorder.

    Args:
        cargs (Jaxpr): classical positional arguemnts to be passed to ``_callee``
        kwargs (Jaxpr): classical keyword arguemnts to be passed to ``_callee``
        qargs (Jaxpr): quantum arguments to consume in the course of tracing
        _callee (Callable): function to trace
        _allow_quantum_measurements (bool): If set to False, raise an exception if quantum
                                            measurements are detected
    Returns (Tuple[Any,Any]):
        - Jax representaion of classical return values of ``_callee``
        - Jax representation of quantum return values obtained in the course of tracing
    """
    assert len(qargs) == 1, f"A single quantum argument was expected, got {qargs}"
    with qml.QueuingManager.stop_recording():
        with JaxTape() as tape:
            with tape.quantum_tape:
                out = _callee(*cargs, **ckwargs)
            if not _allow_quantum_measurements and len(tape.quantum_tape.measurements) > 0:
                raise ValueError("Quantum measurements are not allowed in this scope")
            if isinstance(out, Operation):
                out = None
            tape.set_return_val(out)
            new_quantum_tape = JaxTape.device.expand_fn(tape.quantum_tape)
            tape.quantum_tape = new_quantum_tape
            tape.quantum_tape.jax_tape = tape

    has_tracer_return_values = out is not None
    qreg = qargs[0]
    return_values, qreg, qubit_states = trace_quantum_tape(tape, qreg, has_tracer_return_values)
    qreg = insert_to_qreg(qubit_states, qreg)

    # To support retvals in nested loops
    if return_values and len(return_values) == 1:
        return_values = return_values[0]

    return return_values, [qreg]


class QFunc:
    """A device specific quantum function.

    Args:
        qfunc (Callable): the quantum function
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values
        device (a derived class from QubitDevice): a device specification which determines
            the valid gate set for the quantum function
    """

    # The set of supported devices at runtime
    RUNTIME_DEVICES = (
        "lightning.qubit",
        "lightning.kokkos",
        "braket.aws.qubit",
        "braket.local.qubit",
    )

    def __init__(self, fn, device):
        self.func = fn
        self.device = device
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if isinstance(self, qml.QNode):
            if self.device.short_name not in QFunc.RUNTIME_DEVICES:
                raise CompileError(
                    f"The {self.device.short_name} device is not "
                    "supported for compilation at the moment."
                )

            backend_kwargs = {}
            if hasattr(self.device, "shots"):
                backend_kwargs["shots"] = self.device.shots if self.device.shots else 0
            if self.device.short_name == "braket.local.qubit":  # pragma: no cover
                backend_kwargs["backend"] = self.device._device._delegate.DEVICE_ID
            elif self.device.short_name == "braket.aws.qubit":  # pragma: no cover
                backend_kwargs["device_arn"] = self.device._device._arn
                if self.device._s3_folder:
                    backend_kwargs["s3_destination_folder"] = str(self.device._s3_folder)

            device = QJITDevice(
                self.device.shots, self.device.wires, self.device.short_name, backend_kwargs
            )
        else:
            # Allow QFunc to still be used by itself for internal testing.
            device = self.device

        # traceable_fn = get_traceable_fn(self.func, device)
        # jaxpr, shape = jax.make_jaxpr(traceable_fn, return_shape=True)(*args)
        with TracingContext(EvaluationMode.QUANTUM_COMPILATION):
            jaxpr, shape = trace_quantum_function(self.func, device, args, kwargs)

        retval_tree = tree_structure(shape)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        args_data, _ = tree_flatten(args)

        wrapped = wrap_init(_eval_jaxpr)
        retval = jprim.func_p.bind(wrapped, *args_data, fn=self)

        return tree_unflatten(retval_tree, retval)


def qfunc(num_wires, *, shots=1000, device=None):
    """A Device specific quantum function.

    Args:
        num_wires (int): the number of wires
        fn (Callable): the quantum function
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. It defaults to 1000.
        device (a derived class from QubitDevice): A device specification which determines
            the valid gate set for the quantum function. It defaults to ``QJITDevice`` if not
            specified.

    Returns:
        Grad: A QFunc object that denotes the the declaration of a quantum function.

    """

    if not device:
        device = QJITDevice(shots=shots, wires=num_wires)

    def dec_no_params(fn):
        return QFunc(fn, device)

    return dec_no_params


class Function:
    """An object that represents a compiled function.

    At the moment, it is only used to compute sensible names for higher order derivative
    functions in MLIR.

    Args:
        fn (Callable): the function boundary.

    Raises:
        AssertionError: Invalid function type.
    """

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        jaxpr, shape = jax.make_jaxpr(self.fn, return_shape=True)(*args)
        shape_tree = tree_structure(shape)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        retval = jprim.func_p.bind(wrap_init(_eval_jaxpr), *args, fn=self)
        return tree_unflatten(shape_tree, retval)


Differentiable = Union[Function, QNode]
DifferentiableLike = Union[Differentiable, Callable, "catalyst.compilation_pipelines.QJIT"]
Jaxpr = Any


def _ensure_differentiable(f: DifferentiableLike) -> Differentiable:
    """Narrows down the set of the supported differentiable objects."""

    # Unwrap the function from an existing QJIT object.
    if isinstance(f, catalyst.compilation_pipelines.QJIT):
        f = f.qfunc

    if isinstance(f, (Function, QNode)):
        return f
    elif isinstance(f, Callable):  # keep at the bottom
        return Function(f)

    raise DifferentiableCompileError(f"Non-differentiable object passed: {type(f)}")


def _make_jaxpr_check_differentiable(f: Differentiable, grad_params: GradParams, *args) -> Jaxpr:
    """Gets the jaxpr of a differentiable function. Perform the required additional checks."""
    method = grad_params.method
    jaxpr = jax.make_jaxpr(f)(*args)

    assert len(jaxpr.eqns) == 1, "Expected jaxpr consisting of a single function call."
    assert (
        jaxpr.eqns[0].primitive == jprim.func_p
    ), "Expected jaxpr consisting of a single function call."

    for pos, arg in enumerate(jaxpr.in_avals):
        if arg.dtype.kind != "f" and pos in grad_params.argnum:
            raise DifferentiableCompileError(
                "Catalyst.grad only supports differentiation on floating-point "
                f"arguments, got '{arg.dtype}' at position {pos}."
            )
    for pos, res in enumerate(jaxpr.out_avals):
        if res.dtype.kind != "f":
            raise DifferentiableCompileError(
                "Catalyst.grad only supports differentiation on floating-point "
                f"results, got '{res.dtype}' at position {pos}."
            )

    _check_created_jaxpr_gradient_methods(f, method, jaxpr)

    return jaxpr


def _check_created_jaxpr_gradient_methods(f: Differentiable, method: str, jaxpr: Jaxpr):
    """Additional checks for the given jaxpr of a differentiable function."""
    if method == "fd":
        return

    qnode_jaxpr = jaxpr.eqns[0].params["call_jaxpr"]
    return_ops = []
    for res in qnode_jaxpr.outvars:
        for eq in reversed(qnode_jaxpr.eqns):  # pragma: no branch
            if res in eq.outvars:
                return_ops.append(eq.primitive)
                break

    assert isinstance(
        f, qml.QNode
    ), "Differentiation methods other than finite-differences can only operate on a QNode"
    if f.diff_method is None:
        raise DifferentiableCompileError(
            "Cannot differentiate a QNode explicitly marked non-differentiable (with"
            " diff_method=None)"
        )

    if f.diff_method == "parameter-shift" and any(
        prim not in [expval_p, probs_p] for prim in return_ops
    ):
        raise DifferentiableCompileError(
            "The parameter-shift method can only be used for QNodes "
            "which return either qml.expval or qml.probs."
        )
    if f.diff_method == "adjoint" and any(prim not in [expval_p] for prim in return_ops):
        raise DifferentiableCompileError(
            "The adjoint method can only be used for QNodes which return qml.expval."
        )


def _check_grad_params(
    method: str, h: Optional[float], argnum: Optional[Union[int, List[int]]]
) -> GradParams:
    methods = {"fd", "defer"}
    if method is None:
        method = "fd"
    if method not in methods:
        raise ValueError(
            f"Invalid differentiation method '{method}'. "
            f"Supported methods are: {' '.join(sorted(methods))}"
        )
    if method == "fd" and h is None:
        h = 1e-7
    if not (h is None or isinstance(h, numbers.Number)):
        raise ValueError(f"Invalid h value ({h}). None or number was excpected.")
    if argnum is None:
        argnum = [0]
    elif isinstance(argnum, int):
        argnum = [argnum]
    elif isinstance(argnum, tuple):
        argnum = list(argnum)
    elif isinstance(argnum, list) and all(isinstance(i, int) for i in argnum):
        pass
    else:
        raise ValueError(f"argnum should be integer or a list of integers, not {argnum}")
    return GradParams(method, h, argnum)


class Grad:
    """An object that specifies that a function will be differentiated.

    Args:
        fn (Differentiable): the function to differentiate
        method (str): the method used for differentiation
        h (float): the step-size value for the finite difference method
        argnum (list[int]): the argument indices which define over which arguments to differentiate

    Raises:
        ValueError: Higher-order derivatives and derivatives of non-QNode functions can only be
                    computed with the finite difference method.
        TypeError: Non-differentiable object was passed as `fn` argument.
    """

    def __init__(self, fn: Differentiable, *, grad_params: GradParams):
        self.fn = fn
        self.__name__ = f"grad.{fn.__name__}"
        self.grad_params = grad_params
        if self.grad_params.method != "fd" and not isinstance(self.fn, qml.QNode):
            raise ValueError(
                "Only finite difference can compute higher order derivatives "
                "or gradients of non-QNode functions."
            )

    def __call__(self, *args, **kwargs):
        """Specifies that an actual call to the differentiated function.
        Args:
            args: the arguments to the differentiated function
        """
        TracingContext.check_is_tracing(
            "catalyst.grad can only be used from within @qjit decorated code."
        )
        jaxpr = _make_jaxpr_check_differentiable(self.fn, self.grad_params, *args)

        args_data, _ = tree_flatten(args)

        # It always returns list as required by catalyst control-flows
        return jprim.grad_p.bind(*args_data, jaxpr=jaxpr, fn=self, grad_params=self.grad_params)


def grad(f: DifferentiableLike, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible gradient transformation for PennyLane/Catalyst.

    This function allows the gradient of a hybrid quantum-classical function
    to be computed within the compiled program.

    .. warning::

        If parameter-shift or adjoint is specified, this will only be used
        for internal _quantum_ functions. Classical components will be differentiated
        using finite-differences.

    .. warning::

        Currently, higher-order differentiation or differentiation of non-QNode functions
        is only supported by the finite-difference method.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``grad`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    Args:
        f (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of
                      ``["fd", "defer"]``,
            where:

            - ``"fd"`` represents first-order finite-differences for the entire hybrid
              circuit,

            - ``"defer"`` represents deferring the quantum differentiation to the method
              specified by the QNode, while the classical computation is differentiated
              using traditional auto-diff.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Grad: A Grad object that denotes the derivative of a function.

    Raises:
        ValueError: Invalid method or step size parameters.

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliY(0))

            g = grad(circuit)
            return g(x)

    >>> workflow(2.0)
    array(-3.14159265)
    """
    return Grad(_ensure_differentiable(f), grad_params=_check_grad_params(method, h, argnum))


def jvp(f: DifferentiableLike, params, tangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian-vector product for PennyLane/Catalyst.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    Args:
        f (Callable): Function-like object to calculate JVP for
        params (List[Array]): List (or a tuple) of the fnuction arguments specifying the point
                              to calculate JVP at. A subset of these parameters are declared as
                              differentiable by listing their indices in the ``argnum`` parameter.
        tangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size and
                               shapes must match the ones of differentiable params.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns (Tuple[Array]):
        Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example 1 (basic usage)**

    .. code-block:: python

        @qjit
        def jvp(params, tangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return catalyst.jvp(f, [params], [tangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> tangent = jnp.array([0.3, 0.6])
    >>> jvp(x, tangent)
    [array([0.09983342, 0.04      , 0.02      ]),
    array([0.29850125, 0.24000006, 0.12      ])]

    **Example 2 (argnum usage)**

    Here we show how to use ``argnum`` to ignore the non-differentiable parameter ``n`` of the
    target function. Note that the length and shapes of tangents must match the length and shape of
    primal parameters which we mark as differentiable by passing their indices to ``argnum``.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(n, params):
            qml.RX(params[n, 0], wires=n)
            qml.RY(params[n, 1], wires=n)
            return qml.expval(qml.PauliZ(1))

        @qjit
        def workflow(primals, tangents):
            return catalyst.jvp(circuit, [1, primals], [tangents], argnum=[1])

    >>> params = jnp.array([[0.54, 0.3154], [0.654, 0.123]])
    >>> dy = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    >>> workflow(params, dy)
    [array(0.78766064), array(-0.7011436)]
    """
    TracingContext.check_is_tracing(
        "catalyst.jvp can only be used from within @qjit decorated code."
    )

    def _check(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")
        return x

    params = _check(params, "params")
    tangents = _check(tangents, "tangents")
    fn: Differentiable = _ensure_differentiable(f)
    grad_params = _check_grad_params(method, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return jprim.jvp_p.bind(*params, *tangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


def vjp(f: DifferentiableLike, params, cotangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product for PennyLane/Catalyst.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    Args:
        f(Callable): Function-like object to calculate JVP for
        params(List[Array]): List (or a tuble) of f's arguments specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnum`` paramerer.
        cotangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size
                                 and shapes must match the size and shape of ``f`` outputs.
        method(str): Differentiation method to use, same as in ``grad``.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns (Tuple[Array]):
        Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example**

    .. code-block:: python

        @qjit
        def vjp(params, cotangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return catalyst.vjp(f, [params], [cotangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> dy = jnp.array([-0.5, 0.1, 0.3])
    >>> vjp(x, dy)
    [array([0.09983342, 0.04      , 0.02      ]),
    array([-0.43750208,  0.07000001])]
    """
    TracingContext.check_is_tracing(
        "catalyst.vjp can only be used from within @qjit decorated code."
    )

    def _check(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")
        return x

    params = _check(params, "params")
    cotangents = _check(cotangents, "cotangents")
    fn: Differentiable = _ensure_differentiable(f)
    grad_params = _check_grad_params(method, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return jprim.vjp_p.bind(*params, *cotangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


class Adjoint(Operation):
    """A minimal implementation of PennyLane operation, designed with a sole purpose of being
    placed on the quantum tape"""

    num_wires = AnyWires

    def __init__(self, body_jaxpr, consts, cargs):
        self.body_jaxpr = body_jaxpr
        self.consts = list(consts)
        self.cargs = list(cargs)
        super().__init__(wires=Wires(Adjoint.num_wires))


# def adjoint(f: Union[Callable, Operator]) -> Union[Callable, Operator]:
#     """A :func:`~.qjit` compatible adjoint transformer for PennyLane/Catalyst.

#     Returns a quantum function or operator that applies the adjoint of the
#     provided function or operator.

#     .. warning::

#         This function does not support performing the adjoint
#         of quantum functions that contain Catalyst control flow
#         or mid-circuit measurements.

#     Args:
#         f (Callable or Operator): A PennyLane operation or a Python function
#                                   containing PennyLane quantum operations.

#     Returns:
#         If an Operator is provided, returns an Operator that is the adjoint. If
#         a function is provided, returns a function with the same call signature
#         that returns the Adjoint of the provided function.

#     Raises:
#         ValueError: invalid parameter values

#     **Example**

#     .. code-block:: python

#         @qjit
#         @qml.qnode(qml.device("lightning.qubit", wires=1))
#         def workflow(theta, wires):
#             catalyst.adjoint(qml.RZ)(theta, wires=wires)
#             catalyst.adjoint(qml.RZ(theta, wires=wires))
#             def func():
#                 qml.RX(theta, wires=wires)
#                 qml.RY(theta, wires=wires)
#             catalyst.adjoint(func)()
#             return qml.probs()

#     >>> workflow(pnp.pi/2, wires=0)
#     array([0.5, 0.5])
#     """

#     def _make_adjoint(*args, _callee: Callable, **kwargs):
#         cargs_qargs, tree = tree_flatten((args, kwargs, [jprim.Qreg()]))
#         cargs, _ = tree_flatten((args, kwargs))
#         cargs_qargs_aval = tuple(_abstractify(val) for val in cargs_qargs)
#         body, consts, _ = _initial_style_jaxpr(
#             partial(_trace_quantum_tape, _callee=_callee, _allow_quantum_measurements=False),
#             tree,
#             cargs_qargs_aval,
#             "adjoint",
#         )
#         return Adjoint(body, consts, cargs)

#     if isinstance(f, Callable):

#         def _callable(*args, **kwargs):
#             return _make_adjoint(*args, _callee=f, **kwargs)

#         return _callable
#     elif isinstance(f, Operator):
#         QueuingManager.remove(f)

#         def _callee():
#             QueuingManager.append(f)

#         return _make_adjoint(_callee=_callee)
#     else:
#         raise ValueError(f"Expected a callable or a qml.Operator, not {f}")


class QJITDevice(qml.QubitDevice):
    """QJIT device.

    A device that interfaces the compilation pipeline of Pennylane programs.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically
        backend_name (str): name of the device from the list of supported and compiled backend
            devices by the runtime
        backend_kwargs (Dict(str, AnyType)): An optional dictionary of the device specifications
    """

    name = "QJIT device"
    short_name = "qjit.device"
    pennylane_requires = "0.1.0"
    version = "0.0.1"
    author = ""
    operations = [
        "MidCircuitMeasure",
        "Cond",
        "WhileLoop",
        "ForLoop",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Identity",
        "S",
        "T",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "IsingXX",
        "IsingYY",
        "IsingXY",
        "IsingZZ",
        "ControlledPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "CSWAP",
        "MultiRZ",
        "QubitUnitary",
        "Adjoint",
    ]
    observables = [
        "Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Hamiltonian",
    ]

    def __init__(self, shots=None, wires=None, backend_name=None, backend_kwargs=None):
        self.backend_name = backend_name if backend_name else "default"
        self.backend_kwargs = backend_kwargs if backend_kwargs else {}
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot apply operations.")

    def default_expand_fn(self, circuit, max_expansion=10):
        """
        Most decomposition logic will be equivalent to PennyLane's decomposition.
        However, decomposition logic will differ in the following cases:

        1. All :class:`qml.QubitUnitary <pennylane.ops.op_math.Controlled>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        3. The list of device-supported gates employed by Catalyst is currently different than
            that of the ``lightning.qubit`` device, as defined by the
            :class:`~.pennylane_extensions.QJITDevice`.

        Args:
            circuit: circuit to expand
            max_expansion: the maximum number of expansion steps if no fixed-point is reached.
        """
        # Ensure catalyst.measure is used instead of qml.measure.
        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")

        # Fallback for controlled gates that won't decompose successfully.
        # Doing so before rather than after decomposition is generally a trade-off. For low
        # numbers of qubits, a unitary gate might be faster, while for large qubit numbers prior
        # decomposition is generally faster.
        # At the moment, bypassing decomposition for controlled gates will generally have a higher
        # success rate, as complex decomposition paths can fail to trace (c.f. PL #3521, #3522).

        def _decomp_controlled_unitary(self, *_args, **_kwargs):
            return qml.QubitUnitary(qml.matrix(self), wires=self.wires)

        def _decomp_controlled(self, *_args, **_kwargs):
            return qml.QubitUnitary(qml.matrix(self), wires=self.wires)

        with Patcher(
            (qml.ops.ControlledQubitUnitary, "compute_decomposition", _decomp_controlled_unitary),
            (qml.ops.Controlled, "has_decomposition", lambda self: True),
            (qml.ops.Controlled, "decomposition", _decomp_controlled),
        ):
            expanded_tape = super().default_expand_fn(circuit, max_expansion)

        self.check_validity(expanded_tape.operations, [])
        return expanded_tape
