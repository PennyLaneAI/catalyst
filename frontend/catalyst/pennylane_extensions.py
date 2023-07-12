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
from typing import Any, Callable, Iterable, List, Optional, Union

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
from jax.tree_util import tree_flatten, tree_unflatten, treedef_is_leaf
from pennylane import QNode
from pennylane.measurements import MidMeasureMP
from pennylane.operation import AnyWires, Operation, Wires

import catalyst
import catalyst.jax_primitives as jprim
from catalyst.jax_primitives import GradParams, expval_p, probs_p
from catalyst.jax_tape import JaxTape
from catalyst.jax_tracer import get_traceable_fn, insert_to_qreg, trace_quantum_tape
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.tracing import TracingContext

# pylint: disable=too-many-lines


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

        traceable_fn = get_traceable_fn(self.func, device)
        jaxpr = jax.make_jaxpr(traceable_fn)(*args)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        wrapped = wrap_init(_eval_jaxpr)
        retval = jprim.func_p.bind(wrapped, *args, fn=self)
        if len(retval) == 1:
            retval = retval[0]
        return retval


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
        jaxpr = jax.make_jaxpr(self.fn)(*args)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        return jprim.func_p.bind(wrap_init(_eval_jaxpr), *args, fn=self)


Differentiable = Union[Function, QNode]
DifferentiableLike = Union[Differentiable, Callable, "catalyst.compilation_pipelines.QJIT"]
Jaxpr = Any


def _ensure_differentiable(f: DifferentiableLike) -> Differentiable:
    """Narrows down the set of the supported differentiable objects."""
    if isinstance(f, (Function, QNode)):
        return f
    elif isinstance(f, catalyst.compilation_pipelines.QJIT):
        return f.qfunc
    elif isinstance(f, Callable):  # Keep at the bottom
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
        return jprim.grad_p.bind(*args, jaxpr=jaxpr, fn=self, grad_params=self.grad_params)


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


class Cond(Operation):
    """PennyLane's conditional operation."""

    num_wires = AnyWires

    # pylint: disable=too-many-arguments
    def __init__(self, preds, consts, branch_jaxprs, args_tree, out_trees, *args, **kwargs):
        self.preds = preds
        self.consts = consts
        self.branch_jaxprs = branch_jaxprs
        self.args_tree = args_tree
        self.out_trees = out_trees
        kwargs["wires"] = Wires(Cond.num_wires)
        super().__init__(*args, **kwargs)


class CondCallable:
    """
    Some code in this class has been adapted from the cond implementation in the JAX project at
    https://github.com/google/jax/blob/jax-v0.4.1/jax/_src/lax/control_flow/conditionals.py
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """

    def __init__(self, pred, true_fn):
        self.preds = [pred]
        self.branch_fns = [true_fn]
        self.otherwise_fn = lambda: None

    def else_if(self, pred):
        """
        Block of code to be run if this predicate evaluates to true, skipping all subsequent
        conditional blocks.

        Args:
            pred (bool): The predicate that will determine if this branch is executed.

        Returns:
            A callable decorator that wraps this 'else if' branch of the conditional and returns
            self.
        """

        def decorator(branch_fn):
            if branch_fn.__code__.co_argcount != 0:
                raise TypeError(
                    "Conditional 'else if' function is not allowed to have any arguments"
                )
            self.preds.append(pred)
            self.branch_fns.append(branch_fn)
            return self

        return decorator

    def otherwise(self, otherwise_fn):
        """Block of code to be run if the predicate evaluates to false.

        Args:
            false_fn (Callable): The code to be run in case the condition was not met.

        Returns:
            self
        """
        if otherwise_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'False' function is not allowed to have any arguments")
        self.otherwise_fn = otherwise_fn
        return self

    @staticmethod
    def _check_branches_return_types(branch_jaxprs):
        expected = branch_jaxprs[0].out_avals[:-1]
        for i, jaxpr in list(enumerate(branch_jaxprs))[1:]:
            if expected != jaxpr.out_avals[:-1]:
                raise TypeError(
                    "Conditional branches all require the same return type, got:\n"
                    f" - Branch at index 0: {expected}\n"
                    f" - Branch at index {i}: {jaxpr.out_avals[:-1]}\n"
                    "Please specify an else branch if none was specified."
                )

    def _call_with_quantum_ctx(self, ctx):
        def new_branch_fn(branch_fn):
            def callback(qreg):
                with qml.QueuingManager.stop_recording():
                    with JaxTape() as tape:
                        with tape.quantum_tape:
                            out = branch_fn()
                        tape.set_return_val(out)
                        new_quantum_tape = JaxTape.device.expand_fn(tape.quantum_tape)
                        tape.quantum_tape = new_quantum_tape
                        tape.quantum_tape.jax_tape = tape

                has_tracer_return_values = out is not None
                return_values, qreg, qubit_states = trace_quantum_tape(
                    tape, qreg, has_tracer_return_values
                )
                qreg = insert_to_qreg(qubit_states, qreg)

                return return_values, qreg

            return callback

        args, args_tree = tree_flatten((jprim.Qreg(),))
        args_avals = tuple(map(_abstractify, args))
        branch_fns = self.branch_fns + [self.otherwise_fn]
        branch_jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
            tuple(new_branch_fn(branch_fn) for branch_fn in branch_fns),
            args_tree,
            args_avals,
            "cond",
        )

        CondCallable._check_branches_return_types(branch_jaxprs)
        Cond(self.preds, consts, branch_jaxprs, args_tree, out_trees)

        # Create tracers for any non-qreg return values (if there are any).
        ret_vals, _ = tree_unflatten(out_trees[0], branch_jaxprs[0].out_avals)
        a, t = tree_flatten(ret_vals)
        return ctx.jax_tape.create_tracer(t, a)

    def _call_with_classical_ctx(self):
        args, args_tree = tree_flatten([])
        args_avals = tuple(map(_abstractify, args))

        branch_jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
            (*self.branch_fns, self.otherwise_fn), args_tree, args_avals, "cond"
        )

        CondCallable._check_branches_return_types(branch_jaxprs)

        inputs = self.preds + consts
        ret_tree_flat = jprim.qcond(branch_jaxprs, *inputs)
        return tree_unflatten(out_trees[0], ret_tree_flat)

    def _call_during_trace(self):
        TracingContext.check_is_tracing("Must use 'cond' inside tracing context.")

        ctx = qml.QueuingManager.active_context()
        if ctx is None:
            return self._call_with_classical_ctx()

        return self._call_with_quantum_ctx(ctx)

    def _call_during_interpretation(self):
        """Create a callable for conditionals."""
        for pred, branch_fn in zip(self.preds, self.branch_fns):
            if pred:
                return branch_fn()
        return self.otherwise_fn()

    def __call__(self):
        is_tracing = TracingContext.is_tracing()
        if is_tracing:
            return self._call_during_trace()

        return self._call_during_interpretation()


def cond(pred):
    """A :func:`~.qjit` compatible decorator for if-else conditionals in PennyLane/Catalyst.

    This form of control flow is a functional version of the traditional if-else conditional. This
    means that each execution path, an 'if' branch, any 'else if' branches, and a final 'otherwise'
    branch, is provided as a separate function. All functions will be traced during compilation,
    but only one of them will be executed at runtime, depending on the value of one or more
    Boolean predicates. The JAX equivalent is the ``jax.lax.cond`` function, but this version is
    optimized to work with quantum programs in PennyLane. This version also supports an 'else if'
    construct which the JAX version does not.

    Values produced inside the scope of a conditional can be returned to the outside context, but
    the return type signature of each branch must be identical. If no values are returned, the
    'otherwise' branch is optional. Refer to the example below to learn more about the syntax of
    this decorator.

    This form of control flow can also be called from the Python interpreter without needing to use
    :func:`~.qjit`.

    Args:
        pred (bool): the first predicate with which to control the branch to execute

    Returns:
        A callable decorator that wraps the first 'if' branch of the conditional.

    Raises:
        AssertionError: Branch functions cannot have arguments.

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float):

            # define a conditional ansatz
            @cond(x > 1.4)
            def ansatz():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            @ansatz.otherwise
            def ansatz():
                qml.RY(x, wires=0)

            # apply the conditional ansatz
            ansatz()

            return qml.expval(qml.PauliZ(0))

    >>> circuit(1.4)
    array(0.16996714)
    >>> circuit(1.6)
    array(0.)

    Additional 'else-if' clauses can also be included via the ``else_if`` method:

    .. code-block:: python

        @qjit
        @qml.qnode(dev)
        def circuit(x):

            @catalyst.cond(x > 2.7)
            def cond_fn():
                qml.RX(x, wires=0)

            @cond_fn.else_if(x > 1.4)
            def cond_elif():
                qml.RY(x, wires=0)

            @cond_fn.otherwise
            def cond_else():
                qml.RX(x ** 2, wires=0)

            cond_fn()

            eturn qml.probs(wires=0)

    The conditional function is permitted to also return values.
    Any value that is supported by JAX JIT compilation is supported as a return
    type. Note that this **does not** include PennyLane operations.

    .. code-block:: python

        @cond(predicate: bool)
        def conditional_fn():
            # do something when the predicate is true
            return "optionally return some value"

        @conditional_fn.otherwise
        def conditional_fn():
            # optionally define an alternative execution path
            return "if provided, return types need to be identical in both branches"

        ret_val = conditional_fn()  # must invoke the defined function
    """

    def decorator(true_fn):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return decorator


class WhileLoop(Operation):
    """PennyLane's while loop operation."""

    num_wires = AnyWires

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        iter_args,
        body_jaxpr,
        cond_jaxpr,
        cond_consts,
        body_consts,
        body_tree,
        *args,
        **kwargs,
    ):
        self.iter_args = iter_args
        self.body_jaxpr = body_jaxpr
        self.cond_jaxpr = cond_jaxpr
        self.cond_consts = cond_consts
        self.body_consts = body_consts
        self.body_tree = body_tree
        kwargs["wires"] = Wires(WhileLoop.num_wires)
        super().__init__(*args, **kwargs)


class WhileCallable:
    """
    Some code in this class has been adapted from the while loop implementation in the JAX project
    at https://github.com/google/jax/blob/jax-v0.4.1/jax/_src/lax/control_flow/loops.py released
    under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """

    def __init__(self, cond_fn, body_fn):
        self.cond_fn = cond_fn
        self.body_fn = body_fn

    @staticmethod
    def _create_jaxpr(init_val, new_cond, new_body):
        init_vals, in_tree = tree_flatten(init_val)
        init_avals = tuple(_abstractify(val) for val in init_vals)
        cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
            new_cond, in_tree, init_avals, "while_cond"
        )
        body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
            new_body, in_tree, init_avals, "while_loop"
        )
        if not treedef_is_leaf(cond_tree) or len(cond_jaxpr.out_avals) != 1:
            msg = "cond_fun must return a boolean scalar, but got pytree {}."
            raise TypeError(msg.format(cond_tree))
        pred_aval = cond_jaxpr.out_avals[0]
        if not isinstance(
            pred_aval, ShapedArray
        ) or pred_aval.strip_weak_type().strip_named_shape() != ShapedArray((), jnp.bool_):
            msg = "cond_fun must return a boolean scalar, but got output type(s) {}."
            raise TypeError(msg.format(cond_jaxpr.out_avals))

        return body_jaxpr, cond_jaxpr, cond_consts, body_consts, body_tree

    def _call_with_quantum_ctx(self, ctx, args):
        def new_cond(*args_and_qreg):
            args = args_and_qreg[:-1]
            return self.cond_fn(*args)

        def new_body(*args_and_qreg):
            args, qreg = args_and_qreg[:-1], args_and_qreg[-1]

            with qml.QueuingManager.stop_recording():
                with JaxTape() as tape:
                    with tape.quantum_tape:
                        out = self.body_fn(*args)
                    tape.set_return_val(out)
                    new_quantum_tape = JaxTape.device.expand_fn(tape.quantum_tape)
                    tape.quantum_tape = new_quantum_tape
                    tape.quantum_tape.jax_tape = tape

            has_tracer_return_values = True
            return_values, qreg, qubit_states = trace_quantum_tape(
                tape, qreg, has_tracer_return_values
            )
            qreg = insert_to_qreg(qubit_states, qreg)

            return return_values, qreg

        body_jaxpr, cond_jaxpr, cond_consts, body_consts, body_tree = WhileCallable._create_jaxpr(
            (*args, jprim.Qreg()), new_cond, new_body
        )
        flat_init_vals_no_qubits = tree_flatten(args)[0]

        WhileLoop(
            flat_init_vals_no_qubits,
            body_jaxpr,
            cond_jaxpr,
            cond_consts,
            body_consts,
            body_tree,
        )

        ret_vals, _ = tree_unflatten(body_tree, body_jaxpr.out_avals)
        a, t = tree_flatten(ret_vals)
        return ctx.jax_tape.create_tracer(t, a)

    def _call_with_classical_ctx(self, args):
        body_jaxpr, cond_jaxpr, cond_consts, body_consts, body_tree = WhileCallable._create_jaxpr(
            args, self.cond_fn, self.body_fn
        )
        flat_init_vals_no_qubits = tree_flatten(args)[0]

        inputs = cond_consts + body_consts + flat_init_vals_no_qubits
        ret_tree_flat = jprim.qwhile(
            cond_jaxpr, body_jaxpr, len(cond_consts), len(body_consts), *inputs
        )
        return tree_unflatten(body_tree, ret_tree_flat)

    def _call_during_trace(self, *args):
        TracingContext.check_is_tracing("Must use 'while_loop' inside tracing context.")

        ctx = qml.QueuingManager.active_context()
        if ctx is not None:
            return self._call_with_quantum_ctx(ctx, args)

        return self._call_with_classical_ctx(args)

    def _call_during_interpretation(self, *args):
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None

        while self.cond_fn(*args):
            fn_res = self.body_fn(*args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()

        return fn_res

    def __call__(self, *args):
        is_tracing = TracingContext.is_tracing()
        if is_tracing:
            return self._call_during_trace(*args)

        return self._call_during_interpretation(*args)


def while_loop(cond_fn):
    """A :func:`~.qjit` compatible while-loop decorator for PennyLane/Catalyst.

    This decorator provides a functional version of the traditional while
    loop, similar to ``jax.lax.while_loop``. That is, any variables that are
    modified across iterations need to be provided as inputs and outputs to
    the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration

    - Output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the
    transformed function.

    This form of control flow can also be called from the Python interpreter without needing to use
    :func:`~.qjit`.

    The semantics of ``while_loop`` are given by the following Python pseudo-code:

    .. code-block:: python

        def while_loop(cond_fun, body_fun, *args):
            while cond_fun(*args):
                args = body_fn(*args)
            return args

    Args:
        cond_fn (Callable): the condition function in the while loop

    Returns:
        Callable: A wrapper around the while-loop function.

    Raises:
        TypeError: Invalid return type of the condition expression.

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float):

            @while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x ** 2

            # apply the while loop
            final_x = loop_rx(x)

            return qml.expval(qml.PauliZ(0)), final_x

    >>> circuit(1.6)
    [array(-0.02919952), array(2.56)]
    """

    def _while_loop(body_fn):
        return WhileCallable(cond_fn, body_fn)

    return _while_loop


class ForLoop(Operation):
    """PennyLane ForLoop Operation."""

    num_wires = AnyWires

    # pylint: disable=too-many-arguments
    def __init__(self, loop_bounds, iter_args, body_jaxpr, body_consts, body_tree, *args, **kwargs):
        self.loop_bounds = loop_bounds
        self.iter_args = iter_args
        self.body_jaxpr = body_jaxpr
        self.body_consts = body_consts
        self.body_tree = body_tree
        kwargs["wires"] = Wires(ForLoop.num_wires)
        super().__init__(*args, **kwargs)


class ForLoopCallable:
    """
    Some code in this class has been adapted from the for loop implementation in the JAX project at
    https://github.com/google/jax/blob/jax-v0.4.1/jax/_src/lax/control_flow/for_loop.py
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """

    def __init__(self, lower_bound, upper_bound, step, body_fn):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.body_fn = body_fn

    @staticmethod
    def _create_jaxpr(init_val, new_body):
        init_vals, in_tree = tree_flatten(init_val)
        init_avals = tuple(_abstractify(val) for val in init_vals)
        body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
            new_body, in_tree, init_avals, "for_loop"
        )

        return body_jaxpr, body_consts, body_tree

    def _call_with_quantum_ctx(self, ctx, *args):
        # Insert iteration counter into loop body arguments with the type of the lower bound.
        args = (self.lower_bound, *args)

        def new_body(*args_and_qreg):
            args, qreg = args_and_qreg[:-1], args_and_qreg[-1]

            with qml.QueuingManager.stop_recording():
                with JaxTape() as tape:
                    with tape.quantum_tape:
                        out = self.body_fn(*args)
                    tape.set_return_val(out)
                    new_quantum_tape = JaxTape.device.expand_fn(tape.quantum_tape)
                    tape.quantum_tape = new_quantum_tape
                    tape.quantum_tape.jax_tape = tape

            has_tracer_return_values = out is not None
            return_values, qreg, qubit_states = trace_quantum_tape(
                tape, qreg, has_tracer_return_values
            )
            qreg = insert_to_qreg(qubit_states, qreg)

            return return_values, qreg

        body_jaxpr, body_consts, body_tree = ForLoopCallable._create_jaxpr(
            (*args, jprim.Qreg()), new_body
        )

        flat_init_vals_no_qubits = tree_flatten(args)[0]

        ForLoop(
            [self.lower_bound, self.upper_bound, self.step],
            flat_init_vals_no_qubits,
            body_jaxpr,
            body_consts,
            body_tree,
        )

        # Create tracers for any non-qreg return values (if there are any).
        ret_vals, _ = tree_unflatten(body_tree, body_jaxpr.out_avals)
        a, t = tree_flatten(ret_vals)
        return ctx.jax_tape.create_tracer(t, a)

    def _call_with_classical_ctx(self, *args):
        # Insert iteration counter into loop body arguments with the type of the lower bound.
        args = (self.lower_bound, *args)

        body_jaxpr, body_consts, body_tree = ForLoopCallable._create_jaxpr(args, self.body_fn)

        flat_init_vals_no_qubits = tree_flatten(args)[0]

        inputs = (
            [self.lower_bound, self.upper_bound, self.step] + body_consts + flat_init_vals_no_qubits
        )
        ret_tree_flat = jprim.qfor(body_jaxpr, len(body_consts), *inputs)
        return tree_unflatten(body_tree, ret_tree_flat)

    def _call_during_trace(self, *args):
        TracingContext.check_is_tracing("Must use 'for_loop' inside tracing context.")

        ctx = qml.QueuingManager.active_context()
        if ctx is None:
            return self._call_with_classical_ctx(*args)
        return self._call_with_quantum_ctx(ctx, *args)

    def _call_during_interpretation(self, *args):
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None

        for i in range(self.lower_bound, self.upper_bound, self.step):
            fn_res = self.body_fn(i, *args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()

        return fn_res

    def __call__(self, *args):
        is_tracing = TracingContext.is_tracing()
        if is_tracing:
            return self._call_during_trace(*args)

        return self._call_during_interpretation(*args)


def for_loop(lower_bound, upper_bound, step):
    """A :func:`~.qjit` compatible for-loop decorator for PennyLane/Catalyst.

    This for-loop representation is a functional version of the traditional
    for-loop, similar to ``jax.cond.fori_loop``. That is, any variables that
    are modified across iterations need to be provided as inputs/outputs to
    the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration.

    - output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the transformed
    function.

    This form of control flow can also be called from the Python interpreter without needing to use
    :func:`~.qjit`.

    The semantics of ``for_loop`` are given by the following Python pseudo-code:

    .. code-block:: python

        def for_loop(lower_bound, upper_bound, step, loop_fn, *args):
            for i in range(lower_bound, upper_bound, step):
                args = loop_fn(i, *args)
            return args

    Unlike ``jax.cond.fori_loop``, the step can be negative if it is known at tracing time
    (i.e. constant). If a non-constant negative step is used, the loop will produce no iterations.

    Args:
        lower_bound (int): starting value of the iteration index
        upper_bound (int): (exclusive) upper bound of the iteration index
        step (int): increment applied to the iteration index at the end of each iteration

    Returns:
        Callable[[int, ...], ...]: A wrapper around the loop body function.
        Note that the loop body function must always have the iteration index as its first argument,
        which can be used arbitrarily inside the loop body. As the value of the index across
        iterations is handled automatically by the provided loop bounds, it must not be returned
        from the function.

    **Example**


    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(n: int, x: float):

            def loop_rx(i, x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)

                # update the value of x for the next iteration
                return jnp.sin(x)

            # apply the for loop
            final_x = for_loop(0, n, 1)(loop_rx)(x)

            return qml.expval(qml.PauliZ(0)), final_x

    >>> circuit(7, 1.6)
    [array(0.97926626), array(0.55395718)]
    """

    def _for_loop(body_fn):
        return ForLoopCallable(lower_bound, upper_bound, step, body_fn)

    return _for_loop


class MidCircuitMeasure(Operation):
    """Operation representing a mid-circuit measurement."""

    num_wires = 1

    def __init__(self, measurement_id, *args, **kwargs):
        self.measurement_id = measurement_id
        super().__init__(*args, **kwargs)


def measure(wires):
    """A :func:`qjit` compatible mid-circuit measurement for PennyLane/Catalyst.

    .. important::

        The :func:`qml.measure() <pennylane.measure>` function is **not** QJIT
        compatible and :func:`catalyst.measure` from Catalyst should be used instead.

    Args:
        wires (Wires): The wire of the qubit the measurement process applies to

    Returns:
        A JAX tracer for the mid-circuit measurement.

    Raises:
        ValueError: Called outside the tape context.

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = measure(wires=0)

            qml.RX(m1 * jnp.pi, wires=1)
            m2 = measure(wires=1)

            qml.RZ(m2 * jnp.pi / 2, wires=0)
            return qml.expval(qml.PauliZ(0)), m2

    >>> circuit(0.43)
    [array(1.), array(False)]
    >>> circuit(0.43)
    [array(-1.), array(True)]
    """
    TracingContext.check_is_tracing(
        "catalyst.measure can only be used from within @qjit decorated code."
    )

    ctx = qml.QueuingManager.active_context()
    if ctx is None:
        raise CompileError("catalyst.measure can only be used from within a qml.qnode.")

    measurement_id = str(uuid.uuid4())[:8]
    MidCircuitMeasure(measurement_id, wires=wires)

    a, t = tree_flatten(jax.core.get_aval(True))
    return ctx.jax_tape.create_tracer(t, a)


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
