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

# pylint: disable=too-many-lines

import copy
import numbers
import pathlib
from collections.abc import Sequence, Sized
from functools import update_wrapper
from typing import Any, Callable, Iterable, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax._src.api_util import shaped_abstractify
from jax._src.lax.lax import _abstractify
from jax._src.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_leaves,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jax.core import eval_jaxpr, get_aval
from pennylane import QNode, QueuingManager
from pennylane.operation import Operator
from pennylane.ops.op_math.controlled import create_controlled_op
from pennylane.tape import QuantumTape

import catalyst
from catalyst.jax_extras import (  # infer_output_type3,
    ClosedJaxpr,
    DynamicJaxprTracer,
    Jaxpr,
    ShapedArray,
    _initial_style_jaxpr,
    _input_type_to_tracers,
    convert_constvars_jaxpr,
    deduce_avals,
    get_implicit_and_explicit_flat_args,
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    new_inner_tracer,
    unzip2,
)
from catalyst.jax_primitives import (
    AbstractQreg,
    GradParams,
    adjoint_p,
    cond_p,
    expval_p,
    for_p,
    func_p,
    grad_p,
    jvp_p,
    probs_p,
    qmeasure_p,
    vjp_p,
    while_p,
    zne_p,
)
from catalyst.jax_tracer import (
    Function,
    HybridOp,
    HybridOpRegion,
    QRegPromise,
    has_nested_tapes,
    trace_quantum_function,
    trace_quantum_tape,
    unify_result_types,
)
from catalyst.qjit_device import QJITDevice
from catalyst.tracing.contexts import (
    EvaluationContext,
    EvaluationMode,
    JaxTracingContext,
)
from catalyst.utils.exceptions import DifferentiableCompileError
from catalyst.utils.runtime import extract_backend_info, get_lib_path


def _check_no_measurements(tape: QuantumTape) -> None:
    """Check the nested quantum tape for the absense of quantum measurements of any kind"""

    msg = "Quantum measurements are not allowed"

    if len(tape.measurements) > 0:
        raise ValueError(msg)
    for op in tape.operations:
        if has_nested_tapes(op):
            for r in [r for r in op.regions if r.quantum_tape is not None]:
                _check_no_measurements(r.quantum_tape)
        else:
            if isinstance(op, MidCircuitMeasure):
                raise ValueError(msg)


class QFunc:
    """A device specific quantum function.

    Args:
        qfunc (Callable): the quantum function
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values
        device (a derived class from QubitDevice): a device specification which determines
            the valid gate set for the quantum function
    """

    def __init__(self, fn, device):  # pragma: nocover
        self.func = fn
        self.device = device
        update_wrapper(self, fn)

    @staticmethod
    def _add_toml_file(device):
        """Temporary function. This function adds the config field to devices.
        TODO: Remove this function when `qml.Device`s are guaranteed to have their own
        config file field."""
        if hasattr(device, "config"):  # pragma: no cover
            # Devices that already have a config field do not need it to be overwritten.
            return
        device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        name = device.name
        if isinstance(device, qml.Device):
            name = device.short_name

        # The toml files name convention we follow is to replace
        # the dots with underscores in the device short name.
        toml_file_name = name.replace(".", "_") + ".toml"
        # And they are currently saved in the following directory.
        toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name
        device.config = toml_file

    @staticmethod
    def extract_backend_info(device):
        """Wrapper around extract_backend_info in the runtime module."""
        return extract_backend_info(device)

    def __call__(self, *args, **kwargs):
        qnode = None
        if isinstance(self, qml.QNode):
            qnode = self
            QFunc._add_toml_file(self.device)
            dev_args = QFunc.extract_backend_info(self.device)
            config, rest = dev_args[0], dev_args[1:]
            device = QJITDevice(config, self.device.shots, self.device.wires, *rest)
        else:  # pragma: nocover
            # Allow QFunc to still be used by itself for internal testing.
            device = self.device

        def _eval_quantum(*args):
            closed_jaxpr, out_type, out_tree = trace_quantum_function(
                self.func, device, args, kwargs, qnode
            )
            args_expanded = get_implicit_and_explicit_flat_args(None, *args)
            res_expanded = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_expanded)
            _, out_keep = unzip2(out_type)
            res_flat = [r for r, k in zip(res_expanded, out_keep) if k]
            return tree_unflatten(out_tree, res_flat)

        flattened_fun, _, _, out_tree_promise = deduce_avals(_eval_quantum, args, {})
        args_flat = tree_flatten(args)[0]
        res_flat = func_p.bind(flattened_fun, *args_flat, fn=self)
        return tree_unflatten(out_tree_promise(), res_flat)


def qfunc(device):
    """A Device specific quantum function.

    Args:
        device (a derived class from QubitDevice): A device specification which determines
            the valid gate set for the quantum function.
        fn (Callable): the quantum function

    Returns:
        Grad: A QFunc object that denotes the the declaration of a quantum function.

    """

    assert device is not None

    def dec_no_params(fn):
        return QFunc(fn, device)

    return dec_no_params


Differentiable = Union[Function, QNode]
DifferentiableLike = Union[Differentiable, Callable, "catalyst.QJIT"]


def _ensure_differentiable(f: DifferentiableLike) -> Differentiable:
    """Narrows down the set of the supported differentiable objects."""

    # Unwrap the function from an existing QJIT object.
    if isinstance(f, catalyst.QJIT):
        f = f.user_function

    if isinstance(f, (Function, QNode)):
        return f
    elif isinstance(f, Callable):  # Keep at the bottom
        return Function(f)

    raise DifferentiableCompileError(f"Non-differentiable object passed: {type(f)}")


def _make_jaxpr_check_differentiable(f: Differentiable, grad_params: GradParams, *args) -> Jaxpr:
    """Gets the jaxpr of a differentiable function. Perform the required additional checks and
    return the output tree."""
    method = grad_params.method
    jaxpr, shape = jax.make_jaxpr(f, return_shape=True)(*args)
    _, out_tree = tree_flatten(shape)
    assert len(jaxpr.eqns) == 1, "Expected jaxpr consisting of a single function call."
    assert jaxpr.eqns[0].primitive == func_p, "Expected jaxpr consisting of a single function call."

    for pos, arg in enumerate(jaxpr.in_avals):
        if arg.dtype.kind != "f" and pos in grad_params.expanded_argnum:
            raise DifferentiableCompileError(
                "Catalyst.grad/jacobian only supports differentiation on floating-point "
                f"arguments, got '{arg.dtype}' at position {pos}."
            )

    if grad_params.scalar_out:
        if not (len(jaxpr.out_avals) == 1 and jaxpr.out_avals[0].shape == ()):
            raise DifferentiableCompileError(
                f"Catalyst.grad only supports scalar-output functions, got {jaxpr.out_avals}"
            )

    for pos, res in enumerate(jaxpr.out_avals):
        if res.dtype.kind != "f":
            raise DifferentiableCompileError(
                "Catalyst.grad/jacobian only supports differentiation on floating-point "
                f"results, got '{res.dtype}' at position {pos}."
            )

    _verify_differentiable_child_qnodes(jaxpr, method)

    return jaxpr, out_tree


def _verify_differentiable_child_qnodes(jaxpr, method):
    """Traverse QNodes being differentiated in the 'call graph' of the JAXPR to verify them."""
    visited = set()

    def traverse_children(jaxpr):
        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            if primitive is func_p:
                child_jaxpr = eqn.params.get("call_jaxpr")
            elif primitive is grad_p:
                child_jaxpr = eqn.params.get("jaxpr")
            else:
                continue

            _check_primitive_is_differentiable(primitive, method)

            py_callable = eqn.params.get("fn")
            if py_callable not in visited:
                if isinstance(py_callable, QNode):
                    _check_qnode_against_grad_method(py_callable, method, child_jaxpr)
                traverse_children(child_jaxpr)
                visited.add(py_callable)

    traverse_children(jaxpr)


def _check_primitive_is_differentiable(primitive, method):
    """Verify restriction on primitives in the call graph of a Grad operation."""

    if primitive is grad_p and method != "fd":
        raise DifferentiableCompileError(
            "Only finite difference can compute higher order derivatives."
        )


def _check_qnode_against_grad_method(f: QNode, method: str, jaxpr: Jaxpr):
    """Additional checks for the given jaxpr of a differentiable function."""
    if method == "fd":
        return

    return_ops = []
    for res in jaxpr.outvars:
        for eq in reversed(jaxpr.eqns):  # pragma: no branch
            if res in eq.outvars:
                return_ops.append(eq.primitive)
                break

    if f.diff_method is None:
        raise DifferentiableCompileError(
            "Cannot differentiate a QNode explicitly marked non-differentiable (with "
            "diff_method=None)."
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


# pylint: disable=too-many-arguments
def _check_grad_params(
    method: str,
    scalar_out: bool,
    h: Optional[float],
    argnum: Optional[Union[int, List[int]]],
    len_flatten_args: int,
    in_tree: PyTreeDef,
) -> GradParams:
    """Check common gradient parameters and produce a class``GradParams`` object"""
    methods = {"fd", "auto"}
    if method is None:
        method = "auto"
    if method not in methods:
        raise ValueError(
            f"Invalid differentiation method '{method}'. "
            f"Supported methods are: {' '.join(sorted(methods))}"
        )
    if method == "fd" and h is None:
        h = 1e-7
    if not (h is None or isinstance(h, numbers.Number)):
        raise ValueError(f"Invalid h value ({h}). None or number was expected.")
    if argnum is None:
        argnum_list = [0]
    elif isinstance(argnum, int):
        argnum_list = [argnum]
    elif isinstance(argnum, tuple):
        argnum_list = list(argnum)
    elif isinstance(argnum, list) and all(isinstance(i, int) for i in argnum):
        argnum_list = argnum
    else:
        raise ValueError(f"argnum should be integer or a list of integers, not {argnum}")
    # Compute the argnums of the pytree arg
    total_argnums = list(range(0, len_flatten_args))
    argnum_unflatten = tree_unflatten(in_tree, total_argnums)
    argnum_selected = [argnum_unflatten[i] for i in argnum_list]
    argnum_expanded, _ = tree_flatten(argnum_selected)
    scalar_argnum = isinstance(argnum, int) or argnum is None
    return GradParams(method, scalar_out, h, argnum_list, scalar_argnum, argnum_expanded)


def _unflatten_derivatives(results, in_tree, out_tree, grad_params, num_results):
    """Unflatten the flat list of derivatives results given the out tree."""
    num_trainable_params = len(grad_params.expanded_argnum)
    results_final = []

    for i in range(0, num_results):
        intermediate_results = results[
            i * num_trainable_params : i * num_trainable_params + num_trainable_params
        ]
        intermediate_results = tree_unflatten(in_tree, intermediate_results)
        if grad_params.scalar_argnum:
            intermediate_results = intermediate_results[0]
        else:
            intermediate_results = tuple(intermediate_results)
        results_final.append(intermediate_results)

    results_final = tree_unflatten(out_tree, results_final)
    return results_final


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

    def __init__(self, fn: Differentiable, grad_params: GradParams):
        self.fn = fn
        self.__name__ = f"grad.{getattr(fn, '__name__', 'unknown')}"
        self.grad_params = grad_params

    def __call__(self, *args, **kwargs):
        """Specifies that an actual call to the differentiated function.
        Args:
            args: the arguments to the differentiated function
        """

        if EvaluationContext.is_tracing():
            fn = _ensure_differentiable(self.fn)

            args_data, in_tree = tree_flatten(args)
            grad_params = _check_grad_params(
                self.grad_params.method,
                self.grad_params.scalar_out,
                self.grad_params.h,
                self.grad_params.argnum,
                len(args_data),
                in_tree,
            )
            jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *args)
            args_argnum = tuple(args[i] for i in grad_params.argnum)
            _, in_tree = tree_flatten(args_argnum)

            # It always returns list as required by catalyst control-flows
            results = grad_p.bind(*args_data, jaxpr=jaxpr, fn=fn, grad_params=grad_params)

            results = _unflatten_derivatives(
                results, in_tree, out_tree, grad_params, len(jaxpr.out_avals)
            )
        else:
            if argnums := self.grad_params.argnum is None:
                argnums = 0
            if self.grad_params.scalar_out:
                results = jax.grad(self.fn, argnums=argnums)(*args)
            else:
                results = jax.jacobian(self.fn, argnums=argnums)(*args)

        return results


def grad(f: DifferentiableLike, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible gradient transformation for PennyLane/Catalyst.

    This function allows the gradient of a hybrid quantum-classical function to be computed within
    the compiled program. Outside of a compiled function, this function will simply dispatch to its
    JAX counterpart ``jax.grad``. The function ``f`` can return any pytree-like shape.

    .. warning::

        Currently, higher-order differentiation is only supported by the finite-difference
        method.

    Args:
        f (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` is not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the gradient of the wrapped function for the given
                  arguments.

    Raises:
        ValueError: Invalid method or step size parameters.
        DifferentiableCompilerError: Called on a function that doesn't return a single scalar.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``grad`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    .. seealso:: :func:`~.jacobian`

    **Example 1 (Classical preprocessing)**

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

    **Example 2 (Classical preprocessing and postprocessing)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def grad_loss(theta):
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qml.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return catalyst.grad(loss, method="auto")(theta)

    >>> grad_loss(1.0)
    array(-1.90958669)

    **Example 3 (Multiple QNodes with their own differentiation methods)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def grad_loss(theta):
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit_A(params):
                qml.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
                return qml.probs()

            @qml.qnode(dev, diff_method="adjoint")
            def circuit_B(params):
                qml.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(params):
                return jnp.prod(circuit_A(params)) + circuit_B(params)

            return catalyst.grad(loss)(theta)

    >>> grad_loss(jnp.array([1.0, 2.0]))
    array([ 0.57367285, 44.4911605 ])

    **Example 4 (Purely classical functions)**

    .. code-block:: python

        def square(x: float):
            return x ** 2

        @qjit
        def dsquare(x: float):
            return catalyst.grad(square)(x)

    >>> dsquare(2.3)
    array(4.6)
    """
    scalar_out = True
    return Grad(f, GradParams(method, scalar_out, h, argnum))


def jacobian(f: DifferentiableLike, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian transformation for PennyLane/Catalyst.

    This function allows the Jacobian of a hybrid quantum-classical function to be computed within
    the compiled program. Outside of a compiled function, this function will simply dispatch to its
    JAX counterpart ``jax.jacobian``. The function ``f`` can return any pytree-like shape.

    Args:
        f (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` is not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the Jacobian of the wrapped function for the given
                  arguments.

    Raises:
        ValueError: Invalid method or step size parameters.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``jacobian`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    .. seealso:: :func:`~.grad`

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1], wires=0)
                return qml.probs()

            g = jacobian(circuit)
            return g(x)

    >>> workflow(jnp.array([2.0, 1.0]))
    array([[ 3.48786850e-16 -4.20735492e-01]
           [-8.71967125e-17  4.20735492e-01]])
    """
    scalar_out = False
    return Grad(f, GradParams(method, scalar_out, h, argnum))


# pylint: disable=too-many-arguments
def jvp(f: DifferentiableLike, params, tangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian-vector product for PennyLane/Catalyst.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program. Outside of a compiled function, this function will simply
    dispatch to its JAX counterpart ``jax.jvp``. The function ``f`` can return any pytree-like
    shape.

    Args:
        f (Callable): Function-like object to calculate JVP for
        params (List[Array]): List (or a tuple) of the function arguments specifying the point
                              to calculate JVP at. A subset of these parameters are declared as
                              differentiable by listing their indices in the ``argnum`` parameter.
        tangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size and
                               shapes must match the ones of differentiable params.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Any]: Return values of ``f`` paired with the JVP values.

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
    (array([0.09983342, 0.04      , 0.02      ]), array([0.29850125, 0.24      , 0.12      ]))

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
    (array(0.78766064), array(-0.7011436))
    """

    def check_is_iterable(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")

    check_is_iterable(params, "params")
    check_is_iterable(tangents, "tangents")

    if EvaluationContext.is_tracing():
        scalar_out = False
        fn = _ensure_differentiable(f)
        args_flatten, in_tree = tree_flatten(params)
        tangents_flatten, _ = tree_flatten(tangents)
        grad_params = _check_grad_params(method, scalar_out, h, argnum, len(args_flatten), in_tree)

        jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *params)

        results = jvp_p.bind(
            *args_flatten, *tangents_flatten, jaxpr=jaxpr, fn=fn, grad_params=grad_params
        )

        midpoint = len(results) // 2
        func_res = results[:midpoint]
        jvps = results[midpoint:]

        func_res = tree_unflatten(out_tree, func_res)
        jvps = tree_unflatten(out_tree, jvps)
        results = (func_res, jvps)

    else:
        results = jax.jvp(f, params, tangents)

    return results


# pylint: disable=too-many-arguments
def vjp(f: DifferentiableLike, params, cotangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product for PennyLane/Catalyst.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program. Outside of a compiled function, this function will simply
    dispatch to its JAX counterpart ``jax.vjp``. The function ``f`` can return any pytree-like
    shape.

    Args:
        f(Callable): Function-like object to calculate JVP for
        params(List[Array]): List (or a tuple) of f's arguments specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnum`` parameter.
        cotangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size
                                 and shapes must match the size and shape of ``f`` outputs.
        method(str): Differentiation method to use, same as in ``grad``.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Any]): Return values of ``f`` paired with the VJP values.

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
    (array([0.09983342, 0.04      , 0.02      ]), (array([-0.43750208,  0.07      ]),))
    """

    def check_is_iterable(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")

    check_is_iterable(params, "params")
    check_is_iterable(cotangents, "cotangents")

    if EvaluationContext.is_tracing():
        scalar_out = False
        fn = _ensure_differentiable(f)

        args_flatten, in_tree = tree_flatten(params)
        cotangents_flatten, _ = tree_flatten(cotangents)

        grad_params = _check_grad_params(method, scalar_out, h, argnum, len(args_flatten), in_tree)

        args_argnum = tuple(params[i] for i in grad_params.argnum)
        _, in_tree = tree_flatten(args_argnum)

        jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *params)

        cotangents, _ = tree_flatten(cotangents)

        results = vjp_p.bind(
            *args_flatten, *cotangents_flatten, jaxpr=jaxpr, fn=fn, grad_params=grad_params
        )

        func_res = results[: len(jaxpr.out_avals)]
        vjps = results[len(jaxpr.out_avals) :]
        func_res = tree_unflatten(out_tree, func_res)
        vjps = tree_unflatten(in_tree, vjps)

        results = (func_res, vjps)

    else:
        if isinstance(params, jax.numpy.ndarray) and params.ndim == 0:
            primal_outputs, vjp_fn = jax.vjp(f, params)
        else:
            primal_outputs, vjp_fn = jax.vjp(f, *params)

        results = (primal_outputs, vjp_fn(cotangents))

    return results


class ZNE:
    """An object that specifies how a circuit is mitigated with ZNE.

    Args:
        fn (Callable): the circuit to be mitigated with ZNE.
        scale_factors (array[int]): the range of noise scale factors used.
        deg (int): the degree of the polymonial used for fitting.

    Raises:
        TypeError: Non-QNode object was passed as `fn`.
    """

    def __init__(self, fn: Callable, scale_factors: jnp.ndarray, deg: int):
        if not isinstance(fn, qml.QNode):
            raise TypeError(f"A QNode is expected, got the classical function {fn}")
        self.fn = fn
        self.__name__ = f"zne.{getattr(fn, '__name__', 'unknown')}"
        self.scale_factors = scale_factors
        self.deg = deg

    def __call__(self, *args, **kwargs):
        """Specifies the an actual call to the folded circuit."""
        jaxpr = jaxpr = jax.make_jaxpr(self.fn)(*args)
        shapes = [out_val.shape for out_val in jaxpr.out_avals]
        dtypes = [out_val.dtype for out_val in jaxpr.out_avals]
        set_dtypes = set(dtypes)
        if any(shapes):
            raise TypeError("Only expectations values and classical scalar values can be returned.")
        if len(set_dtypes) != 1 or set_dtypes.pop().kind != "f":
            raise TypeError("All expectation and classical values dtypes must match and be float.")
        args_data, _ = tree_flatten(args)
        results = zne_p.bind(*args_data, self.scale_factors, jaxpr=jaxpr, fn=self.fn)
        float_scale_factors = jnp.array(self.scale_factors, dtype=float)
        results = jnp.polyfit(float_scale_factors, results[0], self.deg)[-1]
        # Single measurement
        if results.shape == ():
            return results
        # Multiple measurements
        return tuple(res for res in results)


def mitigate_with_zne(f, *, scale_factors: jnp.ndarray, deg: int = None):
    """A :func:`~.qjit` compatible error mitigation of an input circuit using zero-noise
    extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    Args:
        f (qml.QNode): the circuit to be mitigated.
        scale_factors (array[int]): the range of noise scale factors used.
        deg (int): the degree of the polymonial used for fitting.

    Returns:
        Callable: A callable object that computes the mitigated of the wrapped :class:`qml.QNode`
        for the given arguments.

    **Example:**

    For example, given a noisy device (such as noisy hardware available through Amazon Braket):

    .. code-block:: python

        # replace "noisy.device" with your noisy device
        dev = qml.device("noisy.device", wires=2)

        @qml.qnode(device=dev)
        def circuit(x, n):
            @for_loop(0, n, 1)
            def loop_rx(i):
                qml.RX(x, wires=0)

            loop_rx()

            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            loop_rx()
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        @qjit
        def mitigated_circuit(args, n):
            s = jax.numpy.array([1, 2, 3])
            return mitigate_with_zne(circuit, scale_factors=s)(args, n)
    """
    if deg is None:
        deg = len(scale_factors) - 1
    return ZNE(f, scale_factors, deg)


def _aval_to_primitive_type(aval):
    if isinstance(aval, DynamicJaxprTracer):
        aval = aval.strip_weak_type()
    if isinstance(aval, ShapedArray):
        aval = aval.dtype
    assert not isinstance(aval, (list, dict)), f"Unexpected type {aval}"
    return aval


def _check_single_bool_value(tree: PyTreeDef, avals: List[Any]) -> None:
    if not treedef_is_leaf(tree):
        raise TypeError(
            f"A single boolean scalar was expected, got the value of tree-shape: {tree}."
        )
    assert len(avals) == 1, f"{avals} does not match {tree}"
    dtype = _aval_to_primitive_type(avals[0])
    if dtype not in (bool, jnp.bool_):
        raise TypeError(f"A single boolean scalar was expected, got the value {avals[0]}.")


def _check_cond_same_shapes(trees: List[PyTreeDef], avals: List[List[Any]]) -> None:
    assert len(trees) == len(avals), f"Input trees ({trees}) don't match input avals ({avals})"
    expected_tree = trees[0]
    for tree in list(trees)[1:]:
        if tree != expected_tree:
            raise TypeError("Conditional requires consistent return types across all branches")


class ForLoop(HybridOp):
    """PennyLane ForLoop Operation."""

    binder = for_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        inner_trace = op.regions[0].trace
        inner_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers

        with EvaluationContext.frame_tracing_context(ctx, inner_trace):
            qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
            qreg_out = qrp_out.actualize()
            jaxpr, _, consts = ctx.frames[inner_trace].to_jaxpr2(res_classical_tracers + [qreg_out])

        step = op.in_classical_tracers[2]
        apply_reverse_transform = isinstance(step, int) and step < 0
        qreg = qrp.actualize()
        qrp2 = QRegPromise(
            op.bind_overwrite_classical_tracers(
                ctx,
                trace,
                op.in_classical_tracers[0],
                op.in_classical_tracers[1],
                step,
                *(consts + op.in_classical_tracers[3:] + [qreg]),
                body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                body_nconsts=len(consts),
                apply_reverse_transform=apply_reverse_transform,
            )
        )
        return qrp2


class MidCircuitMeasure(HybridOp):
    """Operation representing a mid-circuit measurement."""

    binder = qmeasure_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        wire = op.in_classical_tracers[0]
        qubit = qrp.extract([wire])[0]
        postselect = op.in_classical_tracers[1]

        qubit2 = op.bind_overwrite_classical_tracers(ctx, trace, qubit, postselect=postselect)
        qrp.insert([wire], [qubit2])
        return qrp


class Cond(HybridOp):
    """PennyLane's conditional operation."""

    binder = cond_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        jaxprs, consts = [], []
        op = self
        for region in op.regions:
            with EvaluationContext.frame_tracing_context(ctx, region.trace):
                qreg_in = _input_type_to_tracers(region.trace.new_arg, [AbstractQreg()])[0]
                qrp_out = trace_quantum_tape(
                    region.quantum_tape, device, qreg_in, ctx, region.trace
                )
                qreg_out = qrp_out.actualize()
                jaxpr, _, const = ctx.frames[region.trace].to_jaxpr2(
                    region.res_classical_tracers + [qreg_out]
                )
                jaxprs.append(jaxpr)
                consts.append(const)

        jaxprs2, combined_consts = initial_style_jaxprs_with_common_consts2(jaxprs, consts)

        qreg = qrp.actualize()
        qrp2 = QRegPromise(
            op.bind_overwrite_classical_tracers(
                ctx,
                trace,
                *(op.in_classical_tracers + combined_consts + [qreg]),
                branch_jaxprs=unify_result_types(jaxprs2),
            )
        )
        return qrp2


class WhileLoop(HybridOp):
    """PennyLane's while loop operation."""

    binder = while_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        cond_trace = self.regions[0].trace
        res_classical_tracers = self.regions[0].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, cond_trace):
            _input_type_to_tracers(cond_trace.new_arg, [AbstractQreg()])
            cond_jaxpr, _, cond_consts = ctx.frames[cond_trace].to_jaxpr2(res_classical_tracers)

        body_trace = self.regions[1].trace
        body_tape = self.regions[1].quantum_tape
        res_classical_tracers = self.regions[1].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, body_trace):
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
            qreg_out = qrp_out.actualize()
            body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        qreg = qrp.actualize()
        qrp2 = QRegPromise(
            self.bind_overwrite_classical_tracers(
                ctx,
                trace,
                *(cond_consts + body_consts + self.in_classical_tracers + [qreg]),
                cond_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(cond_jaxpr), ()),
                body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                cond_nconsts=len(cond_consts),
                body_nconsts=len(body_consts),
            )
        )
        return qrp2


class Adjoint(HybridOp):
    """PennyLane's adjoint operation"""

    binder = adjoint_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        body_trace = op.regions[0].trace
        body_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, body_trace):
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
            qreg_out = qrp_out.actualize()
            body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        qreg = qrp.actualize()
        args, args_tree = tree_flatten((body_consts, op.in_classical_tracers, [qreg]))
        op_results = adjoint_p.bind(
            *args,
            args_tree=args_tree,
            jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
        )
        qrp2 = QRegPromise(op_results[-1])
        return qrp2

    @property
    def wires(self):
        """The list of all static wires."""

        assert len(self.regions) == 1, "Adjoint is expected to have one region"
        total_wires = sum((op.wires for op in self.regions[0].quantum_tape.operations), [])
        return total_wires


# TODO: This class needs to be made interoperable with qml.Controlled since qml.ctrl dispatches
#       to this class whenever a qjit context is active.
class QCtrl(HybridOp):
    """Catalyst quantum ctrl operation"""

    def __init__(self, *args, control_wires, control_values=None, work_wires=None, **kwargs):
        self._control_wires = qml.wires.Wires(control_wires)
        self._work_wires = qml.wires.Wires([] if work_wires is None else work_wires)
        if control_values is None:
            self._control_values = [True] * len(self._control_wires)
        elif isinstance(control_values, (int, bool)):
            self._control_values = [control_values]
        else:
            self._control_values = control_values

        super().__init__(*args, **kwargs)

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        raise NotImplementedError("QCtrl does not support JAX quantum tracing")  # pragma: no cover

    def decomposition(self):
        """Compute quantum decomposition of the gate by recursively scanning the nested tape and
        distributing the quantum control operaiton over the tape operations."""
        assert len(self.regions) == 1, "Qctrl is expected to have one region"

        _check_no_measurements(self.regions[0].quantum_tape)
        new_tape = qctrl_distribute(
            self.regions[0].quantum_tape,
            self._control_wires,
            self._control_values,
            self._work_wires,
        )
        return new_tape.operations

    @property
    def wires(self):
        """The list of all control-wires, work-wires, and active-wires."""
        assert len(self.regions) == 1, "Qctrl is expected to have one region"

        total_wires = sum(
            (op.wires for op in self.regions[0].quantum_tape.operations),
            self._control_wires,
        )
        total_wires += self._work_wires
        return total_wires

    @property
    def control_wires(self):
        """Wires used in quantum conditioning."""
        return self._control_wires

    @property
    def control_values(self):
        """(Boolean) Values upon which to condition on."""
        return self._control_values

    @property
    def work_wires(self):
        """Optional wires that can be used in the expansion of this op."""
        return self._work_wires

    def map_wires(self, wire_map):
        """Map wires to new wires according to wire_map"""
        new_ops = []
        for op in self.regions[0].quantum_tape.operations:
            new_ops.append(op.map_wires(wire_map))
        self.regions[0].quantum_tape = QuantumTape(new_ops, [])
        self._control_wires = [wire_map.get(wire, wire) for wire in self._control_wires]
        self._work_wires = [wire_map.get(wire, wire) for wire in self._work_wires]
        return self


def qctrl_distribute(
    tape: QuantumTape,
    control_wires: List[Any],
    control_values: List[Any],
    work_wires: Optional[List[Any]] = None,
) -> QuantumTape:
    """Distribute the quantum control operation, described by ``control_wires`` and
    ``control_values``, over all the operations on the nested quantum tape.
    """
    # Note: The transformation modifies operations in the source quantum tape, so we must not use it
    # after we called this function.
    assert len(control_wires) > 0, "This transformation expects a non-empty list of control_wires"
    assert len(control_wires) == len(control_values), (
        f"Length of the control_values ({len(control_values)}) must be equal "
        f"to the lenght of control_wires ({len(control_wires)})"
    )
    ctx = EvaluationContext.get_main_tracing_context()
    ops2 = []
    for op in tape.operations:
        if has_nested_tapes(op):
            if isinstance(op, QCtrl):
                for region in [region for region in op.regions if region.quantum_tape is not None]:
                    tape2 = qctrl_distribute(
                        region.quantum_tape,
                        control_wires + op.control_wires,
                        control_values + op.control_values,
                        work_wires + op.work_wires,
                    )
                    ops2.extend(tape2.operations)
            else:
                for region in [region for region in op.regions if region.quantum_tape is not None]:
                    with EvaluationContext.frame_tracing_context(ctx, region.trace):
                        region.quantum_tape = qctrl_distribute(
                            region.quantum_tape, control_wires, control_values, work_wires
                        )
                ops2.append(op)
        else:
            ops2.append(
                create_controlled_op(
                    copy.copy(op),
                    control=control_wires,
                    control_values=control_values,
                    work_wires=work_wires,
                )
            )
    return QuantumTape(ops2, tape.measurements)


class CondCallable:
    """User-facing wrapper provoding "else_if" and "otherwise" public methods.
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

    def _call_with_quantum_ctx(self, ctx):
        outer_trace = ctx.trace
        in_classical_tracers = self.preds
        regions: List[HybridOpRegion] = []

        out_trees, out_avals = [], []
        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
                wffa, _, _, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))
            out_trees.append(out_tree())
            out_avals.append(res_classical_tracers)

        _check_cond_same_shapes(out_trees, out_avals)
        res_avals = list(map(shaped_abstractify, res_classical_tracers))
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]
        Cond(in_classical_tracers, out_classical_tracers, regions)
        return tree_unflatten(out_tree(), out_classical_tracers)

    def _call_with_classical_ctx(self):
        args, args_tree = tree_flatten([])
        args_avals = tuple(map(_abstractify, args))
        branch_jaxprs, consts, out_trees = initial_style_jaxprs_with_common_consts1(
            (*self.branch_fns, self.otherwise_fn), args_tree, args_avals, "cond"
        )
        _check_cond_same_shapes(out_trees, [j.out_avals for j in branch_jaxprs])
        branch_jaxprs = unify_result_types(branch_jaxprs)
        out_classical_tracers = cond_p.bind(*(self.preds + consts), branch_jaxprs=branch_jaxprs)
        return tree_unflatten(out_trees[0], out_classical_tracers)

    def _call_during_interpretation(self):
        for pred, branch_fn in zip(self.preds, self.branch_fns):
            if pred:
                return branch_fn()
        return self.otherwise_fn()

    def __call__(self):
        mode, ctx = EvaluationContext.get_evaluation_mode()
        if mode == EvaluationMode.QUANTUM_COMPILATION:
            return self._call_with_quantum_ctx(ctx)
        elif mode == EvaluationMode.CLASSICAL_COMPILATION:
            return self._call_with_classical_ctx()
        else:
            assert mode == EvaluationMode.INTERPRETATION, f"Unsupported evaluation mode {mode}"
            return self._call_during_interpretation()


def cond(pred: DynamicJaxprTracer):
    """A :func:`~.qjit` compatible decorator for if-else conditionals in PennyLane/Catalyst.

    .. note::

        Catalyst can automatically convert Python if-statements for you. Requires setting
        ``autograph=True``, see the :func:`~.qjit` function or documentation page for more details.

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

            return qml.probs(wires=0)

    The conditional function is permitted to also return values.
    Any value that is supported by JAX JIT compilation is supported as a return
    type.

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

    .. details::
        :title: Usage details
        :href: usage-details

        There are various constraints and restrictions that should be kept in mind
        when working with conditionals in Catalyst.

        The return values of all branches of :func:`~.cond` must be the same type.
        Returning different types, or ommitting a return value in one branch (e.g.,
        returning ``None``) but not in others will result in an error.

        >>> @qjit
        ... def f(x: float):
        ...     @cond(x > 1.5)
        ...     def cond_fn():
        ...         return x ** 2  # float
        ...     @cond_fn.otherwise
        ...     def else_branch():
        ...         return 6  # int
        ...     return cond_fn()
        TypeError: Conditional requires consistent return types across all branches, got:
        - Branch at index 0: [ShapedArray(float64[], weak_type=True)]
        - Branch at index 1: [ShapedArray(int64[], weak_type=True)]
        Please specify an else branch if none was specified.
        >>> @qjit
        ... def f(x: float):
        ...     @cond(x > 1.5)
        ...     def cond_fn():
        ...         return x ** 2  # float
        ...     @cond_fn.otherwise
        ...     def else_branch():
        ...         return 6.  # float
        ...     return cond_fn()
        >>> f(1.5)
        array(6.)

        Similarly, the else (``my_cond_fn.otherwise``) may be omitted **as long as
        other branches do not return any values**. If other branches do return values,
        the else branch must be specified.

        >>> @qjit
        ... def f(x: float):
        ...     @cond(x > 1.5)
        ...     def cond_fn():
        ...         return x ** 2
        ...     return cond_fn()
        TypeError: Conditional requires consistent return types across all branches, got:
        - Branch at index 0: [ShapedArray(float64[], weak_type=True)]
        - Branch at index 1: []
        Please specify an else branch if none was specified.

        >>> @qjit
        ... def f(x: float):
        ...     @cond(x > 1.5)
        ...     def cond_fn():
        ...         return x ** 2
        ...     @cond_fn.otherwise
        ...     def else_branch():
        ...         return x
        ...     return cond_fn()
        >>> f(1.6)
        array(2.56)
    """

    def _decorator(true_fn: Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator


def for_loop(lower_bound, upper_bound, step):
    """A :func:`~.qjit` compatible for-loop decorator for PennyLane/Catalyst.

    .. note::

        Catalyst can automatically convert Python for loop statements for you. Requires setting
        ``autograph=True``, see the :func:`~.qjit` function or documentation page for more details.

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

    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: JaxTracingContext):
                quantum_tape = QuantumTape()
                outer_trace = ctx.trace
                with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
                    in_classical_tracers = [
                        lower_bound,
                        upper_bound,
                        step,
                        lower_bound,
                    ] + tree_flatten(init_state)[0]
                    wffa, in_avals, _, body_tree = deduce_avals(
                        body_fn, [lower_bound] + list(init_state), {}
                    )
                    arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [
                            inner_trace.full_raise(t)
                            for t in wffa.call_wrapped(*arg_classical_tracers)
                        ]

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]
                ForLoop(
                    in_classical_tracers,
                    out_classical_tracers,
                    [
                        HybridOpRegion(
                            inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
                        )
                    ],
                )

                return tree_unflatten(body_tree(), out_classical_tracers)

            def _call_with_classical_ctx():
                iter_arg = lower_bound
                init_vals, in_tree = tree_flatten((iter_arg, *init_state))
                init_avals = tuple(_abstractify(val) for val in init_vals)
                body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
                    body_fn, in_tree, init_avals, "for_loop"
                )

                apply_reverse_transform = isinstance(step, int) and step < 0
                out_classical_tracers = for_p.bind(
                    lower_bound,
                    upper_bound,
                    step,
                    *(body_consts + init_vals),
                    body_jaxpr=body_jaxpr,
                    body_nconsts=len(body_consts),
                    apply_reverse_transform=apply_reverse_transform,
                )

                return tree_unflatten(body_tree, out_classical_tracers)

            def _call_during_interpretation():
                args = init_state
                fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
                for i in range(lower_bound, upper_bound, step):
                    fn_res = body_fn(i, *args)
                    args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
                return fn_res

            mode, ctx = EvaluationContext.get_evaluation_mode()
            if mode == EvaluationMode.QUANTUM_COMPILATION:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.CLASSICAL_COMPILATION:
                return _call_with_classical_ctx()
            else:
                assert mode == EvaluationMode.INTERPRETATION, f"Unsupported evaluation mode {mode}"
                return _call_during_interpretation()

        return _call_handler

    return _body_query


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

    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: JaxTracingContext):
                outer_trace = ctx.trace
                in_classical_tracers, _ = tree_flatten(init_state)

                with EvaluationContext.frame_tracing_context(ctx) as cond_trace:
                    cond_wffa, cond_in_avals, _, cond_tree = deduce_avals(cond_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(
                        cond_trace.new_arg, cond_in_avals
                    )
                    res_classical_tracers = [
                        cond_trace.full_raise(t)
                        for t in cond_wffa.call_wrapped(*arg_classical_tracers)
                    ]
                    cond_region = HybridOpRegion(
                        cond_trace, None, arg_classical_tracers, res_classical_tracers
                    )

                _check_single_bool_value(cond_tree(), res_classical_tracers)

                with EvaluationContext.frame_tracing_context(ctx) as body_trace:
                    wffa, in_avals, _, body_tree = deduce_avals(body_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(body_trace.new_arg, in_avals)
                    quantum_tape = QuantumTape()
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [
                            body_trace.full_raise(t)
                            for t in wffa.call_wrapped(*arg_classical_tracers)
                        ]
                    body_region = HybridOpRegion(
                        body_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
                    )

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

                WhileLoop(in_classical_tracers, out_classical_tracers, [cond_region, body_region])
                return tree_unflatten(body_tree(), out_classical_tracers)

            def _call_with_classical_ctx():
                init_vals, in_tree = tree_flatten(init_state)
                init_avals = tuple(_abstractify(val) for val in init_vals)
                cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
                    cond_fn, in_tree, init_avals, "while_cond"
                )
                body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
                    body_fn, in_tree, init_avals, "while_loop"
                )
                _check_single_bool_value(cond_tree, cond_jaxpr.out_avals)
                out_classical_tracers = while_p.bind(
                    *(cond_consts + body_consts + init_vals),
                    cond_jaxpr=cond_jaxpr,
                    body_jaxpr=body_jaxpr,
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts),
                )
                return tree_unflatten(body_tree, out_classical_tracers)

            def _call_during_interpretation():
                args = init_state
                fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
                while cond_fn(*args):
                    fn_res = body_fn(*args)
                    args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
                return fn_res

            mode, ctx = EvaluationContext.get_evaluation_mode()
            if mode == EvaluationMode.QUANTUM_COMPILATION:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.CLASSICAL_COMPILATION:
                return _call_with_classical_ctx()
            else:
                assert mode == EvaluationMode.INTERPRETATION, f"Unsupported evaluation mode {mode}"
                return _call_during_interpretation()

        return _call_handler

    return _body_query


def measure(
    wires, reset: Optional[bool] = False, postselect: Optional[int] = None
) -> DynamicJaxprTracer:
    """A :func:`qjit` compatible mid-circuit measurement for PennyLane/Catalyst.

    .. important::

        The :func:`qml.measure() <pennylane.measure>` function is **not** QJIT
        compatible and :func:`catalyst.measure` from Catalyst should be used instead.

    Args:
        wires (Wires): The wire of the qubit the measurement process applies to
        reset (Optional[bool]): Whether to reset the wire to the |0⟩ state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit measurement.

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

    **Example with post-selection**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0, postselect=1)
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    -1.0

    **Example with reset**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0, reset=True)
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    1.0
    """
    EvaluationContext.check_is_tracing("catalyst.measure can only be used from within @qjit.")
    EvaluationContext.check_is_quantum_tracing(
        "catalyst.measure can only be used from within a qml.qnode."
    )
    ctx = EvaluationContext.get_main_tracing_context()
    wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
    if len(wires) != 1:
        raise TypeError(f"One classical argument (a wire) is expected, got {wires}")

    # Copy, so wires remain unmodified
    in_classical_tracers = wires.copy()

    if postselect is not None and postselect not in [0, 1]:
        raise TypeError(f"postselect must be '0' or '1', got {postselect}")
    in_classical_tracers.append(postselect)

    # assert len(ctx.trace.frame.eqns) == 0, ctx.trace.frame.eqns
    m = new_inner_tracer(ctx.trace, get_aval(True))
    MidCircuitMeasure(
        in_classical_tracers=in_classical_tracers,
        out_classical_tracers=[m],
        regions=[],
    )

    # If reset was requested, reset qubit only if the measurement result was 1
    if reset:

        @cond(m)
        def reset_fn():
            qml.PauliX(wires=wires)

        reset_fn()

    return m


def adjoint(f: Union[Callable, Operator]) -> Union[Callable, Operator]:
    """A :func:`~.qjit` compatible adjoint transformer for PennyLane/Catalyst.

    Returns a quantum function or operator that applies the adjoint of the
    provided function or operator.

    .. warning::

        This function does not support performing the adjoint
        of quantum functions that contain mid-circuit measurements.

    Args:
        f (Callable or Operator): A PennyLane operation or a Python function
                                  containing PennyLane quantum operations.

    Returns:
        If an Operator is provided, returns an Operator that is the adjoint. If
        a function is provided, returns a function with the same call signature
        that returns the Adjoint of the provided function.

    Raises:
        ValueError: invalid parameter values

    **Example 1 (basic usage)**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow(theta, wires):
            catalyst.adjoint(qml.RZ)(theta, wires=wires)
            catalyst.adjoint(qml.RZ(theta, wires=wires))
            def func():
                qml.RX(theta, wires=wires)
                qml.RY(theta, wires=wires)
            catalyst.adjoint(func)()
            return qml.probs()

    >>> workflow(jnp.pi/2, wires=0)
    array([0.5, 0.5])

    **Example 2 (with Catalyst control flow)**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow(theta, n, wires):
            def func():
                @catalyst.for_loop(0, n, 1)
                def loop_fn(i):
                    qml.RX(theta, wires=wires)

                loop_fn()
            catalyst.adjoint(func)()
            return qml.probs()

    >>> workflow(jnp.pi/2, 3, 0)
    [1.00000000e+00 7.39557099e-32]
    """

    if not EvaluationContext.is_tracing():
        return qml.adjoint(f)

    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.adjoint can only be used from within a qml.qnode."
        )
        ctx = EvaluationContext.get_main_tracing_context()
        with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
            in_classical_tracers, _ = tree_flatten((args, kwargs))
            wffa, in_avals, _, _ = deduce_avals(_callee, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            quantum_tape = QuantumTape()
            with QueuingManager.stop_recording(), quantum_tape:
                # FIXME: move all full_raise calls into a separate function
                res_classical_tracers = [
                    inner_trace.full_raise(t)
                    for t in wffa.call_wrapped(*arg_classical_tracers)
                    if isinstance(t, DynamicJaxprTracer)
                ]

            _check_no_measurements(quantum_tape)

            adjoint_region = HybridOpRegion(
                inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
            )

        return Adjoint(
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=[],
            regions=[adjoint_region],
        )

    if isinstance(f, Callable):

        def _callable(*args, **kwargs):
            return _call_handler(*args, _callee=f, **kwargs)

        return _callable
    elif isinstance(f, Operator):
        QueuingManager.remove(f)

        def _callee():
            QueuingManager.append(f)

        return _call_handler(_callee=_callee)
    else:
        raise ValueError(f"Expected a callable or a qml.Operator, not {f}")


def ctrl(
    f: Union[Callable, Operator],
    control: List[Any],
    control_values: Optional[List[Any]] = None,
    work_wires: Optional[List[Any]] = None,
) -> Callable:
    """Create a method that applies a controlled version of the provided op. This function is the
    Catalyst version of the ``qml.ctrl`` that supports Catalyst hybrid operations such as loops and
    conditionals.

    Args:
        f (Callable or Operator): A PennyLane operation or a Python function
                                  containing PennyLane quantum operations.
        control (Wires): The control wire(s).
        control_values (List[bool], optional): The value(s) the control wire(s) should take.
            Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns a
        Controlled version of the Operator.  If a function is provided, returns a function with the
        same call signature that creates a controlled version of the provided function.

    Raises:
        ValueError: invalid parameter values, measurements are among the controlled operations.

    **Example**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def workflow(theta, w, cw):
            qml.Hadamard(wires=[0])
            qml.Hadamard(wires=[1])

            def func(arg):
              qml.RX(theta, wires=arg)

            @cond(theta > 0.0)
            def cond_fn():
              qml.RY(theta, wires=w)

            catalyst.ctrl(func, control=[cw])(w)
            catalyst.ctrl(cond_fn, control=[cw])()
            catalyst.ctrl(qml.RZ, control=[cw])(theta, wires=w)
            catalyst.ctrl(qml.RY(theta, wires=w), control=[cw])
            return qml.probs()

    >>> workflow(jnp.pi/4, 1, 0)
    array([0.25, 0.25, 0.03661165, 0.46338835])
    """

    if not EvaluationContext.is_tracing():
        return qml.ctrl(f, control, control_values, work_wires)

    if control_values is not None and (
        (len(control) if isinstance(control, Sized) else 1)
        != (len(control_values) if isinstance(control_values, Sized) else 1)
    ):
        raise ValueError(
            f"Length of the control_values ({len(control_values)}) must be None or equal "
            f"to the lenght of control ({len(control)})"
        )

    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.ctrl can only be used from within a qml.qnode."
        )
        in_classical_tracers, _ = tree_flatten((args, kwargs))
        quantum_tape = QuantumTape()
        with QueuingManager.stop_recording(), quantum_tape:
            res = _callee(*args, **kwargs)
        out_classical_tracers, _ = tree_flatten(res)

        _check_no_measurements(quantum_tape)

        region = HybridOpRegion(None, quantum_tape, [], [])

        # Return the operation instance since PL expects this for qml.ctrl(op).
        return QCtrl(
            control_wires=control,
            control_values=control_values,
            work_wires=work_wires,
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=out_classical_tracers,
            regions=[region],
        )

    if isinstance(f, Callable):

        def _callable(*args, **kwargs):
            return _call_handler(*args, _callee=f, **kwargs)

        return _callable

    elif isinstance(f, Operator):
        QueuingManager.remove(f)

        def _callee():
            QueuingManager.append(f)

        return _call_handler(_callee=_callee)

    else:
        raise ValueError(f"Expected a callable or a qml.Operator, not {f}")  # pragma: no cover


def vmap(
    fn: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
    axis_size: Optional[int] = None,
) -> Callable:
    """A :func:`~.qjit` compatible vectorizing map.
    Creates a function which maps an input function over argument axes.

    Args:
        f (Callable): A Python function containing PennyLane quantum operations.
        in_axes (Union[int, Sequence[Any]]): Specifies the value(s) over which input
            array axes to map.
        out_axes (Union[int, Sequence[Any]]): Specifies where the mapped axis should appear
            in the output.
        axis_size (int): An integer can be optionally provided to indicate the size of the
            axis to be mapped. If omitted, the size of the mapped axis will be inferred from
            the provided arguments.

    Returns:
        Callable: Vectorized version of ``fn``.

    Raises:
        ValueError: Invalid ``in_axes``, ``out_axes``, and ``axis_size`` values.

    **Example**

    For example, consider the following QNode:

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
          qml.RX(jnp.pi * x[0] + y, wires=0)
          qml.RY(x[1] ** 2, wires=0)
          qml.RX(x[1] * x[2], wires=0)
          return qml.expval(qml.PauliZ(0))

    >>> circuit(jnp.array([0.1, 0.2, 0.3]), jnp.pi)
    Array(-0.93005586, dtype=float64)

    We can use ``catalyst.vmap`` to introduce additional batch dimensions
    to our input arguments,
    without needing to use a Python for loop:

    >>> x = jnp.array([[0.1, 0.2, 0.3],
    ...                [0.4, 0.5, 0.6],
    ...                [0.7, 0.8, 0.9]])
    >>> y = jnp.array([jnp.pi, jnp.pi / 2, jnp.pi / 4])
    >>> qjit(vmap(cost))(x, y)
    array([-0.93005586, -0.97165424, -0.6987465 ])

    ``catalyst.vmap()`` has been implemented to match the same behaviour of
    ``jax.vmap``, so should be a drop-in replacement in most cases.
    Under-the-hood, it is automatically inserting Catalyst-compatible for loops,
    which will be compiled and executed outside of Python for increased performance.

    Outside of a Catalyst qjit-compiled function, ``vmap`` will simply dispatch to
    ``jax.vmap``.

    .. details::
        :title: Selecting batching axes for arguments

        The ``in_axes`` parameter provides different modes the allow large- and fine-grained
        control over which arguments to apply the batching transformation on. Enabling batching for
        a particular argument requires that the selected axis be of the same size as the determined
        batch size, which is the same for all arguments.

        The following modes are supported:

        - ``int``: Specifies the same batch axis for all arguments
        - ``Tuple[int]``: Specify a different batch axis for each argument
        - ``Tuple[int | None]``: Same as previous, but selectively disable batching for certain
          arguments with a ``None`` value
        - ``Tuple[int | PyTree[int] | None]``: Same as previous, but specify a different batch
          axis for each leaf of an argument (Note that the ``PyTreeDefs``, i.e. the container
          structure, must match between the ``in_axes`` element and the corresponding argument.)
        - ``Tuple[int | PyTree[int | None] | None]``: Same as previous, but selectively disable
          batching for individual PyTree leaves

        The ``out_axes`` parameter can be also used to specify the positions of the mapped axis
        in the output. ``out_axes`` is subject to the same modes as well.
    """

    # Check the validity of in_axes and out_axes
    if not all(isinstance(l, int) for l in tree_leaves(in_axes)):
        raise ValueError(
            "Invalid 'in_axes'; it must be an int or a tuple of PyTrees with integer leaves, "
            f"but got {in_axes}"
        )

    if not all(isinstance(l, int) for l in tree_leaves(out_axes)):
        raise ValueError(
            "Invalid 'out_axes'; it must be an int or a tuple of PyTree with integer leaves, "
            f"but got {out_axes}"
        )

    def batched_fn(*args, **kwargs):
        """Vectorization wrapper around the hybrid program using catalyst.for_loop"""

        # Dispatch to jax.vmap when it is called outside qjit.
        if not EvaluationContext.is_tracing():
            return jax.vmap(fn, in_axes, out_axes)(*args, **kwargs)

        args_flat, args_tree = tree_flatten(args)
        in_axes_flat, _ = tree_flatten(in_axes, is_leaf=lambda x: x is None)

        # Check the validity of the input arguments w.r.t. in_axes
        in_axes_deep_struct = tree_structure(in_axes, is_leaf=lambda x: x is None)
        args_deep_struct = tree_structure(args, is_leaf=lambda x: x is None)
        if not isinstance(in_axes, int) and in_axes_deep_struct != args_deep_struct:
            raise ValueError(
                "Invalid 'in_axes'; it must be an int or match the length of positional "
                f"arguments, but got {in_axes_deep_struct} axis specifiers "
                f"and {args_deep_struct} arguments."
            )
        if isinstance(in_axes, int):
            in_axes_flat = [
                in_axes,
            ] * len(args_flat)

        batch_size = _get_batch_size(args_flat, in_axes_flat, axis_size)
        batch_loc = _get_batch_loc(in_axes_flat)

        # Prepare args_flat to run 'fn' one time and get the output-shape
        fn_args_flat = args_flat.copy()
        for loc in batch_loc:
            ax = in_axes_flat[loc]
            fn_args_flat[loc] = jnp.take(args_flat[loc], 0, axis=ax)

        fn_args = tree_unflatten(args_tree, fn_args_flat)

        # Run 'fn' one time to get output-shape
        init_result = fn(*fn_args, **kwargs)

        # Check the validity of the output w.r.t. out_axes
        out_axes_deep_struct = tree_structure(out_axes, is_leaf=lambda x: x is None)
        init_result_deep_struct = tree_structure(init_result, is_leaf=lambda x: x is None)
        if not isinstance(out_axes, int) and out_axes_deep_struct != init_result_deep_struct:
            raise ValueError(
                "Invalid 'out_axes'; it must be an int or match "
                "the number of function results, but got "
                f"{out_axes_deep_struct} axis specifiers and {init_result_deep_struct} results."
            )

        init_result_flat, init_result_tree = tree_flatten(init_result)

        num_axes_out = len(init_result_flat)

        if isinstance(out_axes, int):
            out_axes_flat = [
                out_axes,
            ] * num_axes_out
        else:
            out_axes_flat, _ = tree_flatten(out_axes, is_leaf=lambda x: x is None)

        out_loc = _get_batch_loc(out_axes_flat)

        # Store batched results of all leaves
        # in the flatten format with respect to the 'init_result' shape
        batched_result_list = []
        for j in range(num_axes_out):
            out_shape = (
                (batch_size,)
                if not init_result_flat[j].shape
                else (batch_size, *init_result_flat[j].shape)
            )
            batched_result_list.append(jnp.zeros(shape=out_shape, dtype=init_result_flat[j].dtype))
            batched_result_list[j] = batched_result_list[j].at[0].set(init_result_flat[j])

        # Apply mapping batched_args[1:] ---> fn(args)
        @for_loop(1, batch_size, 1)
        def loop_fn(i, batched_result_list):
            fn_args_flat = args_flat
            for loc in batch_loc:
                ax = in_axes_flat[loc]
                fn_args_flat[loc] = jnp.take(args_flat[loc], i, axis=ax)

            fn_args = tree_unflatten(args_tree, fn_args_flat)
            res = fn(*fn_args, **kwargs)

            res_flat, _ = tree_flatten(res)

            # Update the list of results
            for j in range(num_axes_out):
                batched_result_list[j] = batched_result_list[j].at[i].set(res_flat[j])

            return batched_result_list

        batched_result_list = loop_fn(batched_result_list)

        # Support out_axes on dim > 0
        for loc in out_loc:
            if ax := out_axes_flat[loc]:
                up_axes = [*range(batched_result_list[loc].ndim)]
                up_axes[ax], up_axes[0] = up_axes[0], up_axes[ax]
                batched_result_list[loc] = jnp.transpose(batched_result_list[loc], up_axes)

        # Unflatten batched_result before return
        return tree_unflatten(init_result_tree, batched_result_list)

    return batched_fn


def _get_batch_loc(axes_flat):
    """
    Get the list of mapping locations in the flattened list of in-axes or out-axes.

    This function takes a flattened list of axes and identifies all elements with a
    non-None value. The resulting list contains the indices of these non-None values,
    indicating where the mapping should apply.

    Args:
        axes_flat (List): Flattened list of in-axes or out-axes including `None` elements.

    Returns:
        List: A list of indices representing the locations where the mapping should be applied.
    """

    return [i for i, d in enumerate(axes_flat) if d is not None]


def _get_batch_size(args_flat, axes_flat, axis_size):
    """Get the batch size based on the provided arguments and axes specifiers, or the manually
       specified batch size by the user request. The batch size must be the same for all arguments.

    Args:
        args_flat (List): Flatten list of arguments.
        axes_flat (List): Flatten list of `in_axes` or `our_axes` including `None` elements.
        axis_size (Optional[int]): Optional default batch size.

    Returns:
        int: Returns the batch size used as the upper bound of the QJIT-compatible for loop
            in the computation of vmap.

    Raises:
        ValueError: The batch size must be the same for all arguments.
        ValueError: The default batch is expected to be None, or less than or equal
        to the computed batch size.
    """

    batch_sizes = []
    for arg, d in zip(args_flat, axes_flat):
        shape = np.shape(arg) if arg.shape else (1,)
        if d is not None and len(shape) > d:
            batch_sizes.append(shape[d])

    if any(size != batch_sizes[0] for size in batch_sizes[1:]):
        raise ValueError(
            "Invalid batch sizes; expected the batch size to be the same for all arguments, "
            f"but got batch_sizes={batch_sizes} from args_flat={args_flat}"
        )

    batch_size = batch_sizes[0] if batch_sizes else 0

    if axis_size is not None:
        if axis_size <= batch_size:
            batch_size = axis_size
        else:
            raise ValueError(
                "Invalid 'axis_size'; the default batch is expected to be None, "
                "or less than or equal to the computed batch size, but got "
                f"axis_size={axis_size} > batch_size={batch_size}"
            )

    if not batch_size:
        raise ValueError(
            f"Invalid batch size; it must be a non-zero integer, but got {batch_size}."
        )

    return batch_size
