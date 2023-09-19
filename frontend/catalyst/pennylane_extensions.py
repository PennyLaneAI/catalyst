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

import numbers
from typing import Any, Callable, Iterable, List, Optional, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.api_util import shaped_abstractify
from jax._src.core import get_aval
from jax._src.lax.lax import _abstractify
from jax._src.tree_util import PyTreeDef, tree_flatten, tree_unflatten, treedef_is_leaf
from pennylane import QNode, QueuingManager
from pennylane.operation import Operator
from pennylane.tape import QuantumTape

import catalyst
from catalyst.jax_primitives import (
    GradParams,
    expval_p,
    func_p,
    grad_p,
    jvp_p,
    probs_p,
    qcond_p,
    qfor_p,
    qwhile_p,
    vjp_p,
)
from catalyst.jax_tracer import (
    Adjoint,
    Cond,
    ForLoop,
    Function,
    HybridOpRegion,
    MidCircuitMeasure,
    QCtrl,
    QFunc,
    QJITDevice,
    WhileLoop,
    deduce_avals,
)
from catalyst.utils.exceptions import DifferentiableCompileError
from catalyst.utils.jax_extras import (
    DynamicJaxprTracer,
    Jaxpr,
    ShapedArray,
    _initial_style_jaxpr,
    _input_type_to_tracers,
    initial_style_jaxprs_with_common_consts1,
    new_inner_tracer,
)
from catalyst.utils.tracing import EvaluationContext, EvaluationMode, JaxTracingContext


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


Differentiable = Union[Function, QNode]
DifferentiableLike = Union[Differentiable, Callable, "catalyst.compilation_pipelines.QJIT"]


def _ensure_differentiable(f: DifferentiableLike) -> Differentiable:
    """Narrows down the set of the supported differentiable objects."""

    # Unwrap the function from an existing QJIT object.
    if isinstance(f, catalyst.compilation_pipelines.QJIT):
        f = f.user_function

    if isinstance(f, (Function, QNode)):
        return f
    elif isinstance(f, Callable):  # Keep at the bottom
        return Function(f)

    raise DifferentiableCompileError(f"Non-differentiable object passed: {type(f)}")


def _make_jaxpr_check_differentiable(f: Differentiable, grad_params: GradParams, *args) -> Jaxpr:
    """Gets the jaxpr of a differentiable function. Perform the required additional checks."""
    method = grad_params.method
    jaxpr = jax.make_jaxpr(f)(*args)

    assert len(jaxpr.eqns) == 1, "Expected jaxpr consisting of a single function call."
    assert jaxpr.eqns[0].primitive == func_p, "Expected jaxpr consisting of a single function call."

    for pos, arg in enumerate(jaxpr.in_avals):
        if arg.dtype.kind != "f" and pos in grad_params.argnum:
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

    return jaxpr


def _verify_differentiable_child_qnodes(jaxpr, method):
    """Traverse QNodes being differentiated in the 'call graph' of the JAXPR to verify them."""
    visited = set()

    def traverse_children(jaxpr):
        for eqn in jaxpr.eqns:
            # The Python function is stored in the "fn" parameter of func_p JAXPR primitives.
            fn = eqn.params.get("fn")
            if fn and fn not in visited:
                child = eqn.params.get("call_jaxpr", None)
                if isinstance(fn, (qml.QNode, Grad)):
                    _check_created_jaxpr_gradient_methods(fn, method, child)
                if child and child not in visited:
                    traverse_children(child)
            visited.add(fn)

    traverse_children(jaxpr)


def _check_created_jaxpr_gradient_methods(f: Differentiable, method: str, jaxpr: Jaxpr):
    """Additional checks for the given jaxpr of a differentiable function."""
    if method == "fd":
        return

    if isinstance(f, Grad):
        raise DifferentiableCompileError(
            "Only finite difference can compute higher order derivatives"
        )

    assert isinstance(
        f, qml.QNode
    ), "Expected quantum differentiable node to be a qml.QNode or a catalyst.grad op"
    return_ops = []
    for res in jaxpr.outvars:
        for eq in reversed(jaxpr.eqns):  # pragma: no branch
            if res in eq.outvars:
                return_ops.append(eq.primitive)
                break

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
    method: str, scalar_out: bool, h: Optional[float], argnum: Optional[Union[int, List[int]]]
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
    return GradParams(method, scalar_out, h, argnum)


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

    def __call__(self, *args, **kwargs):
        """Specifies that an actual call to the differentiated function.
        Args:
            args: the arguments to the differentiated function
        """
        EvaluationContext.check_is_tracing(
            "catalyst.grad can only be used from within @qjit decorated code."
        )
        jaxpr = _make_jaxpr_check_differentiable(self.fn, self.grad_params, *args)

        args_data, _ = tree_flatten(args)

        # It always returns list as required by catalyst control-flows
        return grad_p.bind(*args_data, jaxpr=jaxpr, fn=self, grad_params=self.grad_params)


def grad(f: DifferentiableLike, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible gradient transformation for PennyLane/Catalyst.

    This function allows the gradient of a hybrid quantum-classical function
    to be computed within the compiled program.

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
    return Grad(
        _ensure_differentiable(f), grad_params=_check_grad_params(method, scalar_out, h, argnum)
    )


def jacobian(f: DifferentiableLike, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian transformation for PennyLane/Catalyst.

    This function allows the Jacobian of a hybrid quantum-classical function
    to be computed within the compiled program.

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
    array([[-1.32116540e-07,  1.33781874e-07],
           [-4.20735506e-01,  4.20735506e-01]])
    """
    scalar_out = False
    return Grad(
        _ensure_differentiable(f), grad_params=_check_grad_params(method, scalar_out, h, argnum)
    )


def jvp(f: DifferentiableLike, params, tangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian-vector product for PennyLane/Catalyst.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program.

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
    EvaluationContext.check_is_tracing(
        "catalyst.jvp can only be used from within @qjit decorated code."
    )

    def _check(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")
        return x

    params = _check(params, "params")
    tangents = _check(tangents, "tangents")
    fn: Differentiable = _ensure_differentiable(f)
    scalar_out = False
    grad_params = _check_grad_params(method, scalar_out, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return jvp_p.bind(*params, *tangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


def vjp(f: DifferentiableLike, params, cotangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product for PennyLane/Catalyst.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program.

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
    EvaluationContext.check_is_tracing(
        "catalyst.vjp can only be used from within @qjit decorated code."
    )

    def _check(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")
        return x

    params = _check(params, "params")
    cotangents = _check(cotangents, "cotangents")
    fn: Differentiable = _ensure_differentiable(f)
    scalar_out = False
    grad_params = _check_grad_params(method, scalar_out, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return vjp_p.bind(*params, *cotangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


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


def _check_cond_same_types(trees: List[PyTreeDef], avals: List[List[Any]]) -> None:
    assert len(trees) == len(avals), f"Input trees ({trees}) don't match input avals ({avals})"
    expected_tree, expected_dtypes = trees[0], [_aval_to_primitive_type(a) for a in avals[0]]
    for tree, aval in list(zip(trees, avals))[1:]:
        if tree != expected_tree:
            raise TypeError("Conditional requires consistent return types across all branches")
        dtypes = [_aval_to_primitive_type(a) for a in aval]
        if dtypes != expected_dtypes:
            raise TypeError("Conditional requires consistent return types across all branches")


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
                wffa, _, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))
            out_trees.append(out_tree())
            out_avals.append(res_classical_tracers)

        _check_cond_same_types(out_trees, out_avals)
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
        _check_cond_same_types(out_trees, [j.out_avals for j in branch_jaxprs])
        out_classical_tracers = qcond_p.bind(*(self.preds + consts), branch_jaxprs=branch_jaxprs)
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

    def _decorator(true_fn: Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator


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
                    wffa, in_avals, body_tree = deduce_avals(
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
                out_classical_tracers = qfor_p.bind(
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
                    cond_wffa, cond_in_avals, cond_tree = deduce_avals(cond_fn, init_state, {})
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
                    wffa, in_avals, body_tree = deduce_avals(body_fn, init_state, {})
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
                out_classical_tracers = qwhile_p.bind(
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


def measure(wires) -> DynamicJaxprTracer:
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
    EvaluationContext.check_is_tracing("catalyst.measure can only be used from within @qjit.")
    EvaluationContext.check_is_quantum_tracing(
        "catalyst.measure can only be used from within a qml.qnode."
    )
    ctx = EvaluationContext.get_main_tracing_context()
    wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
    if len(wires) != 1:
        raise TypeError(f"One classical argument (a wire) is expected, got {wires}")
    # assert len(ctx.trace.frame.eqns) == 0, ctx.trace.frame.eqns
    out_classical_tracer = new_inner_tracer(ctx.trace, get_aval(True))
    MidCircuitMeasure(
        in_classical_tracers=wires, out_classical_tracers=[out_classical_tracer], regions=[]
    )
    return out_classical_tracer


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

    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.adjoint can only be used from within a qml.qnode."
        )
        ctx = EvaluationContext.get_main_tracing_context()
        with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
            in_classical_tracers, _ = tree_flatten((args, kwargs))
            wffa, in_avals, _ = deduce_avals(_callee, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            quantum_tape = QuantumTape()
            with QueuingManager.stop_recording(), quantum_tape:
                # FIXME: move all full_raise calls into a separate function
                res_classical_tracers = [
                    inner_trace.full_raise(t)
                    for t in wffa.call_wrapped(*arg_classical_tracers)
                    if isinstance(t, DynamicJaxprTracer)
                ]

            if len(quantum_tape.measurements) > 0:
                raise ValueError("Quantum measurements are not allowed in Adjoints")

            adjoint_region = HybridOpRegion(
                inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
            )

        Adjoint(
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


def qctrl(callee: Callable, control: List[Any], control_values: List[Any]) -> Callable:
    def _call_handler(*args, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.adjoint can only be used from within a qml.qnode."
        )
        in_classical_tracers, _ = tree_flatten((args, kwargs))
        quantum_tape = QuantumTape()
        with QueuingManager.stop_recording(), quantum_tape:
            res = callee(*args, **kwargs)
        out_classical_tracers, _ = tree_flatten(res)

        if len(quantum_tape.measurements) > 0:
            raise ValueError("Quantum measurements are not allowed in QCtrls")

        region = HybridOpRegion(None, quantum_tape, [], [])

        QCtrl(
            control_wire_tracers=list(control),
            control_value_tracers=list(control_values),
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=[],
            regions=[region],
        )

    return _call_handler
