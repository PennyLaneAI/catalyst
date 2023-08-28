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
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.api_util import shaped_abstractify
from jax._src.core import get_aval
from jax._src.lax.lax import _abstractify
from jax._src.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from pennylane import Device, QNode, QubitDevice, QubitUnitary, QueuingManager
from pennylane.operation import Operator
from pennylane.tape import QuantumTape

import catalyst
from catalyst.jax_primitives import (
    AbstractQbit,
    AbstractQreg,
    GradParams,
    Qreg,
    adjoint_p,
    compbasis,
    compbasis_p,
    counts,
    expval,
    expval_p,
    func_p,
    grad_p,
    hamiltonian,
    hermitian,
    jvp_p,
    namedobs,
    probs,
    probs_p,
    qalloc,
    qcond_p,
    qdealloc,
    qdevice,
    qdevice_p,
    qextract,
    qextract_p,
    qfor_p,
    qinsert,
    qinst,
    qmeasure_p,
    qunitary,
    qwhile_p,
    sample,
    state,
    tensorobs,
)
from catalyst.jax_primitives import var as jprim_var
from catalyst.jax_primitives import vjp_p
from catalyst.jax_tracer import (
    KNOWN_NAMED_OBS,
    Adjoint,
    Cond,
    ForLoop,
    Function,
    HybridOp,
    HybridOpRegion,
    MainTracingContext,
    MidCircuitMeasure,
    QFunc,
    QJITDevice,
    WhileLoop,
    deduce_avals,
    new_inner_tracer,
)
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError
from catalyst.utils.jax_extras import ClosedJaxpr, DynamicJaxprTracer, Jaxpr, JaxprEqn
from catalyst.utils.jax_extras import MainTrace as JaxMainTrace
from catalyst.utils.jax_extras import (
    ShapedArray,
    _initial_style_jaxpr,
    _input_type_to_tracers,
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    new_main2,
    sort_eqns,
)
from catalyst.utils.tracing import (EvaluationMode, EvaluationContext, MainTracingContext)

# pylint: disable=too-many-lines


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
    assert jaxpr.eqns[0].primitive == func_p, "Expected jaxpr consisting of a single function call."

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
    grad_params = _check_grad_params(method, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return jvp_p.bind(*params, *tangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


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
    grad_params = _check_grad_params(method, h, argnum)
    jaxpr = _make_jaxpr_check_differentiable(fn, grad_params, *params)
    return vjp_p.bind(*params, *cotangents, jaxpr=jaxpr, fn=fn, grad_params=grad_params)


# def get_evaluation_mode() -> Tuple[EvaluationMode, Any]:
#     is_tracing = EvaluationContext.is_tracing()
#     if is_tracing:
#         ctx = qml.QueuingManager.active_context()
#         if ctx is not None:
#             mctx = get_main_tracing_context()
#             assert mctx is not None
#             return (EvaluationMode.QUANTUM_COMPILATION, mctx)
#         else:
#             return (EvaluationMode.CLASSICAL_COMPILATION, None)
#     else:
#         return (EvaluationMode.INTERPRETATION, None)


def _aval_to_primitive_type(aval):
    if isinstance(aval, DynamicJaxprTracer):
        aval = aval.strip_weak_type()
    if isinstance(aval, ShapedArray):
        aval = aval.dtype
    assert not isinstance(aval, (list, dict)), f"Unexpected type {aval}"
    return aval


def _check_single_bool_value(tree: PyTreeDef, avals: List[Any], hint=None) -> None:
    hint = f"{hint}: " if hint else ""
    if not treedef_is_leaf(tree):
        raise TypeError(
            f"{hint}A single boolean scalar was expected, got value of tree-shape: {tree}."
        )
    assert len(avals) == 1, f"{avals} does not match {tree}"
    dtype = _aval_to_primitive_type(avals[0])
    if dtype not in (bool, jnp.bool_):
        raise TypeError(f"{hint}A single boolean scalar was expected, got value {avals[0]}.")


def _check_same_types(trees: List[PyTreeDef], avals: List[List[Any]], hint=None) -> None:
    assert len(trees) == len(avals), f"Input trees ({trees}) don't match input avals ({avals})"
    hint = f"{hint}: " if hint else ""
    expected_tree, expected_dtypes = trees[0], [_aval_to_primitive_type(a) for a in avals[0]]
    for i, (tree, aval) in list(enumerate(zip(trees, avals)))[1:]:
        if tree != expected_tree:
            raise TypeError(
                f"{hint}Same return types were expected, got:\n"
                f" - Branch at index 0: {expected_tree}\n"
                f" - Branch at index {i}: {tree}\n"
                "Please specify the default branch if none was specified."
            )
        dtypes = [_aval_to_primitive_type(a) for a in aval]
        if dtypes != expected_dtypes:
            raise TypeError(
                f"{hint}Same return types were expected, got:\n"
                f" - Branch at index 0: {expected_dtypes}\n"
                f" - Branch at index {i}: {dtypes}\n"
                "Please specify the default branch if none was specified."
            )


class CondCallable:
    def __init__(self, pred, true_fn):
        self.preds = [pred]
        self.branch_fns = [true_fn]
        self.otherwise_fn = lambda: None

    def else_if(self, pred):
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
        outer_trace = ctx.trace
        in_classical_tracers = self.preds
        regions: List[HybridOpRegion] = []

        out_trees, out_avals = [], []
        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
                wffa, in_avals, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))
            out_trees.append(out_tree())
            out_avals.append(res_classical_tracers)

        _check_same_types(out_trees, out_avals, hint="Conditional branches")
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
        _check_same_types(
            out_trees, [j.out_avals for j in branch_jaxprs], hint="Conditional branches"
        )
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
        elif mode == EvaluationMode.INTERPRETATION:
            return self._call_during_interpretation()
        raise RuntimeError(f"Unsupported evaluation mode {mode}")


def cond(pred: DynamicJaxprTracer):
    def _decorator(true_fn: Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator


def for_loop(lower_bound, upper_bound, step):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: MainTracingContext):
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
            elif mode == EvaluationMode.INTERPRETATION:
                return _call_during_interpretation()
            raise RuntimeError(f"Unsupported evaluation mode {mode}")

        return _call_handler

    return _body_query


def while_loop(cond_fn):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: MainTracingContext):
                outer_trace = ctx.trace
                in_classical_tracers, in_tree = tree_flatten(init_state)

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

                _check_single_bool_value(
                    cond_tree(), res_classical_tracers, hint="Condition return value"
                )

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
                _check_single_bool_value(
                    cond_tree, cond_jaxpr.out_avals, hint="Condition return value"
                )
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
            elif mode == EvaluationMode.INTERPRETATION:
                return _call_during_interpretation()
            raise RuntimeError(f"Unsupported evaluation mode {mode}")

        return _call_handler

    return _body_query


def measure(wires) -> DynamicJaxprTracer:
    EvaluationContext.check_is_tracing(
        "catalyst.measure can only be used from within @qjit.")
    EvaluationContext.check_is_quantum_tracing(
        "catalyst.measure can only be used from within a qml.qnode.")
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
    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.adjoint can only be used from within a qml.qnode.")
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
        return None

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
