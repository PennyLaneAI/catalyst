# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains public API functions that enabling quantum programming
with control flow, including conditionals, for loops, and while loops.
"""

from typing import Any, Callable, List

import jax.numpy as jnp
from jax._src.api_util import shaped_abstractify
from jax._src.lax.lax import _abstractify
from jax._src.tree_util import PyTreeDef, tree_flatten, tree_unflatten, treedef_is_leaf
from jax.core import AbstractValue
from pennylane import QueuingManager
from pennylane.tape import QuantumTape

from catalyst.jax_extras import (
    ClosedJaxpr,
    DynamicJaxprTracer,
    ShapedArray,
    _initial_style_jaxpr,
    _input_type_to_tracers,
    collapse,
    convert_constvars_jaxpr,
    deduce_avals,
    deduce_signatures,
    expand_args,
    expand_results,
    find_top_trace,
    for_loop_expansion_strategy,
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    input_type_to_tracers,
    new_inner_tracer,
    output_type_to_tracers,
)
from catalyst.jax_primitives import AbstractQreg, cond_p, for_p, while_p
from catalyst.jax_tracer import (
    HybridOp,
    HybridOpRegion,
    QRegPromise,
    trace_function,
    trace_quantum_tape,
    unify_jaxpr_result_types,
)
from catalyst.tracing.contexts import (
    EvaluationContext,
    EvaluationMode,
    JaxTracingContext,
)


## API ##
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


def for_loop(lower_bound, upper_bound, step, experimental_preserve_dimensions=True):
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

    def _decorator(body_fn):
        return ForLoopCallable(
            lower_bound, upper_bound, step, body_fn, experimental_preserve_dimensions
        )

    return _decorator


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

    def _decorator(body_fn):
        return WhileLoopCallable(cond_fn, body_fn)

    return _decorator


## IMPL ##
class CondCallable:
    """User-facing wrapper provoding "else_if" and "otherwise" public methods.
    Some code in this class has been adapted from the cond implementation in the JAX project at
    https://github.com/google/jax/blob/jax-v0.4.1/jax/_src/lax/control_flow/conditionals.py
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.

    Also provides access to the underlying "Cond" operation object.

    **Example**


    .. code-block:: python

        @qml.transform
        def my_quantum_transform(tape):
            ops = tape.operations.copy()

            @cond(isinstance(ops[-1], qml.Hadamard))
            def f():
                qml.Hadamard(1)
                return 1
            @f.otherwise
            def f():
                qml.T(0)
                return 0

            res = f()
            ops.append(f.operation)

            def post_processing_fn(results):
                return results

            modified_tape = qml.tape.QuantumTape(ops, tape.measurements)
            print(res)
            print(modified_tape.operations)
            return [modified_tape], post_processing_fn

        @qml.qjit
        @my_quantum_transform
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def main():
            qml.Hadamard(0)
            return qml.probs()

    >>> main()
    Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=2/1)>
    [Hadamard(wires=[0]), Cond(tapes=[[Hadamard(wires=[1])], [T(wires=[0])]])]
    (array([0.25, 0.25, 0.25, 0.25]),)
    """

    def __init__(self, pred, true_fn):
        self.preds = [pred]
        self.branch_fns = [true_fn]
        self.otherwise_fn = lambda: None
        self._operation = None

    @property
    def operation(self):
        """
        @property for CondCallable.operation
        """
        if self._operation is None:
            raise AttributeError(
                """
                The cond() was not called (or has not been called) in a quantum context,
                and thus has no associated quantum operation.
                """
            )
        return self._operation

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

        out_treedefs, out_signatures = [], []
        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
                wffa, _, _, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
                    res_avals = [shaped_abstractify(res) for res in res_classical_tracers]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))
            out_treedefs.append(out_tree())
            out_signatures.append(res_avals)

        _assert_cond_result_structure(out_treedefs)
        _assert_cond_result_types(out_signatures)
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in out_signatures[0]]
        self._operation = Cond(in_classical_tracers, out_classical_tracers, regions)
        return tree_unflatten(out_tree(), out_classical_tracers)

    def _call_with_classical_ctx(self):
        args, args_tree = tree_flatten([])
        args_avals = tuple(map(_abstractify, args))
        branch_jaxprs, consts, out_treedefs = initial_style_jaxprs_with_common_consts1(
            (*self.branch_fns, self.otherwise_fn), args_tree, args_avals, "cond"
        )
        out_signatures = [jaxpr.out_avals for jaxpr in branch_jaxprs]
        _assert_cond_result_structure(out_treedefs)
        _assert_cond_result_types(out_signatures)
        branch_jaxprs = unify_jaxpr_result_types(branch_jaxprs)
        out_classical_tracers = cond_p.bind(*(self.preds + consts), branch_jaxprs=branch_jaxprs)
        return tree_unflatten(out_treedefs[0], out_classical_tracers)

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


class ForLoopCallable:
    """
    Wrapping for_loop decorator into a class so that the actual "ForLoop" operation object, which
    is created locally in _call_with_quantum_ctx(ctx), can be retrived without changing its
    return type. The retrived ForLoop is in LoopBodyFunction.operation.

    **Example**


    .. code-block:: python

        @qml.transform
        def my_quantum_transform(tape):
            ops = tape.operations.copy()

            @for_loop(0, 4, 1)
            def f(i, sum):
                qml.Hadamard(0)
                return sum+1

            res = f(0)
            ops.append(f.operation)

            def post_processing_fn(results):
                return results
            modified_tape = qml.tape.QuantumTape(ops, tape.measurements)
            print(res)
            print(modified_tape.operations)
            return [modified_tape], post_processing_fn

        @qml.qjit
        @my_quantum_transform
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def main():
            qml.Hadamard(0)
            return qml.probs()

    >>> main()
    Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=2/1)>
    [Hadamard(wires=[0]), ForLoop(tapes=[[Hadamard(wires=[0])]])]
    (array([0.5, 0. , 0.5, 0. ]),)
    """

    def __init__(self, lower_bound, upper_bound, step, body_fn, experimental_preserve_dimensions):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.body_fn = body_fn
        self._operation = None
        self.expansion_strategy = for_loop_expansion_strategy(experimental_preserve_dimensions)
        self.apply_reverse_transform = isinstance(self.step, int) and self.step < 0

    @property
    def operation(self):
        """
        @property for ForLoopCallable.operation
        """
        if self._operation is None:
            raise AttributeError(
                """
                The for_loop() was not called (or has not been called) in a quantum context,
                and thus has no associated quantum operation.
                """
            )
        return self._operation

    def _call_with_quantum_ctx(self, ctx: JaxTracingContext, *init_state):
        quantum_tape = QuantumTape()
        outer_trace = ctx.trace
        aux_classical_tracers = [
            outer_trace.full_raise(t) for t in [self.lower_bound, self.upper_bound, self.step]
        ]
        wfun, in_sig, out_sig = deduce_signatures(
            self.body_fn,
            (aux_classical_tracers[0], *init_state),
            {},
            self.expansion_strategy,
        )
        in_type = in_sig.in_type

        with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
            arg_classical_tracers = input_type_to_tracers(in_type, inner_trace.new_arg)
            with QueuingManager.stop_recording(), quantum_tape:
                res_classical_tracers = [
                    inner_trace.full_raise(t) for t in wfun.call_wrapped(*arg_classical_tracers)
                ]
                out_type = out_sig.out_type()
                out_tree = out_sig.out_tree()
                out_consts = out_sig.out_consts()

        in_expanded_classical_tracers, in_type2 = expand_args(
            aux_classical_tracers + collapse(in_type, in_sig.in_expanded_args),
            self.expansion_strategy,
        )
        out_expanded_classical_tracers = output_type_to_tracers(
            out_type,
            out_consts,
            in_expanded_classical_tracers,
            maker=lambda aval: new_inner_tracer(outer_trace, aval),
        )
        self._operation = ForLoop(
            collapse(in_type2, in_expanded_classical_tracers),
            collapse(out_type, out_expanded_classical_tracers),
            [
                HybridOpRegion(
                    inner_trace,
                    quantum_tape,
                    collapse(in_type, arg_classical_tracers),
                    collapse(out_type, res_classical_tracers),
                )
            ],
            apply_reverse_transform=self.apply_reverse_transform,
            expansion_strategy=self.expansion_strategy,
        )
        return tree_unflatten(out_tree, collapse(out_type, out_expanded_classical_tracers))

    def _call_with_classical_ctx1(self, *init_state):
        iter_arg = self.lower_bound
        init_vals, in_tree = tree_flatten((iter_arg, *init_state))
        init_avals = tuple(_abstractify(val) for val in init_vals)
        body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
            self.body_fn, in_tree, init_avals, "for_loop"
        )

        out_classical_tracers = for_p.bind(
            self.lower_bound,
            self.upper_bound,
            self.step,
            *(body_consts + init_vals),
            body_jaxpr=body_jaxpr,
            body_nconsts=len(body_consts),
            apply_reverse_transform=self.apply_reverse_transform,
        )

        return tree_unflatten(body_tree, out_classical_tracers)

    def _call_with_classical_ctx(self, ctx, *init_state):
        # iter_arg = self.lower_bound
        # init_vals, in_tree = tree_flatten((iter_arg, *init_state))
        # init_avals = tuple(_abstractify(val) for val in init_vals)

        outer_trace = find_top_trace([self.lower_bound, self.upper_bound, self.step])
        aux_tracers = [
            outer_trace.full_raise(t) for t in [self.lower_bound, self.upper_bound, self.step]
        ]

        _, in_sig, out_sig = trace_function(
            ctx,
            self.body_fn,
            *(aux_tracers[0], *init_state),
            expansion_strategy=self.expansion_strategy,
        )

        in_expanded_tracers = [
            *out_sig.out_consts(),
            *expand_args(
                aux_tracers + collapse(in_sig.in_type, in_sig.in_expanded_args),
                expansion_strategy=self.expansion_strategy,
            )[0],
        ]

        out_expanded_tracers = for_p.bind(
            *in_expanded_tracers,
            body_jaxpr=out_sig.out_jaxpr(),
            body_nconsts=len(out_sig.out_consts()),
            apply_reverse_transform=self.apply_reverse_transform,
            nimplicit=in_sig.num_implicit_inputs(),
            preserve_dimensions=not self.expansion_strategy.input_unshare_variables,
        )

        return tree_unflatten(
            out_sig.out_tree(), collapse(out_sig.out_type(), out_expanded_tracers)
        )

    def _call_during_interpretation(self, *init_state):
        args = init_state
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
        for i in range(self.lower_bound, self.upper_bound, self.step):
            fn_res = self.body_fn(i, *args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
        return fn_res

    def __call__(self, *init_state):
        mode, ctx = EvaluationContext.get_evaluation_mode()
        if mode == EvaluationMode.QUANTUM_COMPILATION:
            return self._call_with_quantum_ctx(ctx, *init_state)
        elif mode == EvaluationMode.CLASSICAL_COMPILATION:
            return self._call_with_classical_ctx(ctx, *init_state)
        else:
            assert mode == EvaluationMode.INTERPRETATION, f"Unsupported evaluation mode {mode}"
            return self._call_during_interpretation(*init_state)


class WhileLoopCallable:
    """
    Wrapping while_loop decorator into a class so that the actual "WhileLoop" operation object,
    which is created locally in _call_with_quantum_ctx(ctx), can be retrived without changing
    its return type. The retrived WhileLoop is in LoopBodyFunction.operation.

    **Example**

    .. code-block:: python

        @qml.transform
        def my_quantum_transform(tape):
            ops = tape.operations.copy()
            print("input tape", ops)

            @while_loop(lambda i: i<4)
            def f(i):
                qml.PauliX(0)
                return i+1

            res = f(0)
            ops.append(f.operation)

            def post_processing_fn(results):
                return results

            modified_tape = qml.tape.QuantumTape(ops, tape.measurements)
            print(res)
            print(modified_tape.operations)
            return [modified_tape], post_processing_fn


        @qml.qjit
        @my_quantum_transform
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def main():
            qml.PauliX(0)
            return qml.probs()

    >>> main()
    Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=2/1)>
    [X(0), WhileLoop(tapes=[[X(0)]])]
    (array([0., 0., 1., 0.]),)
    """

    def __init__(self, cond_fn, body_fn):
        self.cond_fn = cond_fn
        self.body_fn = body_fn
        self._operation = None

    @property
    def operation(self):
        """
        @property for WhileLoopCallable.operation
        """
        if self._operation is None:
            raise AttributeError(
                """
                The while_loop() was not called (or has not been called) in a quantum context,
                and thus has no associated quantum operation.
                """
            )
        return self._operation

    def _call_with_quantum_ctx(self, ctx: JaxTracingContext, *init_state):
        outer_trace = ctx.trace
        in_classical_tracers, _ = tree_flatten(init_state)

        with EvaluationContext.frame_tracing_context(ctx) as cond_trace:
            cond_wffa, cond_in_avals, _, cond_tree = deduce_avals(self.cond_fn, init_state, {})
            arg_classical_tracers = _input_type_to_tracers(cond_trace.new_arg, cond_in_avals)
            res_classical_tracers = [
                cond_trace.full_raise(t) for t in cond_wffa.call_wrapped(*arg_classical_tracers)
            ]
            cond_region = HybridOpRegion(
                cond_trace, None, arg_classical_tracers, res_classical_tracers
            )

        _check_single_bool_value(cond_tree(), res_classical_tracers)

        with EvaluationContext.frame_tracing_context(ctx) as body_trace:
            wffa, in_avals, _, body_tree = deduce_avals(self.body_fn, init_state, {})
            arg_classical_tracers = _input_type_to_tracers(body_trace.new_arg, in_avals)
            quantum_tape = QuantumTape()
            with QueuingManager.stop_recording(), quantum_tape:
                res_classical_tracers = [
                    body_trace.full_raise(t) for t in wffa.call_wrapped(*arg_classical_tracers)
                ]
            body_region = HybridOpRegion(
                body_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
            )

        res_avals = list(map(shaped_abstractify, res_classical_tracers))
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

        self._operation = WhileLoop(
            in_classical_tracers, out_classical_tracers, [cond_region, body_region]
        )
        return tree_unflatten(body_tree(), out_classical_tracers)

    def _call_with_classical_ctx(self, *init_state):
        init_vals, in_tree = tree_flatten(init_state)
        init_avals = tuple(_abstractify(val) for val in init_vals)
        cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
            self.cond_fn, in_tree, init_avals, "while_cond"
        )
        body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
            self.body_fn, in_tree, init_avals, "while_loop"
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

    def _call_during_interpretation(self, *init_state):
        args = init_state
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
        while self.cond_fn(*args):
            fn_res = self.body_fn(*args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
        return fn_res

    def __call__(self, *init_state):
        mode, ctx = EvaluationContext.get_evaluation_mode()
        if mode == EvaluationMode.QUANTUM_COMPILATION:
            return self._call_with_quantum_ctx(ctx, *init_state)
        elif mode == EvaluationMode.CLASSICAL_COMPILATION:
            return self._call_with_classical_ctx(*init_state)
        else:
            assert mode == EvaluationMode.INTERPRETATION, f"Unsupported evaluation mode {mode}"
            return self._call_during_interpretation(*init_state)


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
                branch_jaxprs=unify_jaxpr_result_types(jaxprs2),
            )
        )
        return qrp2


class ForLoop(HybridOp):
    """PennyLane ForLoop Operation."""

    binder = for_p.bind

    def trace_quantum1(self, ctx, device, trace, qrp) -> QRegPromise:
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
                apply_reverse_transform=self.apply_reverse_transform,
            )
        )
        return qrp2

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        inner_trace = op.regions[0].trace
        inner_tape = op.regions[0].quantum_tape
        expansion_strategy = self.expansion_strategy

        with EvaluationContext.frame_tracing_context(ctx, inner_trace):
            qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
            qreg_out = qrp_out.actualize()

            arg_expanded_tracers = expand_args(
                self.regions[0].arg_classical_tracers + [qreg_in],
                expansion_strategy=expansion_strategy,
            )[0]

            nimplicit = len(arg_expanded_tracers) - len(self.regions[0].arg_classical_tracers) - 1

            res_expanded_tracers, _ = expand_results(
                [],
                arg_expanded_tracers,
                self.regions[0].res_classical_tracers + [qreg_out],
                expansion_strategy=expansion_strategy,
                num_implicit_inputs=nimplicit,
            )
            jaxpr, _, consts = ctx.frames[inner_trace].to_jaxpr2(res_expanded_tracers)

        in_expanded_tracers = [
            *[trace.full_raise(c) for c in consts],
            *expand_args(op.in_classical_tracers, expansion_strategy=expansion_strategy)[0],
            qrp.actualize(),
        ]

        out_expanded_classical_tracers = expand_results(
            [],
            in_expanded_tracers,
            self.out_classical_tracers,
            expansion_strategy=expansion_strategy,
            num_implicit_inputs=nimplicit,
        )[0]

        qrp2 = QRegPromise(
            op.bind_overwrite_classical_tracers2(
                ctx,
                trace,
                in_expanded_tracers=in_expanded_tracers,
                out_expanded_tracers=out_expanded_classical_tracers,
                body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                body_nconsts=len(consts),
                apply_reverse_transform=self.apply_reverse_transform,
                nimplicit=nimplicit,
                preserve_dimensions=not expansion_strategy.input_unshare_variables,
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


## PRIVATE ##
def _assert_cond_result_structure(trees: List[PyTreeDef]):
    """Ensure a consistent container structure across branch results."""
    expected_tree = trees[0]
    for tree in trees[1:]:
        if tree != expected_tree:
            raise TypeError(
                "Conditional requires a consistent return structure across all branches! "
                f"Got {tree} and {expected_tree}."
            )


def _assert_cond_result_types(signatures: List[List[AbstractValue]]):
    """Ensure a consistent type signature across branch results."""
    num_results = len(signatures[0])
    assert all(len(sig) == num_results for sig in signatures), "mismatch: number or results"

    for i in range(num_results):
        aval_slice = [avals[i] for avals in signatures]
        slice_shapes = [aval.shape for aval in aval_slice]
        expected_shape = slice_shapes[0]
        for shape in slice_shapes:
            if shape != expected_shape:
                raise TypeError(
                    "Conditional requires a consistent array shape per result across all branches! "
                    f"Got {shape} for result #{i} but expected {expected_shape}."
                )


def _check_single_bool_value(tree: PyTreeDef, avals: List[Any]) -> None:
    if not treedef_is_leaf(tree):
        raise TypeError(
            f"A single boolean scalar was expected, got the value of tree-shape: {tree}."
        )
    assert len(avals) == 1, f"{avals} does not match {tree}"
    dtype = _aval_to_primitive_type(avals[0])
    if dtype not in (bool, jnp.bool_):
        raise TypeError(f"A single boolean scalar was expected, got the value {avals[0]}.")


def _aval_to_primitive_type(aval):
    if isinstance(aval, DynamicJaxprTracer):
        aval = aval.strip_weak_type()
    if isinstance(aval, ShapedArray):
        aval = aval.dtype
    assert not isinstance(aval, (list, dict)), f"Unexpected type {aval}"
    return aval
