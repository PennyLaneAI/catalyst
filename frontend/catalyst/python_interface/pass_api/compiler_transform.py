# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core API for registering xDSL transforms for use with PennyLane and Catalyst."""

from collections.abc import Callable
from inspect import signature
from types import UnionType
from typing import ClassVar, Union, get_args, get_origin

from pennylane.transforms.core import Transform
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .apply_transform_sequence import register_pass


class CompilerTransform(Transform):
    """Wrapper class for applying passes to QJIT-ed workflows."""

    module_pass: ModulePass

    def __init__(self, module_pass: ModulePass):
        self.module_pass = module_pass
        super().__init__(pass_name=module_pass.name)


def compiler_transform(module_pass: ModulePass) -> CompilerTransform:
    """Function to register compilation passes to use within :func:`~.qjit`'d workflows as
    decorators on top of QNodes.

    Args:
        module_pass (xdsl.passes.ModulePass): The module pass to be registered. Note that a
            :class:`~.CompilationPass` is a ``ModulePass`` subclass, which is intended to be the
            most common type to be used in conjuction with ``compiler_transform``.

    Returns:
        (Callable): The function that can be used as a traditional PennyLane transform (within
        :func:`~.qjit`'d workflows only).

    .. see-also::

        :class:`~.CompilationPass` for creating custom compilation pass scaffolding.

    **Example**

    Here is a toy example that arbitrarily replaces a Hadamard gate with an ``RX`` gate, where the
    angle of rotation is specified with positional and keyword arguments.

    .. code-block:: python

        import pennylane as qp

        class ReplaceWithRX(qp.CompilationPass):
            name = "replace-with-rx"

            def __init__(self, theta, mult_by_two=True):
                self.theta = theta
                self.mult_by_two = mult_by_two

            def action(self, op, rewriter):
                if qp.compiler.op_eq(op, qp.H):
                    angle = self.theta * 2 if self.mult_by_two else self.theta
                    new_op = qp.RX(angle, wires=op.in_qubits)
                    rewriter.replace_op(op, new_op)

        replace_with_rx = qp.compiler_transform(ReplaceWithRX)

    The ``replace_with_rx`` pass can now be applied to a QNode within a :func:`~.qjit` workflow,
    where the arguments ``theta`` and ``mult_by_two`` can be passed as arguments to it.

    .. code-block:: python

        @qp.qjit
        @replace_with_rx(0.123, mult_by_two=False)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.H(0)
            return qp.state()

    >>> circuit()
    Array([0.99810947+0.j        , 0.        -0.06146124j], dtype=complex128)
    """
    transform = CompilerTransform(module_pass)

    # Registration for apply-transform-sequence interpreter
    def get_pass_cls():
        return module_pass

    register_pass(module_pass.name, get_pass_cls)
    return transform


class CompilationPass(ModulePass):
    """A base class for defining Catalyst-compatible compilation passes using
    `xDSL <https://xdsl.dev/>`_. ``CompilationPass`` is a subclass of xDSL's ``ModulePass`` to
    abstract away compiler-level details that would otherwise need to be provided manually.

    ``CompilationPass`` subclass objects have three attributes:

    * :attr:`~.CompilationPass.name` (required): Specifies a compilation pass' name. Kebab casing is suggested for naming (e.g., ``"my-pass-name"``).
    * :attr:`~.CompilationPass.recursive` (default: ``True``):  Whether or not the actions of the pass should be applied recursively. If ``True``, the actions will be applied repeatedly until a steady-state is reached.
    * :attr:`~.CompilationPass.greedy` (default: ``True``):  Whether or not the actions of the pass should be applied greedily. Only relevant if the compilation pass defines multiple actions. If ``True``, each iteration of the actions' application (if ``recursive == True``) will only apply the first action that modifies the input module.

    .. see-also::

        `~.compiler_transform` for registering ``CompilationPass`` subclasses to use with
        :func:`pennylane.qjit` workflows.

        TODO: add see-also's for utility functions like ``op_eq``.

    A ``CompilationPass`` must specify an :meth:`action <~.CompilationPass.action>` that describes
    how it will transform operations. Please consult the documentation for the
    :meth:`action <~.CompilationPass.action>` method below.
    """

    name: ClassVar[str]
    """str: Specifies a compilation pass' name. Kebab casing is suggested for naming (e.g.,
    ``"my-pass-name"``).
    """

    recursive: ClassVar[bool] = True
    """bool: Whether or not the actions should be applied recursively. If ``True``, the actions will
    be applied repeatedly until a steady-state is reached. ``True`` by default.
    """

    greedy: ClassVar[bool] = True
    """bool: Whether or not the actions should be applied greedily. Only relevant if the compilation
    pass defines multiple actions. If ``True``, each iteration of the actions' application (if
    ``recursive == True``) will only apply the first action that modifies the input module. ``True``
    by default.
    """

    _rewrite_patterns: ClassVar[list[RewritePattern]] = []
    r"""list[xdsl.pattern_rewriter.RewritePattern]: List of registered actions. The
    stored values are ``RewritePattern``\ s used to implement the registered actions.
    """

    def __init_subclass__(cls: type["CompilationPass"]) -> None:
        cls._rewrite_patterns = []

        if cls.action is not CompilationPass.action:
            cls.add_action(cls.action)

    def action(self, op: Operation, rewriter: PatternRewriter) -> None:
        """The action that performs the transformation on an input operation (``op``).

        Args:
            op (xdsl.ir.Operation): the operation currently being transformed. If
                a type hint for the operation is provided, this method will _only_
                be invoked if the input operation matches the type hint.
            rewriter (xdsl.pattern_rewriter.PatternRewriter): a ``PatternRewriter``
                that provides methods for transforming operations.

        .. see-also::

            :meth:`~.CompilationPass.add_action` for specifying additional actions. ``add_action``
            should be used alongside ``action`` when a different behaviour is desired for certain
            operation types.

        **Example**

        A ``CompilationPass`` subclass' functionality is predominantly defined by its ``action``.
        A simple example of an action is provided below, where the pass aims to arbitrarily replace
        a ``Hadamard`` gate a ``Rot`` gate.

        .. code-block:: python

            import pennylane as qp

            class ReplaceWithRot(qp.CompilationPass):
                name = "replace-with-rot"

                def action(self, op, rewriter):
                    if qp.compiler.op_eq(op, qp.H):
                        new_op = qp.Rot(0.1, 0.2, 0.3, wires=op.in_qubits)
                        rewriter.replace_op(op, new_op)

            replace_with_rot = qp.compiler_transform(ReplaceWithRot)

            @qp.qjit
            @replace_with_rot
            @qp.qnode(qp.device("lightning.qubit", wires=1))
            def circuit():
                qp.H(0)
                return qp.state()

        >>> circuit()
        Array([0.97517033-0.19767681j, 0.09933467+0.00996671j], dtype=complex128)

        As expected, the circuit's state is not a uniform superposition as if a Hadamard was
        applied, as it was replaced with a ``Rot`` gate.
        """

    @classmethod
    def add_action(
        cls, action: Callable[["CompilationPass", Operation, PatternRewriter], None]
    ) -> None:
        """Register an additional action that performs a transformation on an input operation.

        The action _must_ type hint which operation is being rewritten. It must have the
        following signature,

        .. code-block:: python

            import pennylane as qp

            @qp.CompilationPass.add_action
            def rewrite_myop(self, op: MyOperationType, rewriter: PatternRewriter) -> None:
                ...

        In the above example, the type hint for the ``op`` argument is used to determine which xDSL
        operation is being matched. If not provided, _all_ xDSL operations will be matched.

        Additionally, the type hint can also contain a union of multiple operation types:

        .. code-block:: python

            @qp.CompilationPass.add_action
            def rewrite_myop(
                self, op: MyOperation1 | MyOperation2 | MyOperation3, rewriter: PatternRewriter
            ) -> None:
                ...

        A full list of quantum operations that can be used for type-hinting can be found in the
        ``catalyst.python_interface.dialects.quantum`` dialect.

        .. note::

            If an action for the provided operation already exists, the existing action will get
            priority over the new one. To see all registered actions, use the
            :meth:`~.CompilationPass.actions` property.

        Args:
            action (Callable): A callable meeting the above constraints that transforms a given
                operation

        **Example**

        A simple example of an action and an added action is provided below, where the pass aims to
        arbitrarily replace a ``MultiRZ`` gate with a series of ``T`` gates via ``action``, and the
        added action (``new_action``) will only replace ``PCPhase`` operations with a series of
        Hadamard gates.

        .. code-block:: python

            import pennylane as qp
            from catalyst.python_interface.dialects import quantum

            class ReplaceOps(qp.CompilationPass):
                name = "replace-ops"

                def action(self, op: quantum.MultiRZOp, rewriter):
                    T_gates = []
                    for qubit in op.in_qubits:
                        T_gates.append(qp.T(qubit))

                    rewriter.replace_op(op, new_ops)

                @qp.CompilationPass.add_action
                def new_action(self, op: quantum.PCPhaseOp, rewriter):
                    H_gates = []
                    for qubit in op.in_qubits:
                        H_gates.append(qp.H(qubit))

                    rewriter.replace_op(op, new_ops)

            replace_ops = qp.compiler_transform(ReplaceOps)

            @qp.qjit
            @replace_ops
            @qp.qnode(qp.device("lightning.qubit", wires=4))
            def circuit():
                qp.PCPhase(1.23, dim=7, wires=[1, 0, 3])
                qp.MultiRZ(0.1, wires=[0, 1, 2, 3])
                qp.adjoint(qp.CNOT)
                return qp.state()

        >>> circuit()
        Array([ 0.35355339+0.j        ,  0.25      +0.25j      ,
                0.        +0.j        ,  0.        +0.j        ,
                0.25      +0.25j      ,  0.        +0.35355339j,
                0.        +0.j        ,  0.        +0.j        ,
                0.25      +0.25j      ,  0.        +0.35355339j,
                0.        +0.j        ,  0.        +0.j        ,
                0.        +0.35355339j, -0.25      +0.25j      ,
                0.        +0.j        ,  0.        +0.j        ],      dtype=complex128)
        """
        if cls is CompilationPass:
            raise TypeError(
                "Cannot use 'CompilationPass.add_action'. New actions can only be registered "
                "for subclasses of 'CompilationPass'."
            )

        # xdsl.pattern_rewriter.op_type_rewrite_pattern was used as a reference to
        # implement the type hint collection. Source:
        # https://github.com/xdslproject/xdsl/blob/main/xdsl/pattern_rewriter.py
        params = list(signature(action, eval_str=True).parameters.values())
        if len(params) != 3 or params[0].name != "self":
            raise ValueError("The action must have 3 arguments, with the first one being 'self'.")

        # If a type hint for the op we're trying to match isn't provided, match all ops
        hint = Operation if params[-2].name not in action.__annotations__ else params[-2].annotation

        expected_types = get_args(hint) if get_origin(hint) in (Union, UnionType) else (hint,)
        if not all(issubclass(e, Operation) for e in expected_types):
            raise TypeError(
                "Only Operation types or unions of Operation types can be used to register actions."
            )
        rewrite_pattern = _create_rewrite_pattern(hint, action)

        cls._rewrite_patterns.append(rewrite_pattern)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:  # pylint: disable=unused-argument
        """Apply the transformation to the input module.

        If ``greedy`` is ``True``, the actions will be applied greedily, i.e., for each
        operation in the worklist, we will apply only the first action that matches the
        operation. Otherwise, they will be applied by creating a different worklist for each
        action.

        If ``recursive`` is ``True``, the worklist algorithm will continue applying the actions
        until a steady-state is reached.

        .. note::

            The input module is mutated in-place.

        Args:
            ctx: Context containing operation and attribute registrations
            op: Module to which to apply the transform
        """
        if self.greedy:
            pattern = GreedyRewritePatternApplier(
                rewrite_patterns=[rp(self) for rp in self._rewrite_patterns]
            )
            walker = PatternRewriteWalker(pattern=pattern, apply_recursively=self.recursive)
            walker.rewrite_module(op)

        else:
            for rp in self._rewrite_patterns:
                walker = PatternRewriteWalker(pattern=rp(self), apply_recursively=self.recursive)
                walker.rewrite_module(op)


# pylint: disable=missing-function-docstring


def _update_op_type_hint(hint: type[Operation]) -> Callable:
    """Update the signature of a ``match_and_rewrite`` method to use the provided type hint
    for the ``op`` argument."""

    def _update_match_and_rewrite(method: Callable) -> Callable:
        params = tuple(signature(method).parameters)
        # Update type hint of operation argument
        # TODO: Is it fine to mutate in-place or should we return a new function?
        op_arg_name = params[-2]
        method.__annotations__[op_arg_name] = hint

        return method

    return _update_match_and_rewrite


def _create_rewrite_pattern(hint: type[Operation], action: Callable) -> RewritePattern:
    """Given an action defined as a function, create a ``RewritePattern`` which
    can be used with xDSL's pass API."""

    # pylint: disable=too-few-public-methods
    class LocalRewritePattern(RewritePattern):
        """Rewrite pattern for transforming a matched operation."""

        _pass: CompilationPass

        def __init__(self, _pass):
            self._pass = _pass
            super().__init__()

        @op_type_rewrite_pattern
        @_update_op_type_hint(hint)
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /) -> None:
            action(self._pass, op, rewriter)

    return LocalRewritePattern
