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

"""This file contains the implementation of the combine_global_phases transform,
written using xDSL."""

from dataclasses import dataclass
from inspect import signature

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.scf import ForOp, IfOp, WhileOp
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects.quantum import GlobalPhaseOp
from catalyst.python_interface.pass_api import compiler_transform


class CombineGlobalPhasesPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for combining all :class:`~pennylane.GlobalPhase` gates within the same region
    at the last global phase gate."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        root: func.FuncOp | IfOp | ForOp | WhileOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):  # pylint: disable=cell-var-from-loop
        """Match and rewrite for the combine-global-phases pattern acting on functions or
        control-flow blocks containing GlobalPhase operations.
        """

        for region in root.regions:
            phi = None
            global_phases = []
            for op in region.ops:
                if isinstance(op, GlobalPhaseOp):
                    global_phases.append(op)

            if len(global_phases) < 2:
                continue

            prev = global_phases[0]
            phi_sum = prev.operands[0]
            for current in global_phases[1:]:
                phi = current.operands[0]
                addOp = arith.AddfOp(phi, phi_sum)
                rewriter.insert_op(addOp, InsertPoint.before(current))
                phi_sum = addOp.result

                rewriter.erase_op(prev)
                prev = current

            prev.operands[0].replace_by_if(phi_sum, lambda use: use.operation == prev)
            rewriter.notify_op_modified(prev)


@dataclass(frozen=True)
class CombineGlobalPhasesPass(passes.ModulePass):
    """Pass that combines all global phases within a region into the last global phase operation
    within the region.
    """

    name = "combine-global-phases"

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the combine-global-phases pass."""
        pattern_rewriter.PatternRewriteWalker(
            CombineGlobalPhasesPattern(),
            apply_recursively=False,
        ).rewrite_module(op)


_combine_global_phases = compiler_transform(CombineGlobalPhasesPass)


def combine_global_phases(qnode):
    """Combine all ``GlobalPhase`` operations into a single ``GlobalPhase`` operation.

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. note::

        It is recommended to enable PennyLane program capture when using ``combine_global_phases``
        via ``qjit(capture=True)``.

    **Example**

    The ``combine_global_phases`` compilation pass merges :class:`pennylane.GlobalPhase` operators
    together into one :class:`pennylane.GlobalPhase` operator.

    .. code-block:: python

    import pennylane as qml
    import catalyst

    @qml.qjit(capture=True)
    @catalyst.passes.combine_global_phases
    @qml.qnode(qml.device("lightning.qubit", wires=5))
    def circuit():
        qml.GlobalPhase(0)
        qml.GlobalPhase(1)
        qml.GlobalPhase(2)
        qml.GlobalPhase(3)
        qml.GlobalPhase(4)
        return qml.state()

    >>> print(qml.specs(circuit, level=2)())
    Device: lightning.qubit
    Device wires: 5
    Shots: Shots(total=None)
    Level: combine-global-phases (MLIR-1)
    <BLANKLINE>
    Wire allocations: 5
    Total gates: 1
    Gate counts:
    - GlobalPhase: 1
    Measurements:
    - state(all wires): 1
    Depth: Not computed

    .. details::
        :title: Usage details

        The ``combine_global_phases`` pass does not support optimization around structured
        control flow. Consider the following circuit.

        .. code-block:: python

            import pennylane as qml
            import catalyst

            @qml.qjit(capture=True, autograph=True)
            @catalyst.passes.combine_global_phases
            @qml.qnode(qml.device("lightning.qubit", wires=5))
            def circuit():
                qml.GlobalPhase(0)
                qml.GlobalPhase(1)
                qml.GlobalPhase(2)
                qml.GlobalPhase(3)
                qml.GlobalPhase(4)

                for i in range(3):
                    qml.GlobalPhase(i)
                    qml.GlobalPhase(i + 1)

                return qml.state()

        >>> print(qml.specs(circuit, level=2)())
        Device: lightning.qubit
        Device wires: 5
        Shots: Shots(total=None)
        Level: combine-global-phases (MLIR-1)
        <BLANKLINE>
        Wire allocations: 5
        Total gates: 4
        Gate counts:
        - GlobalPhase: 4
        Measurements:
        - state(all wires): 1
        Depth: Not computed

        The resulting circuit contains 4 ``GlobalPhase`` operations: one from the 5 ``GlobalPhase``s
        merged outside of the ``for`` loop, and three ``GlobalPhase``s total from the entire ``for``
        loop (the two within the body of the ``for`` loop are merged).

        Lastly, ``GlobalPhase`` operations can be merged together when nested in symbolic operations
        like ``ctrl`` or ``adjoint``:

        .. code-block:: python

            import pennylane as qml
            import catalyst

            @qml.qjit(capture=True, autograph=True)
            @catalyst.passes.combine_global_phases
            @qml.qnode(qml.device("lightning.qubit", wires=5))
            def circuit():

                qml.ctrl(qml.GlobalPhase, control=(0, 1, 2))(3)
                qml.ctrl(qml.GlobalPhase, control=(0, 1, 2))(4)

                return qml.state()

        >>> print(qml.specs(circuit, level=2)())
        Device: lightning.qubit
        Device wires: 5
        Shots: Shots(total=None)
        Level: combine-global-phases (MLIR-1)
        <BLANKLINE>
        Wire allocations: 5
        Total gates: 1
        Gate counts:
        - 3C(GlobalPhase): 1
        Measurements:
        - state(all wires): 1
        Depth: Not computed
    """
    return _combine_global_phases(qnode)


combine_global_phases.__signature__ = signature(_combine_global_phases)
