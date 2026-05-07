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
written using xDSL. Within each basic block, consecutive unconditional global
phase operations are merged into a single one by summing their angles."""

from dataclasses import dataclass

from xdsl import context, passes
from xdsl.dialects import arith, builtin
from xdsl.rewriter import InsertPoint, Rewriter

from catalyst.python_interface.dialects.quantum import GlobalPhaseOp
from catalyst.python_interface.pass_api import compiler_transform


def _walk_blocks(op):
    """Recursively yield every basic block nested inside the regions of *op*."""
    for region in op.regions:
        for block in region.blocks:
            yield block
            for block_op in list(block.ops):
                yield from _walk_blocks(block_op)


@dataclass(frozen=True)
class CombineGlobalPhasesPass(passes.ModulePass):
    """Pass for combining consecutive unconditional global phase operations.

    Within each basic block, all ``quantum.gphase`` operations that have no
    control qubits are merged into a single one by summing (or subtracting, for
    adjoint operations) their angle arguments.
    """

    name = "xdsl-combine-global-phases"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the combine global phases pass."""
        rewriter = Rewriter()
        for block in _walk_blocks(module):
            simple_phases = [
                op
                for op in block.ops
                if isinstance(op, GlobalPhaseOp) and not list(op.in_ctrl_qubits)
            ]
            if len(simple_phases) < 2:
                continue

            last = simple_phases[-1]
            running_sum = last.angle

            if last.adjoint is not None:
                neg_op = arith.NegfOp(running_sum)
                rewriter.insert_op(neg_op, InsertPoint.before(last))
                running_sum = neg_op.result
                last.adjoint = None

            for phase_op in simple_phases[:-1]:
                if phase_op.adjoint is not None:
                    combine = arith.SubfOp(running_sum, phase_op.angle)
                else:
                    combine = arith.AddfOp(running_sum, phase_op.angle)
                rewriter.insert_op(combine, InsertPoint.before(last))
                running_sum = combine.result
                rewriter.erase_op(phase_op)

            last.operands[0] = running_sum


combine_global_phases_pass = compiler_transform(CombineGlobalPhasesPass)
