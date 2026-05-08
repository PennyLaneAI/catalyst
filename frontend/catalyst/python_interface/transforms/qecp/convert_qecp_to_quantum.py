# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QEC Physical to Quantum dialect conversion.

This module contains the implementation of the xDSL convert-qecp-to-quantum dialect-conversion pass.

Known Limitations
-----------------

- The convert-qecp-to-quantum pass does not support the following cases:

  * QEC codes where the number of logical qubits per codeblock, k, is greater than 1.

- This pass is for the debugging purpose only.

- Current qecp.AllocOp lowering only works when all idx used in qecp.ExtractCodeblockOp
can be resolved as concrete integer numbers.

"""

from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.ir import SSAValue
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)

from catalyst.python_interface.dialects import qecp, quantum
from catalyst.python_interface.dialects.quantum.attributes import QuregType, QubitType
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.python_interface.inspection.xdsl_conversion import resolve_constant_params

# MARK: Type Conversion Pattern


@dataclass
class PhysicalCodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecp.codeblock -> quantum.reg."""

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecp.PhysicalCodeblockType) -> QuregType:
        """Type conversion rewrite pattern for physical codeblock types."""

        return QuregType()


@dataclass
class QecPhysicalQubitTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecp.codeblock -> quantum.reg."""

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecp.QecPhysicalQubitType) -> QubitType:
        """Type conversion rewrite pattern for physical qubit types."""

        return QubitType()


@dataclass
class AllocationConversion(RewritePattern):
    """Op conversion pattern from qecp.alloc to one or more quantum.alloc ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.AllocOp, rewriter: PatternRewriter):
        """Lower a physical hyper-register allocation to per-slice quantum.register allocations."""
        hyper_ty = cast(qecp.PhysicalHyperRegisterType, op.result_types[0])
        width = int(hyper_ty.width.value.data)
        num_qubits = int(hyper_ty.n.value.data)

        logical_qregs: list[SSAValue] = []
        for _ in range(width):
            alloc_op = quantum.AllocOp(num_qubits)
            rewriter.insert_op(alloc_op)
            logical_qregs.append(alloc_op.results[0])

        for use in list(op.hyper_reg.uses):
            extract_op = use.operation
            if not isinstance(extract_op, qecp.ExtractCodeblockOp):
                continue
            idx_attr = extract_op.idx_attr
            if idx_attr is None:
                # Dynamic extract index is not supported yet (see module docstring).
                return
            idx = int(idx_attr.value.data)
            if idx < 0 or idx >= len(logical_qregs):
                return

            extract_res = extract_op.results[0]
            users = [u.operation for u in list(extract_res.uses)]
            extract_res.replace_all_uses_with(logical_qregs[idx])
            for user_op in users:
                rewriter.notify_op_modified(user_op)
            rewriter.erase_op(extract_op)

        rewriter.erase_op(op)


@dataclass(frozen=True)
class ConvertQecPhysicalToQuantumPass(ModulePass):
    """
    Convert QEC physical instructions to Quantum instructions.
    """

    name = "convert-qecp-to-quantum"

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecp-to-quantum pass."""

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PhysicalCodeblockTypeConversion(),
                    QecPhysicalQubitTypeConversion(),
                    AllocationConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
