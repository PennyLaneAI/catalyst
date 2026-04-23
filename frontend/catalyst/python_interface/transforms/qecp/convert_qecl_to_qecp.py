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

"""QEC Logical to QEC Physical dialect conversion.

This module contains the implementation of the xDSL convert-qecl-to-qecp dialect-conversion pass.
To apply this pass, the QEC code must be know.
"""

from dataclasses import dataclass, field

from xdsl.context import Context
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

from catalyst.python_interface.dialects import qecl, qecp
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

from .convert_qecl_noise_to_qec_noise import ConvertQECLNoiseOpToQECPNoisePass
from .qec_code_lib import QecCode

# MARK: Type Conversion Pattern


@dataclass
class CodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecl.codeblock -> qecp.codeblock."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalCodeblockType) -> qecp.PhysicalCodeblockType:
        """Type conversion rewrite pattern for logical codeblock types."""
        if typ.k.value.data != self.qec_code.k:
            raise CompileError(
                f"Failed to convert type {typ} with QEC code '{self.qec_code}'; codeblock has "
                f"k = {typ.k.value.data} but QEC code has k = {self.qec_code.k}"
            )

        return qecp.PhysicalCodeblockType(typ.k, self.qec_code.n)


@dataclass
class HyperRegisterTypeConversion(TypeConversionPattern):
    """Hyper-register type conversion pattern from qecl.hyperreg -> qecp.hyperreg."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalHyperRegisterType) -> qecp.PhysicalHyperRegisterType:
        """Type conversion rewrite pattern for physical codeblock types."""
        if typ.k.value.data != self.qec_code.k:
            raise CompileError(
                f"Failed to convert type {typ} with QEC code '{self.qec_code}'; hyper-register has "
                f"k = {typ.k.value.data} but QEC code has k = {self.qec_code.k}"
            )

        return qecp.PhysicalHyperRegisterType(typ.width, typ.k, self.qec_code.n)


# MARK: Alloc/Dealloc Patterns


@dataclass
class AllocationConversion(RewritePattern):
    """Op conversion pattern from qecl.alloc -> qecp.alloc."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.AllocOp, rewriter: PatternRewriter) -> qecp.AllocOp:
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        # assert isinstance(op.hyper_reg, qecp.PhysicalHyperRegisterType)
        assert isinstance(
            op.result_types[0], qecp.PhysicalHyperRegisterType
        ), "lowering of hyper-register types is expected before lowering allocate ops"
        rewriter.replace_op(op, qecp.AllocOp(op.result_types[0]))


@dataclass
class DeallocationConversion(RewritePattern):
    """Op conversion pattern from qecl.dealloc -> qecp.dealloc."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.DeallocOp, rewriter: PatternRewriter) -> qecp.DeallocOp:
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        assert isinstance(
            op.hyper_reg.type, qecp.PhysicalHyperRegisterType
        ), "lowering of hyper-register types is expected before lowering deallocate ops"
        rewriter.replace_op(op, qecp.DeallocOp(op.hyper_reg))


# MARK: Extract/Insert Patterns


@dataclass
class ExtractBlockConversion(RewritePattern):
    """Op conversion pattern from qecl.extract_block -> qecp.extract_block."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qecl.ExtractCodeblockOp, rewriter: PatternRewriter
    ) -> qecp.ExtractCodeblockOp:
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        assert isinstance(
            op.hyper_reg.type, qecp.PhysicalHyperRegisterType
        ), "lowering of hyper-register types is expected before lowering extract_block ops"
        rewriter.replace_op(op, qecp.ExtractCodeblockOp(op.hyper_reg, op.idx_attr))


@dataclass
class InsertBlockConversion(RewritePattern):
    """Op conversion pattern from qecl.insert_block -> qecp.insert_block."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qecl.InsertCodeblockOp, rewriter: PatternRewriter
    ) -> qecp.InsertCodeblockOp:
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        assert isinstance(
            op.in_hyper_reg.type, qecp.PhysicalHyperRegisterType
        ), "lowering of hyper-register types is expected before lowering insert_block ops"
        rewriter.replace_op(op, qecp.InsertCodeblockOp(op.in_hyper_reg, op.idx_attr, op.codeblock))


# MARK: Conversion Pass


@dataclass(frozen=True)
class ConvertQecLogicalToQecPhysicalPass(ModulePass):
    """
    Convert QEC logical instructions to QEC physical instructions.
    """

    name = "convert-qecl-to-qecp"

    qec_code: QecCode

    # To specify the number of errors to be injected in the noise subroutine,
    # which is needed for the convert-qecl-noise-to-qecp-noise pass.
    number_errors: int = 1

    def __post_init__(self):
        # This method handles the case where `qec_code` is given as a dictionary rather than a
        # `QecCode` object. This is possible when the pass is registered in the IR and applied via
        # the `transform.apply_registered_pass` op, in which case the QecCode pass option is
        # represented as a dictionary attribute. Converting it from this dictionary attribute back
        # to a QecCode object allows for regular usage of this variable.
        if isinstance(self.qec_code, dict):
            # Because the class is frozen, we cannot assign to self.qec_code directly.
            # We use object.__setattr__ to bypass the frozen restriction.
            new_code = QecCode.from_dict(self.qec_code)
            object.__setattr__(self, "qec_code", new_code)

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-to-qecp pass."""
        if self.qec_code.k != 1:
            raise NotImplementedError(
                f"The {self.name} pass only supports QEC codes where the number of logical qubits "
                f"per codeblock, k, is 1, but got k = {self.qec_code.k}"
            )

        # n is the number of physical data qubits from the QEC code.
        ConvertQECLNoiseOpToQECPNoisePass(
            n=self.qec_code.n, number_errors=self.number_errors
        ).apply(ctx, op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CodeblockTypeConversion(qec_code=self.qec_code),
                    HyperRegisterTypeConversion(qec_code=self.qec_code),
                    AllocationConversion(),
                    DeallocationConversion(),
                    InsertBlockConversion(),
                    ExtractBlockConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
