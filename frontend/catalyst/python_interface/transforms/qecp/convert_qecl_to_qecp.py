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
import numpy as np

from dataclasses import dataclass
from enum import StrEnum

from xdsl import builder
from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from catalyst.python_interface.dialects import qecl, qecp
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError


class QecCodeKey(StrEnum):
    """The set of supported QEC codes."""

    # List the supported QEC codes here, e.g.
    STEANE_7_1_3 = "steane[[7,1,3]]"

@dataclass
class QECCode:
    n: int
    k: int
    x_tanner: np.array
    z_tanner: np.array

Steane713 = QECCode(
    7, 
    1, 
    np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]), 
    np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
    )

QEC_codes = {"steane[[7,1,3]]": Steane713}


# MARK: Conversion Pass


@dataclass(frozen=True)
class ConvertQecLogicalToQecPhysicalPass(ModulePass):
    """
    Convert QEC logical instructions to QEC physical instructions.
    """

    name = "convert-qecl-to-qecp"

    qec_code: QecCodeKey

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-to-qecp pass."""

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    EncodeOpConversion(self.qec_code),
                ]
            )
        ).rewrite_module(op)

        # Certain patterns leave behind `builtin.unrealized_conversion_cast` ops;
        # this pass removes them
        ReconcileUnrealizedCastsPass().apply(ctx, op)


# MARK: Encode Op Pattern


class EncodeOpConversion(RewritePattern):
    """Converts qecl.encode [zero] to the equivalent subroutine of qecp gates"""

    def __init__(self, qec_code: str):
        self.code_name = qec_code

        code = QEC_codes[qec_code]
        self._n = code.n
        self._k = code.k
        self._x_tanner = code.x_tanner
        self._z_tanner = code.z_tanner

    # def _create_encode_subroutine(self):
    #     codeblock_type = qecp.PhysicalCodeblockType(self._k, self._n)
    #     input_types = (codeblock_type,)
    #     output_types = (codeblock_type,)

    #     block = Block(arg_types=input_types)

    #     with builder.ImplicitBuilder(block):
    #         (in_codeblock,) = block.args
    #         func.ReturnOp(in_codeblock)

    #     region = Region([block])
    #     symbol_name = f"encode_zero_{self.code_name}"
    #     funcOp = func.FuncOp(
    #         symbol_name,
    #         (input_types, output_types),
    #         visibility="private",
    #         region=region,
    #     )

    #     return funcOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.EncodeOp, rewriter: PatternRewriter):
        """Rewrite pattern for `qecl.encode [zero]` op"""
    
        if not op.init_state.data == "zero":
            raise NotImplementedError("Lowering qecl.EncodeOp to the qecp dialect is only implemented for init_state 'zero'")

        k = op.in_codeblock.type.k.value.data
        if not k == self._k:
            raise CompileError(f"Circuit expressed in the qecl dialect with k={k} is not compatible with lowering to a code with k={self._k}")

        # # we will only need to do this the first time!!!
        # # Insert encode subroutine into the module.
        # encode_subroutine = self._create_encode_subroutine()
        # assert op.regions[0].blocks.first is not None
        # op.regions[0].blocks.first.add_op(encode_subroutine)
    
        in_block_cast = builtin.UnrealizedConversionCastOp.get(
            (op.in_codeblock,), (qecp.PhysicalCodeblockType(self._k, self._n),)
        )

        # ops_to_insert = [in_block_cast,]
        # # mock-up of code, not correct
        for row in self._x_tanner:
            alloc_op = qecp.AllocAuxQubitOp()
        #     hadamard_op = qecp.HadamardOp(alloc_op.results[0])
        #     for val, idx in enumerate(row):
        #         if val:
        #             #probably need to insert the idx
        #             extract_op = qecp.ExtractQubitOp(in_block_cast.results[0], idx)
        #             cnot = qecp.CnotOp(hadamard_op.results[0], extract_op.results[0])

        rewriter.insert_op(in_block_cast, InsertPoint.before(op))
        # rewriter.replace_op(op, *ops_to_insert)

        rewriter.erase_op(op)



convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
