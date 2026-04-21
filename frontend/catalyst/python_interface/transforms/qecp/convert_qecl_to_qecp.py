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

from dataclasses import dataclass, field

from xdsl import builder
from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects import qecl, qecp
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

from .qec_code_lib import QecCode

# MARK: Type Conversion Pattern


@dataclass
class CodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecl.codeblock -> qecp.codeblock."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalCodeblockType) -> qecp.PhysicalCodeblockType:
        """Type conversion rewrite pattern for logical codeblock types."""
        return qecp.PhysicalCodeblockType(typ.k, self.qec_code.n)


@dataclass
class HyperRegisterTypeConversion(TypeConversionPattern):
    """Hyper-register type conversion pattern from qecl.hyperreg -> qecp.hyperreg."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalHyperRegisterType) -> qecp.PhysicalHyperRegisterType:
        """Type conversion rewrite pattern for physical codeblock types."""
        return qecp.PhysicalHyperRegisterType(typ.width, typ.k, self.qec_code.n)


# MARK: Encode Op Pattern


class EncodeOpConversion(RewritePattern):
    """Converts qecl.encode [zero] to the equivalent subroutine of qecp gates"""

    def __init__(self, qec_code: QecCode, encode_subroutine: func.FuncOp):
        self.qec_code = qec_code
        self.encode_subroutine = encode_subroutine

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.EncodeOp, rewriter: PatternRewriter):
        """Rewrite pattern for `qecl.encode [zero]` op"""

        if not op.init_state.data == "zero":
            raise NotImplementedError(
                "Lowering qecl.EncodeOp to the qecp dialect is only implemented for init_state 'zero'"
            )

        if (k := op.in_codeblock.type.k.value.data) != self.qec_code.k:
            raise CompileError(
                f"Circuit expressed in the qecl dialect with k={k} is not compatible with lowering to a code with k={self.qec_code.k}"
            )

        callee = builtin.SymbolRefAttr(self.encode_subroutine.sym_name)
        arguments = (op.in_codeblock,)
        return_types = self.encode_subroutine.function_type.outputs.data
        callOp = func.CallOp(callee, arguments, return_types)

        rewriter.replace_op(op, callOp)


# MARK: Conversion Pass


@dataclass(frozen=True)
class ConvertQecLogicalToQecPhysicalPass(ModulePass):
    """
    Convert QEC logical instructions to QEC physical instructions.
    """

    name = "convert-qecl-to-qecp"

    qec_code: QecCode

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-to-qecp pass."""

        assert op.regions[0].blocks.first is not None

        encode_funcop = self.create_encode_cycle()
        op.regions[0].blocks.first.add_op(encode_funcop)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CodeblockTypeConversion(qec_code=self.qec_code),
                    HyperRegisterTypeConversion(qec_code=self.qec_code),
                    EncodeOpConversion(qec_code=self.qec_code, encode_subroutine=encode_funcop),
                ]
            )
        ).rewrite_module(op)

    def create_encode_cycle(self):
        # ToDo: docstring

        codeblock_type = qecp.PhysicalCodeblockType(self.qec_code.k, self.qec_code.n)
        input_types = (codeblock_type,)
        output_types = (codeblock_type,)

        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):

            (codeblock,) = block.args

            aux_allocate_ops = (qecp.AllocAuxQubitOp() for row in self.qec_code.z_tanner)
            aux_qubits = [op.results[0] for op in aux_allocate_ops]

            measure_ops, codeblock = self._z_check_pattern(aux_qubits, codeblock)

            dealloc_ops = [qecp.DeallocAuxQubitOp(meas_op.results[1]) for meas_op in measure_ops]

            func.ReturnOp(codeblock)

        region = Region([block])

        symbol_name = f"encode_zero_{self.qec_code.name}"
        funcOp = func.FuncOp(
            symbol_name,
            (input_types, output_types),
            visibility="private",
            region=region,
        )

        return funcOp

    def _z_check_pattern(self, aux_qbs, codeblock):
        # ToDo: add docstring

        tanner_graph = self.qec_code.z_tanner

        hadamard_ops = [qecp.HadamardOp(qb) for qb in aux_qbs]  # skip for X check
        aux_qbs = [h_op.results[0] for h_op in hadamard_ops]

        extract_ops = [qecp.ExtractQubitOp(codeblock, i) for i in range(self.qec_code.n)]
        data_qbs = [ext_op.results[0] for ext_op in extract_ops]

        aux_qbs, data_qbs = self._cnot_pattern(aux_qbs, data_qbs, check_type="Z")

        for i in range(self.qec_code.n):
            insert_op = qecp.InsertQubitOp(codeblock, i, data_qbs[i])
            codeblock = insert_op.results[0]

        hadamard_ops2 = [qecp.HadamardOp(aux) for aux in aux_qbs]  # skip for X check
        measure_ops = [qecp.MeasureOp(h_op.results[0]) for h_op in hadamard_ops2]

        return measure_ops, codeblock

    def _x_check_pattern(self, aux_qbs, codeblock):
        # ToDo: add docstring

        tanner_graph = self.qec_code.z_tanner

        extract_ops = [qecp.ExtractQubitOp(codeblock, i) for i in range(self.qec_code.n)]
        data_qbs = [ext_op.results[0] for ext_op in extract_ops]

        aux_qbs, data_qbs = self._cnot_pattern(aux_qbs, data_qbs, check_type="Z")

        measure_ops = [qecp.MeasureOp(qb) for qb in aux_qbs]

        for i in range(self.qec_code.n):
            insert_op = qecp.InsertQubitOp(codeblock, i, data_qbs[i])
            codeblock = insert_op.results[0]

        return measure_ops, codeblock

    def _cnot_pattern(self, aux_qubits, data_qubits, check_type):
        # ToDo: docstring

        aux_qbs_out = []
        if check_type == "Z":
            tanner_graph = self.qec_code.z_tanner
        elif check_type == "X":
            tanner_graph = (
                self.qec_code.x_tanner
            )  # ToDo: reconsider all of this once the QecCode PR is merged

        for aux_qb, row in zip(aux_qubits, tanner_graph):
            indices = [idx for idx, val in enumerate(row) if val]
            for idx in indices:
                if check_type == "Z":
                    cnot_op = qecp.CnotOp(aux_qb, data_qubits[idx])
                elif check_type == "X":
                    cnot_op = qecp.CnotOp(data_qubits[idx], aux_qb)
                aux_qb, data_qb = cnot_op.results
                data_qubits[idx] = data_qb
            aux_qbs_out.append(aux_qb)

        return aux_qbs_out, data_qubits


convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
