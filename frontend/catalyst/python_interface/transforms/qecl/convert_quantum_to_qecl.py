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

"""Quantum to QEC Logical dialect conversion.

This module contains the implementation of the xDSL quantum-to-qecl dialect-conversion pass.
"""

import math
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.builtin import IntegerAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from catalyst.python_interface.dialects import qecl, quantum
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform


@dataclass(frozen=True)
class AllocOpConversion(RewritePattern):
    """Converts `quantum.alloc` ops to equivalent `qecl.alloc` ops."""

    k: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.AllocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.alloc` ops."""
        nqubits_attr = op.properties.get("nqubits_attr")
        if nqubits_attr is None:
            raise NotImplementedError(
                f"Failed to convert op '{op}': conversion pattern for '{op.name}' does not support "
                f"a dynamic number of qubits"
            )

        assert isinstance(
            nqubits_attr, IntegerAttr
        ), f"Expected 'nqubits_attr' to be an IntegerAttr, but got {type(nqubits_attr)}"

        nqubits = nqubits_attr.value.data

        hyper_reg_width = math.ceil(nqubits / self.k)
        rewriter.replace_op(
            op,
            qecl.AllocOp(qecl.LogicalHyperRegisterType(width=hyper_reg_width, k=self.k)),
        )


@dataclass(frozen=True)
class ExtractOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.extract_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.ExtractOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.extract` ops."""

        hyper_reg = op.qreg
        assert isinstance(
            hyper_reg.type, qecl.LogicalHyperRegisterType
        ), f"Expected 'hyper_reg' to be a LogicalHyperRegisterType, but got {type(hyper_reg)}"

        idx = op.idx if op.idx is not None else op.idx_attr
        assert idx is not None, "Both idx and idx_attr are null"

        # If we extract immediately after an alloc operation, we must also insert ops to perform
        # encoding and a QEC cycle. Cases where we don't immediately extract are typically around
        # control-flow operations.
        hyper_reg_owner = op.qreg.owner
        if isinstance(hyper_reg_owner, qecl.AllocOp):
            ops_to_insert = [
                extract_codeblock_op := qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=idx),
                encode_op := qecl.EncodeOp(
                    in_codeblock=extract_codeblock_op,
                    init_state=qecl.LogicalCodeblockInitState.Zero,
                ),
                qecl.QecCycleOp(in_codeblock=encode_op.results[0]),
            ]
        else:
            ops_to_insert = [qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=idx)]

        rewriter.replace_op(op, ops_to_insert)


class InsertOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.insert_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.InsertOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.insert` ops."""

        idx = op.idx if op.idx is not None else op.idx_attr
        assert idx is not None, "Both idx and idx_attr are null"

        in_hyper_reg = op.in_qreg
        assert isinstance(
            in_hyper_reg.type, qecl.LogicalHyperRegisterType
        ), f"Expected 'in_hyper_reg' to be a LogicalHyperRegisterType, but got {type(in_hyper_reg)}"

        codeblock = op.qubit
        assert isinstance(
            codeblock.type, qecl.LogicalCodeblockType
        ), f"Expected 'codeblock' to be a LogicalCodeblockType, but got {type(codeblock)}"

        rewriter.replace_op(
            op, qecl.InsertCodeblockOp(in_hyper_reg=in_hyper_reg, idx=idx, codeblock=codeblock)
        )


class DeallocOpConversion(RewritePattern):
    """Converts `quantum.dealloc` ops to equivalent `qecl.dealloc` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.DeallocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.dealloc` ops."""

        hyper_reg = op.qreg
        assert isinstance(
            hyper_reg.type, qecl.LogicalHyperRegisterType
        ), f"Expected 'hyper_reg' to be a LogicalHyperRegisterType, but got {type(hyper_reg)}"

        rewriter.replace_op(op, qecl.DeallocOp(hyper_reg=hyper_reg))


class CustomOpConversion(RewritePattern):
    """TODO"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.CustomOp, rewriter: PatternRewriter):
        gate_name = op.gate_name.data

        match gate_name:
            case "Hadamard":
                codeblock = op.in_qubits[0]
                assert isinstance(codeblock.type, qecl.LogicalCodeblockType)
                assert codeblock.type.k.value.data == 1
                op_to_insert = qecl.HadamardOp(in_codeblock=codeblock, idx=0)

            case "S":
                codeblock = op.in_qubits[0]
                assert isinstance(codeblock.type, qecl.LogicalCodeblockType)
                assert codeblock.type.k.value.data == 1
                adjoint = True if op.properties.get("adjoint") else False
                op_to_insert = qecl.SOp(in_codeblock=codeblock, idx=0, adjoint=adjoint)

            case "CNOT":
                ctrl_codeblock = op.in_qubits[0]
                trgt_codeblock = op.in_qubits[1]
                assert isinstance(ctrl_codeblock.type, qecl.LogicalCodeblockType)
                assert ctrl_codeblock.type.k.value.data == 1
                assert isinstance(trgt_codeblock.type, qecl.LogicalCodeblockType)
                assert trgt_codeblock.type.k.value.data == 1
                op_to_insert = qecl.CnotOp(
                    in_ctrl_codeblock=ctrl_codeblock,
                    idx_ctrl=0,
                    in_trgt_codeblock=trgt_codeblock,
                    idx_trgt=0,
                )

            case _:
                raise NotImplementedError(
                    f"Conversion of 'quantum.custom' op only supports gates 'Hadamard', 'S' and "
                    f"'CNOT', but got {gate_name}"
                )

        rewriter.replace_op(op, op_to_insert)


class MeasureOpConversion(RewritePattern):
    """TODO"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.MeasureOp, rewriter: PatternRewriter):
        """TODO"""
        codeblock = op.in_qubit
        assert isinstance(codeblock.type, qecl.LogicalCodeblockType)
        assert codeblock.type.k.value.data == 1
        rewriter.replace_op(op, qecl.MeasureOp(in_codeblock=codeblock, idx=0))


@dataclass(frozen=True)
class ConvertQuantumToQecLogicalPass(ModulePass):
    """
    Convert quantum instructions to QEC logical instructions.
    """

    name = "convert-quantum-to-qecl"

    k: int

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.k != 1:
            raise NotImplementedError(
                f"The {self.name} only supports QEC codes where the number of logical qubits per "
                f"codeblock, k, is 1, but got k = {self.k}"
            )

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AllocOpConversion(self.k),
                    ExtractOpConversion(),
                    InsertOpConversion(),
                    DeallocOpConversion(),
                    CustomOpConversion(),
                    MeasureOpConversion(),
                ]
            )
        ).rewrite_module(op)


convert_quantum_to_qecl_pass = compiler_transform(ConvertQuantumToQecLogicalPass)
