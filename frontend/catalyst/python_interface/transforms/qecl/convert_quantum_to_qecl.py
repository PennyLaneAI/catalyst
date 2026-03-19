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
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Attribute, Operation, SSAValue, TypeAttribute
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

# MARK: Conversion Patterns


@dataclass(frozen=True)
class AllocOpConversion(RewritePattern):
    """Converts `quantum.alloc` ops to equivalent `qecl.alloc` ops."""

    k: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.AllocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.alloc` ops."""
        nqubits_attr = op.get_attr_or_prop("nqubits_attr")
        if nqubits_attr is None:
            raise NotImplementedError(
                f"Failed to convert op '{op}': conversion pattern for '{op.name}' does not support "
                f"a dynamic number of qubits"
            )

        _assert_attribute_type(nqubits_attr, IntegerAttr, "nqubits_attr", op)
        nqubits_attr = cast(IntegerAttr, nqubits_attr)
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
        _assert_operand_type(hyper_reg, qecl.LogicalHyperRegisterType, "qreg", op)
        hyper_reg = cast(qecl.LogicalHyperRegisterSSAValue, hyper_reg)

        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

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
                qecl.QecCycleOp(
                    in_codeblock=cast(qecl.LogicalCodeBlockSSAValue, encode_op.results[0])
                ),
            ]
        else:
            ops_to_insert = [qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=idx)]

        rewriter.replace_op(op, ops_to_insert)


class InsertOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.insert_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.InsertOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.insert` ops."""

        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        in_hyper_reg = op.in_qreg
        _assert_operand_type(in_hyper_reg, qecl.LogicalHyperRegisterType, "in_qreg", op)
        in_hyper_reg = cast(qecl.LogicalHyperRegisterSSAValue, in_hyper_reg)

        codeblock = op.qubit
        _assert_operand_type(codeblock, qecl.LogicalCodeblockType, "qubit", op)
        codeblock = cast(qecl.LogicalCodeBlockSSAValue, codeblock)

        rewriter.replace_op(
            op, qecl.InsertCodeblockOp(in_hyper_reg=in_hyper_reg, idx=idx, codeblock=codeblock)
        )


class DeallocOpConversion(RewritePattern):
    """Converts `quantum.dealloc` ops to equivalent `qecl.dealloc` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.DeallocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.dealloc` ops."""

        hyper_reg = op.qreg
        _assert_operand_type(hyper_reg, qecl.LogicalHyperRegisterType, "qreg", op)
        hyper_reg = cast(qecl.LogicalHyperRegisterSSAValue, hyper_reg)

        rewriter.replace_op(op, qecl.DeallocOp(hyper_reg=hyper_reg))


class CustomOpConversion(RewritePattern):
    """Converts `quantum.custom` ops to equivalent `qecl.hadamard`, `qecl.s` and `qecl.cnot` ops.

    NOTES
    -----

    This conversion pattern assumes that k = 1, and as such always applies the logical gate
    operation to the codeblock at index 0. This simplification will need to be addressed when the
    quantum-to-qecl dialect conversion supports arbitrary values of k >= 1.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.CustomOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.custom` ops."""
        gate_name = op.gate_name.data

        match gate_name:
            case "Hadamard":
                codeblock = op.in_qubits[0]
                _assert_operand_type(codeblock, qecl.LogicalCodeblockType, "in_qubits[0]", op)
                codeblock = cast(qecl.LogicalCodeBlockSSAValue, codeblock)
                assert codeblock.type.k.value.data == 1

                op_to_insert = qecl.HadamardOp(in_codeblock=codeblock, idx=0)

            case "S":
                codeblock = op.in_qubits[0]
                _assert_operand_type(codeblock, qecl.LogicalCodeblockType, "in_qubits[0]", op)
                codeblock = cast(qecl.LogicalCodeBlockSSAValue, codeblock)
                assert codeblock.type.k.value.data == 1
                adjoint = True if op.properties.get("adjoint") else False

                op_to_insert = qecl.SOp(in_codeblock=codeblock, idx=0, adjoint=adjoint)

            case "CNOT":
                ctrl_codeblock = op.in_qubits[0]
                trgt_codeblock = op.in_qubits[1]
                _assert_operand_type(ctrl_codeblock, qecl.LogicalCodeblockType, "in_qubits[0]", op)
                _assert_operand_type(trgt_codeblock, qecl.LogicalCodeblockType, "in_qubits[1]", op)
                ctrl_codeblock = cast(qecl.LogicalCodeBlockSSAValue, ctrl_codeblock)
                trgt_codeblock = cast(qecl.LogicalCodeBlockSSAValue, trgt_codeblock)
                assert ctrl_codeblock.type.k.value.data == 1
                assert trgt_codeblock.type.k.value.data == 1

                op_to_insert = qecl.CnotOp(
                    in_ctrl_codeblock=ctrl_codeblock,
                    idx_ctrl=0,
                    in_trgt_codeblock=trgt_codeblock,
                    idx_trgt=0,
                )

            case _:
                raise NotImplementedError(
                    f"Conversion of op '{op.name}' only supports gates 'Hadamard', 'S' and 'CNOT', "
                    f"but got {gate_name}"
                )

        rewriter.replace_op(op, op_to_insert)


class MeasureOpConversion(RewritePattern):
    """Converts `quantum.measure` ops to equivalent `qecl.measure` ops.

    NOTES
    -----

    This conversion pattern assumes that k = 1, and as such always applies the logical measurement
    operation to the codeblock at index 0. This simplification will need to be addressed when the
    quantum-to-qecl dialect conversion supports arbitrary values of k >= 1.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.MeasureOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.measure` ops."""
        codeblock = op.in_qubit
        _assert_operand_type(codeblock, qecl.LogicalCodeblockType, "in_qubit", op)
        codeblock = cast(qecl.LogicalCodeBlockSSAValue, codeblock)
        assert codeblock.type.k.value.data == 1
        rewriter.replace_op(op, qecl.MeasureOp(in_codeblock=codeblock, idx=0))


# MARK: Helpers


def _assert_operand_type(
    operand: SSAValue, expected_type: type[TypeAttribute], operand_name: str, op: Operation
):
    """Helper function to assert that an operand of an op has the expected type."""
    assert isinstance(operand.type, expected_type), (
        f"Expected operand '{operand_name}' of {op.name} op to have type "
        f"'{type(expected_type).__name__}', but got {type(operand.type)}"
    )


def _assert_attribute_type(
    attr: Attribute, expected_type: type[Attribute], attr_name: str, op: Operation
):
    """Helper function to assert that an attribute of an op has the expected type."""
    assert isinstance(attr, expected_type), (
        f"Expected attribute '{attr_name}' of {op.name} op to have type "
        f"'{type(expected_type).__name__}', but got {type(attr)}"
    )


def _get_idx_value_or_attr_from_extract_or_insert_op(
    op: quantum.ExtractOp | quantum.InsertOp, rewriter: PatternRewriter
) -> IntegerAttr | SSAValue[IndexType]:
    """Helper function to get the index value 'idx' or attribute 'idx_attr' from a `quantum.extract`
    or `quantum.insert` op.

    If the index value has type IntegerType, an `arith.cast_index` op is inserted to cast it to type
    IndexType. We must cast such values because `qecl.extract_block` ops expect an idx operand of
    type IndexType.
    """
    if op.idx is not None:
        if isinstance(op.idx.type, IndexType):
            idx = cast(SSAValue[IndexType], op.idx)
        elif isinstance(op.idx.type, IntegerType):
            # Insert cast operation integer -> index
            arith_op = arith.IndexCastOp(op.idx, IndexType())
            rewriter.insert_op(arith_op)
            idx = cast(SSAValue[IndexType], arith_op.result)
        else:
            assert False, (
                f"Expected idx value '{op.idx}' to have type 'IndexType' or 'IntegerType', "
                f"but got {op.idx.type}"
            )

    elif op.idx_attr is not None:
        idx = op.idx_attr

    else:
        assert False, f"Both idx and idx_attr of op '{op}' are None"

    return idx


# MARK: Conversion Pass


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
