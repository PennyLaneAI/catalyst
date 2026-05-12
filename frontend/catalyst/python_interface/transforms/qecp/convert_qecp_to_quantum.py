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
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.ir import SSAValue
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from catalyst.python_interface.dialects import qecp, quantum
from catalyst.python_interface.dialects.quantum.attributes import QubitType, QuregType
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

from typing import cast

_QECP_GATENAMES_TO_QUANTUM_OPS = {
    "qecp.hadamard": "Hadamard",
    "qecp.identity": "Identity",
    "qecp.s": "S",
    "qecp.x": "PauliX",
    "qecp.y": "PauliY",
    "qecp.z": "PauliZ",
    "qecp.cnot": "CNOT",
}


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
            index_cast_op = arith.IndexCastOp(op.idx, IndexType())
            rewriter.insert_op(index_cast_op)
            idx = cast(SSAValue[IndexType], index_cast_op.result)
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


# MARK: Type Conversion Pattern


@dataclass
class PhysicalCodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecp.codeblock -> quantum.reg."""

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: qecp.PhysicalCodeblockType  # pylint: disable=unused-argument
    ) -> QuregType:
        """Type conversion rewrite pattern for physical codeblock types."""

        return QuregType()


@dataclass
class QecPhysicalQubitTypeConversion(TypeConversionPattern):
    """Qubit type conversion pattern from qecp.qubit -> quantum.bit."""

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: qecp.QecPhysicalQubitType  # pylint: disable=unused-argument
    ) -> QubitType:
        """Type conversion rewrite pattern for QEC physical qubit types."""

        return QubitType()


# MARK: Auxiliary qubit Alloc/Dealloc Patterns


@dataclass(frozen=True)
class AllocAuxQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.alloc_aux -> quantum.alloc_qb."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.AllocAuxQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate an auxiliary qubit."""
        rewriter.replace_op(op, quantum.AllocQubitOp())


@dataclass(frozen=True)
class DeallocAuxQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.dealloc_aux -> quantum.dealloc_qb."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.DeallocAuxQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that deallocate an auxiliary qubit."""
        rewriter.replace_op(op, quantum.DeallocQubitOp(op.qubit))


# MARK: Data qubit extract and insertion patterns
@dataclass(frozen=True)
class ExtractQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.extract -> quantum.extract."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.ExtractQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that extract a data qubit from a quantum.reg."""
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        rewriter.replace_op(op, quantum.ExtractOp(op.codeblock, idx=idx))


@dataclass(frozen=True)
class InsertQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.insert -> quantum.insert."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.InsertQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that insert a data qubit into a quantum.reg."""
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)
        insert_op = quantum.InsertOp(op.in_codeblock, idx=idx, qubit=op.qubit)
        rewriter.replace_op(op, insert_op)


# MARK: Gate/Measurement conversion patterns


@dataclass(frozen=True)
class GateConversion(RewritePattern):
    """Op conversion pattern from qecp.gate -> quantum.gate."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qecp.SingleQubitPhysicalGateOp | qecp.CnotOp, rewriter: PatternRewriter
    ):
        """Op conversion rewrite pattern for lowering Clifford gate ops to quantum.gate ops."""
        gate_name = _QECP_GATENAMES_TO_QUANTUM_OPS.get(op.name)

        adjoint = op.properties.get("adjoint", False)

        gate_op = quantum.CustomOp(gate_name=gate_name, in_qubits=op.operands, adjoint=adjoint)
        rewriter.replace_op(op, gate_op)


@dataclass(frozen=True)
class NoiseRotConversion(RewritePattern):
    """Op conversion pattern from qecp.noise_rot_c -> quantum.noise_rot_c."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.RotOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering noise rotation ops to quantum.custom "Rot" ops."""
        gate_name = "Rot"
        params = (op.phi, op.theta, op.omega)
        gate_op = quantum.CustomOp(
            gate_name=gate_name, params=params, in_qubits=(op.in_qubit,), adjoint=False
        )
        rewriter.replace_op(op, gate_op)


@dataclass(frozen=True)
class MeasureConversion(RewritePattern):
    """Op conversion pattern from qecp.measure -> quantum.measure."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.MeasureOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering measurement ops to quantum.measure ops."""
        measure_op = quantum.MeasureOp(in_qubit=op.operands[0])
        rewriter.replace_op(op, measure_op)


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
                    AllocAuxQubitConversion(),
                    DeallocAuxQubitConversion(),
                    InsertQubitConversion(),
                    ExtractQubitConversion(),
                    GateConversion(),
                    NoiseRotConversion(),
                    MeasureConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
