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
from xdsl.dialects import builtin, func
from xdsl.ir import Attribute
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
from catalyst.python_interface.dialects.quantum.attributes import QubitType, QuregType
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

from ..qecl.convert_quantum_to_qecl import _get_idx_value_or_attr_from_extract_or_insert_op

_QECP_GATENAMES_TO_QUANTUM_OPS = {
    "qecp.hadamard": "Hadamard",
    "qecp.identity": "Identity",
    "qecp.s": "S",
    "qecp.x": "PauliX",
    "qecp.y": "PauliY",
    "qecp.z": "PauliZ",
    "qecp.cnot": "CNOT",
}


def _convert_qecp_type(typ: Attribute) -> Attribute:
    """Helper function to convert qecp types to quantum types."""
    if isinstance(typ, qecp.PhysicalCodeblockType):
        return quantum.QuregType()
    if isinstance(typ, qecp.QecPhysicalQubitType):
        return quantum.QubitType()
    return typ


# MARK: Type Conversion Pattern

@dataclass
class PhysicalCodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecp.codeblock to quantum.reg."""

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: qecp.PhysicalCodeblockType  # pylint: disable=unused-argument
    ) -> QuregType:
        """Type conversion rewrite pattern for physical codeblock types."""

        return QuregType()


@dataclass
class QecPhysicalQubitTypeConversion(TypeConversionPattern):
    """Qubit type conversion pattern from qecp.qubit to quantum.bit."""

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: qecp.QecPhysicalQubitType  # pylint: disable=unused-argument
    ) -> QubitType:
        """Type conversion rewrite pattern for QEC physical qubit types."""

        return QubitType()


# MARK: Auxiliary qubit Alloc/Dealloc Patterns


@dataclass(frozen=True)
class AllocAuxQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.alloc_aux to quantum.alloc_qb."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.AllocAuxQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate an auxiliary qubit."""
        rewriter.replace_op(op, quantum.AllocQubitOp())


@dataclass(frozen=True)
class DeallocAuxQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.dealloc_aux to quantum.dealloc_qb."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.DeallocAuxQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that deallocate an auxiliary qubit."""
        rewriter.replace_op(op, quantum.DeallocQubitOp(op.qubit))


# MARK: Data qubit extract and insertion patterns


@dataclass(frozen=True)
class ExtractQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.extract to quantum.extract."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.ExtractQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that extract a data qubit from a
        quantum.reg."""
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        rewriter.replace_op(op, quantum.ExtractOp(op.codeblock, idx=idx))


@dataclass(frozen=True)
class InsertQubitConversion(RewritePattern):
    """Op conversion pattern from qecp.insert to quantum.insert."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.InsertQubitOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops for data qubit insertion."""
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)
        insert_op = quantum.InsertOp(op.in_codeblock, idx=idx, qubit=op.qubit)
        rewriter.replace_op(op, insert_op)


# MARK: Gate/Measurement conversion patterns


@dataclass(frozen=True)
class CliffordGateConversion(RewritePattern):
    """Op conversion pattern from gates in qecp to quantum.custom."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qecp.SingleQubitPhysicalGateOp | qecp.CnotOp, rewriter: PatternRewriter
    ):
        """Op conversion rewrite pattern for lowering gate ops in the qecp to quantum.custom ops."""
        gate_name = _QECP_GATENAMES_TO_QUANTUM_OPS.get(op.name)

        adjoint = op.properties.get("adjoint", False)

        gate_op = quantum.CustomOp(gate_name=gate_name, in_qubits=op.operands, adjoint=adjoint)
        rewriter.replace_op(op, gate_op)


@dataclass(frozen=True)
class NoiseRotConversion(RewritePattern):
    """Op conversion pattern from qecp.rot to quantum.custom. Note that this pattern is for
    validation purposes only and is separated from the general GateConversion pattern."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.RotOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering noise rotation ops to quantum.custom
        "Rot" ops."""
        gate_name = "Rot"
        params = (op.phi, op.theta, op.omega)
        gate_op = quantum.CustomOp(
            gate_name=gate_name, params=params, in_qubits=(op.in_qubit,), adjoint=False
        )
        rewriter.replace_op(op, gate_op)


@dataclass(frozen=True)
class MeasureConversion(RewritePattern):
    """Op conversion pattern from qecp.measure to quantum.measure."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.MeasureOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering measurement ops in qecp to
        quantum.measure ops."""
        measure_op = quantum.MeasureOp(in_qubit=op.operands[0])
        rewriter.replace_op(op, measure_op)


# MARK: Subroutine conversion patterns for encoder, decoder, qeccycle, and logical operations.
# These patterns convert the signatures of subroutines and their call sites to replace
# qecp.codeblock and qecp.qubit types with quantum.reg and quantum.bit types, respectively.


@dataclass(frozen=True)
class SubroutineSignatureConversion(RewritePattern):
    """Op conversion pattern from subroutines with qecp.codeblock/qecp.qubit types."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering subroutines with qecp.codeblock/qecp.qubit types in their signatures."""
        old_inputs = tuple(op.function_type.inputs.data)
        old_outputs = tuple(op.function_type.outputs.data)
        new_inputs = tuple(_convert_qecp_type(t) for t in old_inputs)
        new_outputs = tuple(_convert_qecp_type(t) for t in old_outputs)
        if new_inputs == old_inputs and new_outputs == old_outputs:
            return
        op.function_type = builtin.FunctionType.from_lists(
            list(new_inputs),
            list(new_outputs),
        )

        rewriter.notify_op_modified(op)


@dataclass(frozen=True)
class SubroutineCallOpSignatureConversion(RewritePattern):
    """Op conversion pattern for calls to subroutines with qecp.codeblock/qecp.qubit types."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering call ops with qecp.codeblock/qecp.qubit types in their operands or results."""

        old_results = tuple(op.result_types)
        new_results = tuple(_convert_qecp_type(t) for t in old_results)
        if new_results == old_results:
            return
        new_call = func.CallOp(op.callee, op.arguments, new_results)
        rewriter.replace_op(op, new_call)


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
                    CliffordGateConversion(),
                    NoiseRotConversion(),
                    MeasureConversion(),
                    SubroutineSignatureConversion(),
                    SubroutineCallOpSignatureConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
