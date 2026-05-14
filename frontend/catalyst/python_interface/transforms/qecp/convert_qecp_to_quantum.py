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
from typing import cast

from xdsl import pattern_rewriter
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Attribute, SSAValue
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
from catalyst.python_interface.inspection.xdsl_conversion import resolve_constant_params
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

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
    op: qecp.ExtractQubitOp | qecp.InsertQubitOp,
    rewriter: PatternRewriter,
) -> IntegerAttr | SSAValue:
    """Get an i64 index value/attr for quantum.extract/insert from a qecp extract/insert op."""
    if op.idx is not None:
        if isinstance(op.idx.type, IntegerType) and op.idx.type.width.data == 64:
            return op.idx
        if isinstance(op.idx.type, IndexType):
            index_cast_op = arith.IndexCastOp(op.idx, builtin.i64)
            rewriter.insert_op(index_cast_op)
            return index_cast_op.result
        raise TypeError(
            f"Expected idx value '{op.idx}' to have type IndexType or i64, got {op.idx.type}"
        )
    if op.idx_attr is not None:
        return IntegerAttr(op.idx_attr.value.data, 64)
    raise ValueError(f"Both idx and idx_attr of op '{op}' are None")


def _convert_qecp_type(typ: Attribute) -> Attribute:
    if isinstance(typ, qecp.PhysicalCodeblockType):
        return quantum.QuregType()
    if isinstance(typ, qecp.QecPhysicalQubitType):
        return quantum.QubitType()
    return typ


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
        """Op conversion rewrite pattern for lowering ops that extract a data qubit from a quantum.reg."""
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
    """Op conversion pattern from qecp.rot to quantum.custom. Note that this pattern is for validation purposes
    only and is separated from the general GateConversion pattern."""

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
    """Op conversion pattern from qecp.measure to quantum.measure."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecp.MeasureOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering measurement ops in qecp to quantum.measure ops."""
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
class RemoveEncodeLoop(RewritePattern):
    """Op conversion pattern for lowering hyperregister-related qecp ops to quantum ops. This includes
    qecp.create_hyperregister and qecp.hyperregister_extract/insert ops."""

    def _is_exact_encode_zero_steane_loop(self, for_op: scf.ForOp) -> bool:
        """Return true iff `for_op` is exactly the qecp encode loop shape."""
        if len(for_op.iter_args) != 1 or len(for_op.results) != 1:
            return False
        block = for_op.body.block
        if len(block.args) != 2:
            return False
        iv = block.args[0]
        carried_hreg = block.args[1]
        body_ops = list(block.ops)
        if len(body_ops) != 4:
            return False
        extract_op, call_op, insert_op, yield_op = body_ops
        if not isinstance(extract_op, qecp.ExtractCodeblockOp):
            return False
        if not isinstance(call_op, func.CallOp):
            return False
        if not isinstance(insert_op, qecp.InsertCodeblockOp):
            return False
        if not isinstance(yield_op, scf.YieldOp):
            return False
        if call_op.callee.string_value().find("encode_zero_") == -1:
            return False
        if extract_op.hyper_reg is not carried_hreg:
            return False
        if extract_op.idx is not iv:
            return False
        if tuple(call_op.arguments) != (extract_op.codeblock,):
            return False
        if len(call_op.results) != 1:
            return False
        if insert_op.in_hyper_reg is not carried_hreg:
            return False
        if insert_op.idx is not iv:
            return False
        if insert_op.codeblock is not call_op.results[0]:
            return False
        if tuple(yield_op.operands) != (insert_op.out_hyper_reg,):
            return False
        return call_op.callee.string_value()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp | qecp.ExtractCodeblockOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering hyperregister-related qecp ops to quantum ops."""
        if isinstance(op, scf.ForOp) and self._is_exact_encode_zero_steane_loop(op):
            init_hreg = op.iter_args[0]
            if isinstance(init_hreg.owner, qecp.AllocOp):
                op.results[0].replace_all_uses_with(init_hreg)

                n_regs = init_hreg.type.width.value.data
                num_qubits = init_hreg.type.n.value.data

                for i in range(n_regs):
                    reg = quantum.AllocOp(num_qubits)
                    rewriter.insert_op(reg)

                    encode_subroutine_symbol = self._is_exact_encode_zero_steane_loop(op)
                    args = (reg.results[0],)
                    call_op = func.CallOp(
                        builtin.SymbolRefAttr(encode_subroutine_symbol),
                        args,
                        (reg.results[0].type,),
                    )
                    rewriter.insert_op(call_op)

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
                    AllocAuxQubitConversion(),
                    DeallocAuxQubitConversion(),
                    InsertQubitConversion(),
                    ExtractQubitConversion(),
                    CliffordGateConversion(),
                    NoiseRotConversion(),
                    MeasureConversion(),
                    SubroutineSignatureConversion(),
                    SubroutineCallOpSignatureConversion(),
                    RemoveEncodeLoop(),
                ]
            )
        ).rewrite_module(op)

        module_op = op
        for op_ in module_op.walk():
            if isinstance(op_, func.FuncOp) and "quantum.node" in op_.attributes:
                qecp_alloc_op = None
                regs = []
                reg_idx = 0
                qecp_ops_to_remove = []

                for quantum_op in op_.walk():
                    if isinstance(quantum_op, qecp.AllocOp):
                        qecp_alloc_op = quantum_op
                        qecp_ops_to_remove.append(quantum_op)
                    if isinstance(quantum_op, func.CallOp) and quantum_op.callee.string_value() == "encode_zero_Steane":
                        regs.append(quantum_op.results[0])
                    if isinstance(quantum_op, qecp.ExtractCodeblockOp):
                        qecp_ops_to_remove.append(quantum_op)
                        quantum_op.codeblock.replace_all_uses_with(regs[reg_idx])
                        reg_idx += 1
                    if isinstance(quantum_op, qecp.InsertCodeblockOp):
                        qecp_ops_to_remove.append(quantum_op)
                        rewriter = pattern_rewriter.PatternRewriter(quantum_op)
                        idx = resolve_constant_params(quantum_op.idx)
                        dealloc = quantum.DeallocOp(regs[idx])
                        rewriter.insert_op(dealloc)
                        quantum_op.results[0].replace_all_uses_with(qecp_alloc_op.results[0])
                    if isinstance(quantum_op, qecp.DeallocOp):
                        rewriter = pattern_rewriter.PatternRewriter(quantum_op)
                        rewriter.erase_op(quantum_op)
                for quantum_op in reversed(qecp_ops_to_remove):
                    rewriter = pattern_rewriter.PatternRewriter(quantum_op)
                    rewriter.erase_op(quantum_op)
                
                print(op_)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
