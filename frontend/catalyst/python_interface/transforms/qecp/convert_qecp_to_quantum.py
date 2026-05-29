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

  * The hyper-register lowering is experimental can only target programs with more than one logical
    codeblock, where there is a loop for encoding each logical codeblock. It's sufficient for the
    GHZ circuit. We might have to come back to this later.
  * The current hyper-register lowering implementation also does not support any control flow that
    iterates over hyper registers, except for the encoding loop.
"""

from dataclasses import dataclass
from typing import cast

from pennylane.exceptions import CompileError
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.builtin import I64, IndexType, IntegerAttr, i64
from xdsl.ir import SSAValue
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
from xdsl.transforms.dead_code_elimination import region_dce

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
    op: qecp.ExtractQubitOp | qecp.InsertQubitOp, rewriter: PatternRewriter
) -> IntegerAttr | SSAValue[IntegerAttr[I64]]:
    """Helper function to get the index value 'idx' or attribute 'idx_attr' from a `qecp.extract`
    or `qecp.insert` op.

    If the index value has type `index`, an `arith.cast_index` op is inserted to cast it to type
    `i64`. We must cast such values because `quantum.extract` and `quantum.insert` ops expect an idx
    operand of type `i64`.
    """
    if op.idx is not None:
        if isinstance(op.idx.type, IntegerAttr):
            idx = cast(SSAValue[IntegerAttr[I64]], op.idx)
        elif isinstance(op.idx.type, IndexType):
            # Insert cast operation index -> i64
            index_cast_op = arith.IndexCastOp(op.idx, i64)
            rewriter.insert_op(index_cast_op)
            idx = cast(SSAValue[IntegerAttr[I64]], index_cast_op.result)
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
class GateConversion(RewritePattern):
    """Op conversion pattern from gates in qecp to quantum.custom."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qecp.SingleQubitPhysicalGateOp | qecp.CnotOp, rewriter: PatternRewriter
    ):
        """Op conversion rewrite pattern for lowering gate ops in the qecp to quantum.custom ops."""
        gate_name = _QECP_GATENAMES_TO_QUANTUM_OPS.get(op.name)
        assert gate_name is not None, f"Unknown gate {op.name}"

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


@dataclass(frozen=True)
class UnrollEncodeLoop(RewritePattern):
    """Op conversion pattern for removing scf.ForOp with encode operations."""

    def _is_exact_encode_zero_loop(self, for_op: scf.ForOp) -> tuple[bool, str | None]:
        """Return true and encoder subroutine symbol iff `for_op` is exactly the qecp encode
        loop shape."""
        # 1. Validate structural constraints of the loop
        if len(for_op.iter_args) != 1 or len(for_op.results) != 1:
            return False, None

        block = for_op.body.block
        if len(block.args) != 2 or len(block.ops) != 4:
            return False, None

        # 2. Unpack and validate the sequence of operation types
        expected_types = (
            qecp.ExtractCodeblockOp,
            func.CallOp,
            qecp.InsertCodeblockOp,
            scf.YieldOp,
        )

        if not all(isinstance(op, t) for op, t in zip(block.ops, expected_types)):
            return False, None

        # 3. Verify the callee target name
        _, call_op, _, _ = block.ops
        if "encode_zero_" not in call_op.callee.string_value():
            return False, None

        return True, call_op.callee.string_value()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for unrolling encode loops."""
        is_encode_zero_loop, encode_subroutine_symbol = self._is_exact_encode_zero_loop(op)
        if isinstance(op, scf.ForOp) and is_encode_zero_loop:
            init_hreg = op.iter_args[0]
            if isinstance(init_hreg.owner, qecp.AllocOp):
                op.results[0].replace_all_uses_with(init_hreg)

                n_regs = init_hreg.type.width.value.data
                num_qubits = init_hreg.type.n.value.data

                for _ in range(n_regs):
                    reg = quantum.AllocOp(num_qubits)
                    rewriter.insert_op(reg)

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

    def _apply_experimental_hyperregister_lowering(self, op: builtin.ModuleOp):
        """Apply a separate pattern rewrite for lowering hyperregister-related qecp ops to quantum
        ops.
        NOTE: This is an experimental rewriting for the hyperregister related operations and types.
        1. Each codeblock allocated by qecp.alloc is replaced with a quantum.reg allocation.
        2. The encoding loop operation is unrolled.
        3. `qecp.extract_codeblock` operations are removed from the IR by replacing the uses with
        the corresponding quantum.reg SSA value.
        4. `qecp.insert_codeblock` operations are replaced with `quantum.dealloc` operation.
        NOTE: The current implementation only targets the 3-logical qubit GHZ circuit. The
        implementation is based on the IR structure of the specific circuit.
        TODO: We might come back to update the logic below to support 1-logical qubit circuits,
        where there is no ForOp encoding loop in the IR.
        """
        # Step 1: Unroll encoding loops and ensure the quantum.node op body contains no
        # nested regions.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [UnrollEncodeLoop()],
            )
        ).rewrite_module(op)

        # Step 2: Walk the quantum.node func to update the hyperregister related operations.
        module_op = op
        for op_ in module_op.walk():
            if isinstance(op_, func.FuncOp) and "quantum.node" in op_.attributes:
                qecp_alloc_op = None
                regs = []
                qecp_ops_to_remove = []
                dealloced_regs = {}

                for quantum_op in op_.walk():
                    rewriter = PatternRewriter(quantum_op)
                    match quantum_op:
                        case qecp.AllocOp():
                            qecp_alloc_op = quantum_op
                            qecp_ops_to_remove.append(quantum_op)
                        case func.CallOp() if "encode_zero_" in quantum_op.callee.string_value():
                            regs.append(quantum_op.results[0])
                        case qecp.ExtractCodeblockOp():
                            qecp_ops_to_remove.append(quantum_op)
                            if quantum_op.idx is not None:
                                idx = resolve_constant_params(quantum_op.idx)
                            elif quantum_op.idx_attr is not None:
                                idx = quantum_op.idx_attr.value.data
                            else:
                                raise CompileError("Expected an index on qecp.extract_codeblock op")
                            quantum_op.codeblock.replace_all_uses_with(regs[idx])
                        case qecp.InsertCodeblockOp():
                            qecp_ops_to_remove.append(quantum_op)
                            if quantum_op.idx is not None:
                                idx = resolve_constant_params(quantum_op.idx)
                            elif quantum_op.idx_attr is not None:
                                idx = quantum_op.idx_attr.value.data
                            if regs[idx] not in dealloced_regs:
                                dealloc_op = quantum.DeallocOp(regs[idx])
                                dealloced_regs[regs[idx]] = dealloc_op
                            quantum_op.results[0].replace_all_uses_with(qecp_alloc_op.results[0])
                        case qecp.DeallocOp():
                            rewriter.erase_op(quantum_op)
                        case quantum.DeviceReleaseOp():
                            # Dealloc qregs before device release
                            for _, dealloced_reg in dealloced_regs.items():
                                rewriter.insert_op(dealloced_reg)

                for quantum_op in reversed(qecp_ops_to_remove):
                    rewriter = PatternRewriter(quantum_op)
                    rewriter.erase_op(quantum_op)

                # Remove dead code
                region_dce(op_.body)

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecp-to-quantum pass."""
        # pylint: disable = unexpected-keyword-arg

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PhysicalCodeblockTypeConversion(recursive=True),
                    QecPhysicalQubitTypeConversion(recursive=True),
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

        self._apply_experimental_hyperregister_lowering(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
