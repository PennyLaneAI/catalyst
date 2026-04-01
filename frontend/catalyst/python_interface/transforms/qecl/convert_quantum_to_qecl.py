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
from typing import NoReturn, cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Operation, SSAValue
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

from catalyst.python_interface.dialects import qecl, quantum
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

# MARK: Alloc Op Pattern


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

        assert isinstance(nqubits_attr, IntegerAttr), (
            f"Expected attribute 'nqubits_attr' of {op.name} op to have type '{IntegerAttr.name}', "
            f"but got {nqubits_attr.name}"
        )
        nqubits = nqubits_attr.value.data

        hyper_reg_width = math.ceil(nqubits / self.k)
        rewriter.replace_op(
            op,
            [
                alloc_op := qecl.AllocOp(
                    qecl.LogicalHyperRegisterType(width=hyper_reg_width, k=self.k)
                ),
                _cast_to_qureg(alloc_op.hyper_reg),
            ],
        )


# MARK: Extract Op Pattern


@dataclass(frozen=True)
class ExtractOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.extract_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.ExtractOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.extract` ops."""

        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        # If we extract immediately after an alloc operation, we must also insert ops to perform
        # encoding and a QEC cycle. Cases where we don't immediately extract are typically around
        # control-flow operations.
        # Recall that the `alloc` conversion pattern inserts a builtin.unrealized_conversion_cast
        # op immediately after the alloc that converts from qecl.hyperreg -> quantum.reg.
        qreg_owner_op = op.qreg.owner
        if isinstance(qreg_owner_op, builtin.UnrealizedConversionCastOp) and isinstance(
            qreg_owner_op.operand_types[0], qecl.LogicalHyperRegisterType
        ):
            conv_cast_qreg_owner_op = qreg_owner_op.operands[0].owner

            # Convert type quantum.reg -> qecl.hyperreg
            # (to be resolved by ReconcileUnrealizedCastsPass)
            conv_cast_op = builtin.UnrealizedConversionCastOp.get(
                (qreg_owner_op.results[0],), (qreg_owner_op.operands[0].type,)
            )
            extract_codeblock_op = qecl.ExtractCodeblockOp(
                hyper_reg=conv_cast_op.results[0], idx=idx
            )

            if isinstance(conv_cast_qreg_owner_op, qecl.AllocOp):
                ops_to_insert = [
                    conv_cast_op,
                    extract_codeblock_op,
                    encode_op := qecl.EncodeOp(
                        in_codeblock=extract_codeblock_op.codeblock,
                        init_state=qecl.LogicalCodeblockInitState.Zero,
                    ),
                    qec_cycle_op := qecl.QecCycleOp(in_codeblock=encode_op.out_codeblock),
                    _cast_to_qubit(qec_cycle_op.out_codeblock),
                ]

            else:
                ops_to_insert = [
                    conv_cast_op,
                    extract_codeblock_op,
                    _cast_to_qubit(extract_codeblock_op.codeblock),
                ]

        else:
            # In this case, the extract op uses a register that originated from an op other than
            # `alloc`. In this case, we need to trace up the SSA graph until we find a qecl.alloc +
            # builtin.unrealized_conversion_cast op pair to determine the correct hyper-register
            # type.
            #
            # TODO!!!
            # owner_op = qreg_owner_op
            # while True:
            #     if self._is_op_conv_cast_after_alloc(owner_op):
            #         break

            #     if isinstance(owner_op, Operation) and len(owner_op.operands) == 1:
            #         owner_op = owner_op.operands[0].owner

            # ops_to_insert = []

            # Maybe we don't need this? And we just error out?
            _raise_failed_to_convert_op_compile_error(op)

        rewriter.replace_op(op, ops_to_insert)

    @classmethod
    def _is_op_conv_cast_after_alloc(cls, candidate_op: Operation | Block):
        return isinstance(candidate_op, builtin.UnrealizedConversionCastOp) and isinstance(
            candidate_op.operand_types[0], qecl.LogicalHyperRegisterType
        )


# MARK: Insert Op Pattern


class InsertOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.insert_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.InsertOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.insert` ops."""

        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        in_qreg_owner_op = op.in_qreg.owner
        qubit_owner_op = op.qubit.owner
        if (
            isinstance(qubit_owner_op, builtin.UnrealizedConversionCastOp)
            and isinstance(qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType)
            and isinstance(in_qreg_owner_op, builtin.UnrealizedConversionCastOp)
            and isinstance(in_qreg_owner_op.operand_types[0], qecl.LogicalHyperRegisterType)
        ):
            # Convert types quantum.reg -> qecl.hyperreg and quantum.qubit -> qecl.codeblock
            # (to be resolved by ReconcileUnrealizedCastsPass)
            ops_to_insert = [
                qreg_conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                    (in_qreg_owner_op.results[0],), (in_qreg_owner_op.operands[0].type,)
                ),
                qubit_conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                    (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
                ),
                insert_codeblock_op := qecl.InsertCodeblockOp(
                    in_hyper_reg=qreg_conv_cast_op.results[0],
                    idx=idx,
                    codeblock=qubit_conv_cast_op.results[0],
                ),
                _cast_to_qureg(insert_codeblock_op.out_hyper_reg),
            ]
        else:
            _raise_failed_to_convert_op_compile_error(op)

        rewriter.replace_op(op, ops_to_insert)


# MARK: Dealloc Op Pattern


class DeallocOpConversion(RewritePattern):
    """Converts `quantum.dealloc` ops to equivalent `qecl.dealloc` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.DeallocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.dealloc` ops."""

        qreg_owner_op = op.qreg.owner
        if isinstance(qreg_owner_op, builtin.UnrealizedConversionCastOp) and isinstance(
            qreg_owner_op.operand_types[0], qecl.LogicalHyperRegisterType
        ):
            ops_to_insert = [
                conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                    (qreg_owner_op.results[0],), (qreg_owner_op.operands[0].type,)
                ),
                qecl.DeallocOp(hyper_reg=conv_cast_op.results[0]),
            ]
        else:
            _raise_failed_to_convert_op_compile_error(op)

        rewriter.replace_op(op, ops_to_insert)


# MARK: Custom Op Pattern


class CustomOpConversion(RewritePattern):
    """Converts `quantum.custom` ops to equivalent `qecl.hadamard`, `qecl.s` and `qecl.cnot` ops.

    For now, we insert cycles of QEC after every gate operation.

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

        new_results = None

        match gate_name:
            case "Hadamard":
                assert len(op.in_qubits) == 1
                qubit_owner_op = op.in_qubits[0].owner
                if isinstance(qubit_owner_op, builtin.UnrealizedConversionCastOp) and isinstance(
                    qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType
                ):
                    ops_to_insert = [
                        conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                            (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
                        ),
                        gate_op := qecl.HadamardOp(in_codeblock=conv_cast_op.results[0], idx=0),
                        qec_cycle_op := qecl.QecCycleOp(in_codeblock=gate_op.out_codeblock),
                        _cast_to_qubit(qec_cycle_op.out_codeblock),
                    ]
                else:
                    _raise_failed_to_convert_op_compile_error(op)

            case "S":
                adjoint = True if op.properties.get("adjoint") else False
                assert len(op.in_qubits) == 1
                qubit_owner_op = op.in_qubits[0].owner
                if isinstance(qubit_owner_op, builtin.UnrealizedConversionCastOp) and isinstance(
                    qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType
                ):
                    ops_to_insert = [
                        conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                            (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
                        ),
                        gate_op := qecl.SOp(
                            in_codeblock=conv_cast_op.results[0], idx=0, adjoint=adjoint
                        ),
                        qec_cycle_op := qecl.QecCycleOp(in_codeblock=gate_op.out_codeblock),
                        _cast_to_qubit(qec_cycle_op.out_codeblock),
                    ]
                else:
                    _raise_failed_to_convert_op_compile_error(op)

            case "CNOT":
                assert len(op.in_qubits) == 2
                ctrl_qubit_owner_op = op.in_qubits[0].owner
                trgt_qubit_owner_op = op.in_qubits[1].owner
                if (
                    isinstance(ctrl_qubit_owner_op, builtin.UnrealizedConversionCastOp)
                    and isinstance(ctrl_qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType)
                    and isinstance(trgt_qubit_owner_op, builtin.UnrealizedConversionCastOp)
                    and isinstance(trgt_qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType)
                ):
                    ops_to_insert = [
                        ctrl_conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                            (ctrl_qubit_owner_op.results[0],),
                            (ctrl_qubit_owner_op.operands[0].type,),
                        ),
                        trgt_conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                            (trgt_qubit_owner_op.results[0],),
                            (trgt_qubit_owner_op.operands[0].type,),
                        ),
                        gate_op := qecl.CnotOp(
                            in_ctrl_codeblock=ctrl_conv_cast_op.results[0],
                            idx_ctrl=0,
                            in_trgt_codeblock=trgt_conv_cast_op.results[0],
                            idx_trgt=0,
                        ),
                        ctrl_qecl_cycle_op := qecl.QecCycleOp(
                            in_codeblock=gate_op.out_ctrl_codeblock
                        ),
                        trgt_qecl_cycle_op := qecl.QecCycleOp(
                            in_codeblock=gate_op.out_trgt_codeblock
                        ),
                        ctrl_conv_cast_op := _cast_to_qubit(ctrl_qecl_cycle_op.out_codeblock),
                        trgt_conv_cast_op := _cast_to_qubit(trgt_qecl_cycle_op.out_codeblock),
                    ]
                    new_results = (ctrl_conv_cast_op.results[0], trgt_conv_cast_op.results[0])
                else:
                    _raise_failed_to_convert_op_compile_error(op)

            case _:
                raise NotImplementedError(
                    f"Conversion of op '{op.name}' only supports gates 'Hadamard', 'S' and 'CNOT', "
                    f"but got {gate_name}"
                )

        rewriter.replace_op(op, ops_to_insert, new_results=new_results)


# MARK: Measure Op Pattern


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
        new_results = None

        qubit_owner_op = op.in_qubit.owner
        if isinstance(qubit_owner_op, builtin.UnrealizedConversionCastOp) and isinstance(
            qubit_owner_op.operand_types[0], qecl.LogicalCodeblockType
        ):
            ops_to_insert = [
                conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                    (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
                ),
                measure_op := qecl.MeasureOp(in_codeblock=conv_cast_op.results[0], idx=0),
                conv_cast_op := _cast_to_qubit(measure_op.out_codeblock),
            ]
            new_results = (measure_op.mres, conv_cast_op.results[0])
        else:
            _raise_failed_to_convert_op_compile_error(op)

        rewriter.replace_op(op, ops_to_insert, new_results=new_results)


# MARK: Scf.For Op Pattern


class ScfForOpConversion(RewritePattern):
    """TODO"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter):
        if any(
            [
                isinstance(result_type, qecl.LogicalHyperRegisterType)
                for result_type in op.result_types
            ]
        ):
            # This op has already been converted
            return

        # Steps:
        #   1. Insert conv_cast op before the for op (reg -> hyperreg)
        #   2. Replace yield op with conv-cast (reg -> hyperreg) + updated yield op
        #   3. Update return type of for op (if necessary)
        #   4. Insert conv_cast op after the for op (hyperreg -> reg)
        iter_args = op.iter_args

        for i, iter_arg in enumerate(iter_args):
            if (
                isinstance(iter_arg.type, quantum.QuregType)
                and isinstance(iter_arg.owner, builtin.UnrealizedConversionCastOp)
                and isinstance(iter_arg.owner.operand_types[0], qecl.LogicalHyperRegisterType)
            ):
                conv_cast_op = builtin.UnrealizedConversionCastOp.get(
                    (iter_arg,), (iter_arg.owner.operand_types[0],)
                )
                rewriter.insert_op(conv_cast_op, InsertPoint.before(rewriter.current_operation))
                rewriter.replace_uses_with_if(
                    iter_arg.owner.results[0],
                    conv_cast_op.results[0],
                    lambda use: use.operation is not conv_cast_op,
                )

                # The first block arg is the indexing arg
                i_block_arg = i + 1

                rewriter.replace_value_with_new_type(
                    op.body.block.args[i_block_arg], iter_arg.owner.operand_types[0]
                )

                conv_cast_op = _cast_to_qureg(op.body.block.args[i_block_arg])
                assert op.body.ops.first is not None
                rewriter.insert_op(conv_cast_op, InsertPoint.before(op.body.ops.first))
                rewriter.replace_uses_with_if(
                    op.body.block.args[i_block_arg],
                    conv_cast_op.results[0],
                    lambda use: use.operation is not conv_cast_op,
                )

        yield_op = op.body.ops.last
        assert isinstance(yield_op, scf.YieldOp)

        for i, yield_arg in enumerate(yield_op.arguments):
            conv_cast_op = builtin.UnrealizedConversionCastOp.get(
                (yield_arg,), (op.iter_args[i].type,)
            )
            rewriter.insert_op(conv_cast_op, InsertPoint.before(yield_op))
            rewriter.replace_uses_with_if(
                yield_arg, conv_cast_op.results[0], lambda use: use.operation is not conv_cast_op
            )
            rewriter.replace_value_with_new_type(op.results[i], conv_cast_op.results[0].type)

        for result in op.results:
            if isinstance(result.type, qecl.LogicalHyperRegisterType):
                conv_cast_op = _cast_to_qureg(result)
                rewriter.insert_op(conv_cast_op, InsertPoint.after(rewriter.current_operation))
                rewriter.replace_uses_with_if(
                    result, conv_cast_op.results[0], lambda use: use.operation is not conv_cast_op
                )


# MARK: Helpers


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


def _cast_to_qureg(value: SSAValue) -> builtin.UnrealizedConversionCastOp:
    """Return a `builtin.unrealized_conversion_cast` op that casts `value` to type `quantum.reg`."""
    # For now, we restrict usage of this function to cast from `qecl.hyperreg` to `quantum.reg`
    assert isinstance(value.type, qecl.LogicalHyperRegisterType), (
        f"Value '{value}' must have type '{qecl.LogicalHyperRegisterType.name}' to cast to "
        f"'{quantum.QuregType.name}', but got {value.type}"
    )
    return builtin.UnrealizedConversionCastOp.get((value,), (quantum.QuregType(),))


def _cast_to_qubit(value: SSAValue) -> builtin.UnrealizedConversionCastOp:
    """Return a `builtin.unrealized_conversion_cast` op that casts `value` to type `quantum.bit`."""
    # For now, we restrict usage of this function to cast from `qecl.codeblock` to `quantum.bit`
    assert isinstance(value.type, qecl.LogicalCodeblockType), (
        f"Value '{value}' must have type '{qecl.LogicalCodeblockType.name}' to cast to "
        f"'{quantum.QubitType.name}', but got {value.type}"
    )
    return builtin.UnrealizedConversionCastOp.get((value,), (quantum.QubitType(),))


def _raise_failed_to_convert_op_compile_error(op: Operation) -> NoReturn:
    """Raise a `CompileError` for cases where the conversion pattern for `op` failed."""
    raise CompileError(
        f"Failed to convert op '{op}': conversion pattern for '{op.name}' could not identity "
        f"appropriate type-conversion operation(s)"
    )


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
                    ScfForOpConversion(),
                ]
            )
        ).rewrite_module(op)

        # Certain patterns leave behind `builtin.unrealized_conversion_cast` ops;
        # this pass removes them
        ReconcileUnrealizedCastsPass().apply(ctx, op)


convert_quantum_to_qecl_pass = compiler_transform(ConvertQuantumToQecLogicalPass)
