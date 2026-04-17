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

This module contains the implementation of the xDSL convert-quantum-to-qecl dialect-conversion pass.
To apply this pass, the logical codeblock size, k, must be known.

Example
-------

Before:

```mlir
func.func public @circuit() -> tensor<1x1xi64> attributes {quantum.node} {
  %c1_i64 = arith.constant 1 : i64
  quantum.device shots(%c1_i64) ["", "", ""]
  %0 = quantum.alloc( 1) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.mcmobs %mres : !quantum.obs
  %3 = quantum.sample %2 : tensor<1x1xf64>
  %4 = stablehlo.convert %3 : (tensor<1x1xf64>) -> tensor<1x1xi64>
  %5 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %5 : !quantum.reg
  quantum.device_release
  return %4 : tensor<1x1xi64>
}
```

After applying convert-quantum-to-qecl (with k = 1):

```mlir
func.func public @circuit() -> tensor<1x1xi64> attributes {quantum.node} {
  %c1_i64 = arith.constant 1 : i64
  quantum.device shots(%c1_i64) ["", "", ""]
  %0 = qecl.alloc() : !qecl.hyperreg<1 x 1>
  %1 = qecl.extract_block %0[0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
  %2 = qecl.encode[zero] %1 : !qecl.codeblock<1>
  %3 = qecl.insert_block %0[0], %2 : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
  %4 = qecl.extract_block %3[0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
  %out_qubits = qecl.qec %4 : !qecl.codeblock<1>
  %out_qubits_1 = qecl.hadamard %out_qubits[0] : !qecl.codeblock<1>
  %out_qubits_2 = qecl.qec %out_qubits_1 : !qecl.codeblock<1>
  %mres, %5 = qecl.measure %out_qubits_2[0] : i1, !qecl.codeblock<1>
  %6 = quantum.mcmobs %mres : !quantum.obs
  %7 = quantum.sample %6 : tensor<1x1xf64>
  %8 = stablehlo.convert %7 : (tensor<1x1xf64>) -> tensor<1x1xi64>
  %9 = qecl.insert_block %3[0], %5 : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
  qecl.dealloc %9 : !qecl.hyperreg<1 x 1>
  quantum.device_release
  func.return %8 : tensor<1x1xi64>
}
```

Known Limitations
-----------------

The convert-quantum-to-qecl pass does not support the following cases:

  * QEC codes where the number of logical qubits per codeblock, k, is greater than 1.
  * `quantum.alloc` ops with a dynamic number of qubits.
  * Programs with non-Clifford gates; specifically any gates other than I, X, Y, Z, Hadamard, S or
    CNOT.
  * Programs with control-flow operations (scf.for, scf.if, etc.).
"""

import math
from dataclasses import dataclass
from typing import NoReturn, TypeGuard, cast

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, BlockArgument, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from catalyst.python_interface.dialects import qecl, quantum
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

# MARK: Alloc Op Pattern


@dataclass(frozen=True)
class AllocOpConversion(RewritePattern):
    """Converts `quantum.alloc` ops to equivalent `qecl.alloc` ops.

    While the `quantum.alloc` implicitly initializes all abstract qubits within the allocated
    register to the |0> state, the logical codeblocks within a hyper-register must be explicitly
    initialized to the logical |0> state by means of an *encoding* protocol. The appropriate
    encoding ops are inserted in this conversion pattern.
    """

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

        assert isinstance(nqubits_attr, expected_type := IntegerAttr), (
            f"Expected attribute 'nqubits_attr' of {op.name} op to have type "
            f"'{expected_type.name}', but got {nqubits_attr.name}"
        )

        nqubits = nqubits_attr.value.data
        hyper_reg_width = math.ceil(nqubits / self.k)

        alloc_op = qecl.AllocOp(qecl.LogicalHyperRegisterType(width=hyper_reg_width, k=self.k))
        encoding_ops = self._get_hyper_reg_encoding_ops(alloc_op.hyper_reg)
        last_encoding_op = encoding_ops[-1]

        assert (
            results_len := len(last_encoding_op.result_types)
        ) == 1, f"Expected last encoding op to have exactly 1 result, but got {results_len}"
        assert isinstance(
            received_type := last_encoding_op.result_types[0],
            expected_type := qecl.LogicalHyperRegisterType,
        ), (
            f"Expected last encoding op to return type '{expected_type.name}', but got "
            f"{received_type.name}"
        )

        ops_to_insert = (
            alloc_op,
            *encoding_ops,
            _cast_to_qureg(last_encoding_op.results[0]),
        )

        rewriter.replace_op(op, ops_to_insert)

    @classmethod
    def _get_hyper_reg_encoding_ops(
        cls, hyper_reg: SSAValue[qecl.LogicalHyperRegisterType]
    ) -> tuple[Operation, ...]:
        # pylint: disable=line-too-long
        """Helper function to get the operations that encode each codeblock in the hyper-register."""
        hyper_reg_width = hyper_reg.type.width.value.data

        assert (
            hyper_reg_width >= 1
        ), f"Expected hyper-register width >= 1, but got width {hyper_reg_width}"

        ops_to_insert: tuple[Operation, ...] = ()

        if hyper_reg_width == 1:
            # No need to loop, insert encode op directly.
            ops_to_insert = (
                extract_op := qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=0),
                encode_op := qecl.EncodeOp(extract_op.codeblock, init_state="zero"),
                qecl.InsertCodeblockOp(
                    in_hyper_reg=hyper_reg, idx=0, codeblock=encode_op.out_codeblock
                ),
            )

        else:
            # Loop over all codeblocks in the hyper-register and encode them to logical zero state.
            # Ops for lower bound, upper bound, and step size.
            lb_op = arith.ConstantOp.from_int_and_width(0, IndexType())
            ub_op = arith.ConstantOp.from_int_and_width(hyper_reg_width, IndexType())
            step_op = arith.ConstantOp.from_int_and_width(1, IndexType())

            for_body = Block(
                [],
                arg_types=(builtin.IndexType(), hyper_reg.type),
            )

            for_each_codeblock_op = scf.ForOp(
                lb=lb_op,
                ub=ub_op,
                step=step_op,
                iter_args=(hyper_reg,),
                body=for_body,
            )

            # Build the body of the for loop. On each iteration, extract the codeblock at the
            # iteration index, encode it, and re-insert into hyper-register. Finally, yield the
            # updated hyper-register.
            with ImplicitBuilder(for_each_codeblock_op.body):
                indvar = cast(BlockArgument[IndexType], for_each_codeblock_op.body.block.args[0])
                hyper_reg = cast(
                    BlockArgument[qecl.LogicalHyperRegisterType],
                    for_each_codeblock_op.body.block.args[1],
                )

                extract_op = qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=indvar)
                encode_op = qecl.EncodeOp(extract_op.codeblock, init_state="zero")
                insert_op = qecl.InsertCodeblockOp(
                    in_hyper_reg=hyper_reg, idx=indvar, codeblock=encode_op.out_codeblock
                )
                scf.YieldOp(insert_op.out_hyper_reg)

            ops_to_insert = (
                lb_op,
                ub_op,
                step_op,
                for_each_codeblock_op,
            )

        assert ops_to_insert, "Sequence of ops to insert is empty"
        return ops_to_insert


# MARK: Extract Op Pattern


@dataclass(frozen=True)
class ExtractOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.extract_block` ops.

    Every time we extract a codeblock from a hyper-register, we perform a cycle of error correction
    on that codeblock.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.ExtractOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.extract` ops."""
        # Recall that the `alloc` conversion pattern inserts a builtin.unrealized_conversion_cast
        # op immediately after the alloc that converts from qecl.hyperreg -> quantum.reg.
        qreg_owner_op = op.qreg.owner
        if not _is_type_convertible(qreg_owner_op, qecl.LogicalHyperRegisterType):
            # TODO: We will need to also support the case where a quantum.extract op does not
            # immediately follow a `quantum.alloc` op, e.g. when a block takes in a register as an
            # argument, or if there is some other op that acts on the register in-between alloc and
            # extract:
            #   %0 = quantum.alloc(1) : !quantum.reg
            #   %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            #   %2 = "some.op"(%0) : (!quantum.reg) -> !quantum.reg
            #   %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit
            _raise_failed_to_convert_op_compile_error(op)

        # NOTE: For now we assume k=1, so the quantum.extract index maps 1:1 with the
        # qecl.extract_block index.
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        ops_to_insert = (
            # Convert type quantum.reg -> qecl.hyperreg
            # (to be resolved by ReconcileUnrealizedCastsPass)
            conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                (qreg_owner_op.results[0],), (qreg_owner_op.operands[0].type,)
            ),
            extract_codeblock_op := qecl.ExtractCodeblockOp(
                hyper_reg=conv_cast_op.results[0], idx=idx
            ),
            qec_cycle_op := qecl.QecCycleOp(in_codeblock=extract_codeblock_op.codeblock),
            _cast_to_qubit(qec_cycle_op.out_codeblock),
        )

        rewriter.replace_op(op, ops_to_insert)


# MARK: Insert Op Pattern


class InsertOpConversion(RewritePattern):
    """Converts `quantum.extract` ops to equivalent `qecl.insert_block` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.InsertOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.insert` ops."""
        in_qreg_owner_op = op.in_qreg.owner
        qubit_owner_op = op.qubit.owner
        if not (
            _is_type_convertible(qubit_owner_op, qecl.LogicalCodeblockType)
            and _is_type_convertible(in_qreg_owner_op, qecl.LogicalHyperRegisterType)
        ):
            _raise_failed_to_convert_op_compile_error(op)

        # NOTE: As with extract ops, for now we assume k=1, so the quantum.insert index maps 1:1
        # with the qecl.insert_block index.
        idx = _get_idx_value_or_attr_from_extract_or_insert_op(op, rewriter)

        ops_to_insert = (
            # Convert types quantum.reg -> qecl.hyperreg and quantum.qubit -> qecl.codeblock
            # (to be resolved by ReconcileUnrealizedCastsPass)
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
        )

        rewriter.replace_op(op, ops_to_insert)


# MARK: Dealloc Op Pattern


class DeallocOpConversion(RewritePattern):
    """Converts `quantum.dealloc` ops to equivalent `qecl.dealloc` ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.DeallocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.dealloc` ops."""

        qreg_owner_op = op.qreg.owner
        if not _is_type_convertible(qreg_owner_op, qecl.LogicalHyperRegisterType):
            _raise_failed_to_convert_op_compile_error(op)

        ops_to_insert = (
            conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                (qreg_owner_op.results[0],), (qreg_owner_op.operands[0].type,)
            ),
            qecl.DeallocOp(hyper_reg=conv_cast_op.results[0]),
        )

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
            case "Identity" | "PauliX" | "PauliY" | "PauliZ" | "Hadamard" | "S":
                ops_to_insert = self._get_qecl_ops_for_single_qubit_gate(op)

            case "CNOT":
                assert len(op.in_qubits) == 2
                ctrl_qubit_owner_op = op.in_qubits[0].owner
                trgt_qubit_owner_op = op.in_qubits[1].owner
                if not (
                    _is_type_convertible(ctrl_qubit_owner_op, qecl.LogicalCodeblockType)
                    and _is_type_convertible(trgt_qubit_owner_op, qecl.LogicalCodeblockType)
                ):
                    _raise_failed_to_convert_op_compile_error(op)
                else:
                    ops_to_insert = (
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
                    )
                    new_results = (ctrl_conv_cast_op.results[0], trgt_conv_cast_op.results[0])

            case _:
                raise CompileError(
                    f"Conversion of op '{op.name}' only supports gates 'Identity', 'PauliX', "
                    f"'PauliY', 'PauliZ', 'Hadamard', 'S' and 'CNOT', but got '{gate_name}'"
                )

        rewriter.replace_op(op, ops_to_insert, new_results=new_results)

    @classmethod
    def _get_qecl_ops_for_single_qubit_gate(cls, op: quantum.CustomOp) -> tuple[Operation, ...]:
        """Helper function that returns the sequence of qecl operations to insert given the matched
        quantum.custom op.
        """
        assert len(op.in_qubits) == 1

        qubit_owner_op = op.in_qubits[0].owner
        if not _is_type_convertible(qubit_owner_op, qecl.LogicalCodeblockType):
            _raise_failed_to_convert_op_compile_error(op)

        conv_cast_op = builtin.UnrealizedConversionCastOp.get(
            (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
        )

        gate_name = op.gate_name.data

        match gate_name:
            case "Identity":
                qecl_gate_op = qecl.IdentityOp(in_codeblock=conv_cast_op.results[0], idx=0)
            case "PauliX":
                qecl_gate_op = qecl.PauliXOp(in_codeblock=conv_cast_op.results[0], idx=0)
            case "PauliY":
                qecl_gate_op = qecl.PauliYOp(in_codeblock=conv_cast_op.results[0], idx=0)
            case "PauliZ":
                qecl_gate_op = qecl.PauliZOp(in_codeblock=conv_cast_op.results[0], idx=0)
            case "Hadamard":
                qecl_gate_op = qecl.HadamardOp(in_codeblock=conv_cast_op.results[0], idx=0)
            case "S":
                adjoint = bool(op.properties.get("adjoint"))
                qecl_gate_op = qecl.SOp(
                    in_codeblock=conv_cast_op.results[0], idx=0, adjoint=adjoint
                )
            case _:
                assert False, (
                    f"Expected single-qubit gate from set {{'Identity', 'PauliX', 'PauliY', "
                    f"'PauliZ', 'Hadamard', 'S'}}, but got '{gate_name}'"
                )

        return (
            conv_cast_op,
            qecl_gate_op,
            qec_cycle_op := qecl.QecCycleOp(in_codeblock=qecl_gate_op.out_codeblock),
            _cast_to_qubit(qec_cycle_op.out_codeblock),
        )


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
        if not _is_type_convertible(qubit_owner_op, qecl.LogicalCodeblockType):
            _raise_failed_to_convert_op_compile_error(op)
        else:
            ops_to_insert = (
                conv_cast_op := builtin.UnrealizedConversionCastOp.get(
                    (qubit_owner_op.results[0],), (qubit_owner_op.operands[0].type,)
                ),
                measure_op := qecl.MeasureOp(in_codeblock=conv_cast_op.results[0], idx=0),
                conv_cast_op := _cast_to_qubit(measure_op.out_codeblock),
            )
            new_results = (measure_op.mres, conv_cast_op.results[0])

        rewriter.replace_op(op, ops_to_insert, new_results=new_results)


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


def _is_type_convertible(
    op_or_block: Operation | Block, expected_type: type
) -> TypeGuard[builtin.UnrealizedConversionCastOp]:
    """Determine if the given operation returns a type that can be converted to the expected type.

    Specifically, this function checks if the given operation is a
    `builtin.unrealized_conversion_cast` op whose operand is of the expected type. If it is, then
    another `builtin.unrealized_conversion_cast` can be inserted after it to "undo" the unrealized
    conversion cast. These pairs of `unrealized_conversion_cast` ops are removed by applying the
    `ReconcileUnrealizedCastsPass` at the end of the pass.

    The input argument can be of type `Operation | Block` since this function is typically used by
    passing in the owner of an SSA value; if the SSA value is a block argument, then it's owner is a
    Block.
    """
    return isinstance(op_or_block, builtin.UnrealizedConversionCastOp) and isinstance(
        op_or_block.operand_types[0], expected_type
    )


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
        """Apply the convert-quantum-to-qecl pass."""
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

        # Certain patterns leave behind `builtin.unrealized_conversion_cast` ops;
        # this pass removes them
        ReconcileUnrealizedCastsPass().apply(ctx, op)


convert_quantum_to_qecl_pass = compiler_transform(ConvertQuantumToQecLogicalPass)
