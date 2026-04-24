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

Known Limitations
-----------------

The convert-qecl-to-qecp pass does not support the following cases:

  * QEC codes where the number of logical qubits per codeblock, k, is greater than 1.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.builtin import I1, IndexType, SymbolRefAttr, TensorType, i1
from xdsl.ir import Block, BlockArgument, OpResult, Region
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

from catalyst.python_interface.dialects import qecl, qecp
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

from .convert_qecl_noise_to_qec_noise import ConvertQECLNoiseOpToQECPNoisePass
from .qec_code_lib import QecCode


class CheckType(StrEnum):
    """Check types for QEC codes. Currently limited to CSS codes (X and Z checks)."""

    X = "X"
    Z = "Z"


# MARK: Type Conversion Pattern


@dataclass
class CodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecl.codeblock -> qecp.codeblock."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalCodeblockType) -> qecp.PhysicalCodeblockType:
        """Type conversion rewrite pattern for logical codeblock types."""
        if typ.k.value.data != self.qec_code.k:
            raise CompileError(
                f"Failed to convert type {typ} with QEC code '{self.qec_code}'; codeblock has "
                f"k = {typ.k.value.data} but QEC code has k = {self.qec_code.k}"
            )

        return qecp.PhysicalCodeblockType(typ.k, self.qec_code.n)


@dataclass
class HyperRegisterTypeConversion(TypeConversionPattern):
    """Hyper-register type conversion pattern from qecl.hyperreg -> qecp.hyperreg."""

    qec_code: QecCode = field(kw_only=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecl.LogicalHyperRegisterType) -> qecp.PhysicalHyperRegisterType:
        """Type conversion rewrite pattern for physical codeblock types."""
        if typ.k.value.data != self.qec_code.k:
            raise CompileError(
                f"Failed to convert type {typ} with QEC code '{self.qec_code}'; hyper-register has "
                f"k = {typ.k.value.data} but QEC code has k = {self.qec_code.k}"
            )

        return qecp.PhysicalHyperRegisterType(typ.width, typ.k, self.qec_code.n)


# MARK: Measure Op Pattern


@dataclass(frozen=True)
class MeasureOpConversion(RewritePattern):
    """Converts `qecl.measure` ops to a call to a subroutine that performs a transversal measurement
    on the physical codeblock.

    In order to make the corresponding logical measurement outcome(s) of the `qecl.measure` op
    available for subsequent use, this pattern also inserted a `qecp.decode_physical_meas` op that
    acts on the results of the transversal measurements and returns the corresponding k logical
    measurements outcomes.

    While this rewrite pattern may appear as though it supports arbitrary values of k, it is
    important to note that as implemented, it does not consider the `idx` attribute/operand of the
    `qecl.measure` op, which indicates the position of the logical qubit within the codeblock to
    measure. Therefore, for codes with k > 1, it is not possible to perform a measurement that
    corresponds to a single logical qubit within the codeblock since the measurement is transversal
    and collapses the entire state of the codeblock.
    """

    qec_code: QecCode

    measure_subroutine: func.FuncOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.MeasureOp, rewriter: PatternRewriter):
        """Rewrite pattern for `qecl.measure` ops."""

        k = op.out_codeblock.type.k.value.data
        n = self.qec_code.n

        # The type-converter should already raise a CompileError if the values of k don't agree;
        # assert just in case.
        assert k == self.qec_code.k, (
            f"Value mismatch: codeblock {op.out_codeblock} has k = {k} but QEC code has "
            f"k = {self.qec_code.k}"
        )

        ops_to_insert = (
            subroutine_call_op := func.CallOp(
                callee=SymbolRefAttr(self.measure_subroutine.sym_name),
                arguments=(op.in_codeblock,),
                return_types=(builtin.TensorType(i1, shape=(n,)), op.in_codeblock.type),
            ),
            decode_op := qecp.DecodePhysicalMeasurementOp(
                physical_measurements=cast(
                    OpResult[builtin.TensorType], subroutine_call_op.results[0]
                ),
                logical_measurements_type=builtin.TensorType(i1, shape=(k,)),
            ),
            extract_idx_op := arith.ConstantOp.from_int_and_width(0, IndexType()),
            tensor_extract_op := tensor.ExtractOp(
                decode_op.logical_measurements, indices=extract_idx_op.result, result_type=i1
            ),
        )

        new_results = (tensor_extract_op.result, subroutine_call_op.results[1])

        rewriter.replace_op(op, ops_to_insert, new_results=new_results)


# MARK: Alloc/Dealloc Patterns


@dataclass
class AllocationConversion(RewritePattern):
    """Op conversion pattern from qecl.alloc -> qecp.alloc."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.AllocOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        rewriter.replace_op(op, qecp.AllocOp(op.result_types[0]))


@dataclass
class DeallocationConversion(RewritePattern):
    """Op conversion pattern from qecl.dealloc -> qecp.dealloc."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.DeallocOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        rewriter.replace_op(op, qecp.DeallocOp(op.hyper_reg))


# MARK: Extract/Insert Patterns


@dataclass
class ExtractBlockConversion(RewritePattern):
    """Op conversion pattern from qecl.extract_block -> qecp.extract_block."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.ExtractCodeblockOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        rewriter.replace_op(op, qecp.ExtractCodeblockOp(op.hyper_reg, op.idx_attr))


@dataclass
class InsertBlockConversion(RewritePattern):
    """Op conversion pattern from qecl.insert_block -> qecp.insert_block."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.InsertCodeblockOp, rewriter: PatternRewriter):
        """Op conversion rewrite pattern for lowering ops that allocate codeblocks."""
        rewriter.replace_op(op, qecp.InsertCodeblockOp(op.in_hyper_reg, op.idx_attr, op.codeblock))


# MARK: Encode Op Pattern


@dataclass
class EncodeOpConversion(RewritePattern):
    """Converts qecl.encode [zero] to the equivalent subroutine of qecp gates"""

    qec_code: QecCode
    encode_subroutine: func.FuncOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.EncodeOp, rewriter: PatternRewriter):
        """Rewrite pattern for `qecl.encode [zero]` op"""

        if not op.init_state.data == "zero":
            raise NotImplementedError(
                "Lowering qecl.EncodeOp to the qecp dialect is only implemented "
                "for init_state 'zero'"
            )

        if (k := op.in_codeblock.type.k.value.data) != self.qec_code.k:
            raise CompileError(
                f"Circuit expressed in the qecl dialect with k={k} is not compatible with "
                f"lowering to a code with k={self.qec_code.k}"
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

    # To specify the number of errors to be injected in the noise subroutine,
    # which is needed for the convert-qecl-noise-to-qecp-noise pass.
    number_errors: int = 1

    def __post_init__(self):
        # This method handles the case where `qec_code` is given as a dictionary rather than a
        # `QecCode` object. This is possible when the pass is registered in the IR and applied via
        # the `transform.apply_registered_pass` op, in which case the QecCode pass option is
        # represented as a dictionary attribute. Converting it from this dictionary attribute back
        # to a QecCode object allows for regular usage of this variable.
        if isinstance(self.qec_code, dict):
            # Because the class is frozen, we cannot assign to self.qec_code directly.
            # We use object.__setattr__ to bypass the frozen restriction.
            new_code = QecCode.from_dict(self.qec_code)
            object.__setattr__(self, "qec_code", new_code)

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-to-qecp pass."""
        if self.qec_code.k != 1:
            raise NotImplementedError(
                f"The {self.name} pass only supports QEC codes where the number of logical qubits "
                f"per codeblock, k, is 1, but got k = {self.qec_code.k}"
            )

        # n is the number of physical data qubits from the QEC code.
        ConvertQECLNoiseOpToQECPNoisePass(
            n=self.qec_code.n, number_errors=self.number_errors
        ).apply(ctx, op)

        # Insert subroutines into the module
        module_block = op.regions[0].blocks.first
        assert module_block is not None, "Module has no block"

        encode_subroutine = self.create_encode_subroutine()
        module_block.add_op(encode_subroutine)

        measure_subroutine = self.create_measure_subroutine()
        module_block.add_op(measure_subroutine)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CodeblockTypeConversion(qec_code=self.qec_code),
                    HyperRegisterTypeConversion(qec_code=self.qec_code),
                    EncodeOpConversion(qec_code=self.qec_code, encode_subroutine=encode_subroutine),
                    MeasureOpConversion(
                        qec_code=self.qec_code,
                        measure_subroutine=measure_subroutine,
                    ),
                    AllocationConversion(),
                    DeallocationConversion(),
                    InsertBlockConversion(),
                    ExtractBlockConversion(),
                ]
            )
        ).rewrite_module(op)

    def create_measure_subroutine(self) -> func.FuncOp:
        """Create the subroutine that performs the transversal measurement of a physical codeblock.

        Note that this method does not insert the subroutine into the module op. Instead it returns
        the built func.FuncOp object that can then be subsequently inserted where desired.
        """
        codeblock_type = qecp.PhysicalCodeblockType(self.qec_code.k, self.qec_code.n)
        block = Block(arg_types=(codeblock_type,))

        with ImplicitBuilder(block):
            # Loop over all physical qubits in the codeblock and measure them.
            in_codeblock = cast(BlockArgument[qecp.PhysicalCodeblockType], block.args[0])
            n = in_codeblock.type.n.value.data

            # Initialize an empty tensor to store the physical measurement results
            empty_tensor_op = tensor.EmptyOp([], TensorType(i1, shape=(n,)))

            # Ops for lower bound, upper bound, and step size.
            lb_op = arith.ConstantOp.from_int_and_width(0, IndexType())
            ub_op = arith.ConstantOp.from_int_and_width(n, IndexType())
            step_op = arith.ConstantOp.from_int_and_width(1, IndexType())

            for_body = Block(
                [],
                arg_types=(builtin.IndexType(), empty_tensor_op.tensor.type, in_codeblock.type),
            )

            for_each_qubit_op = scf.ForOp(
                lb=lb_op,
                ub=ub_op,
                step=step_op,
                iter_args=(empty_tensor_op.tensor, in_codeblock),
                body=for_body,
            )

            # Build the body of the for loop. On each iteration, extract the physical qubit at the
            # iteration index, measure it, and re-insert into the codeblock. Also insert the
            # measurement result into the tensor. Finally, yield updated tensor of measurement
            # results and the updated codeblock.
            with ImplicitBuilder(for_each_qubit_op.body):
                indvar = cast(BlockArgument[IndexType], for_each_qubit_op.body.block.args[0])
                mres_tensor = cast(
                    BlockArgument[TensorType[I1]], for_each_qubit_op.body.block.args[1]
                )
                codeblock = cast(
                    BlockArgument[qecp.PhysicalCodeblockType],
                    for_each_qubit_op.body.block.args[2],
                )

                extract_op = qecp.ExtractQubitOp(codeblock=codeblock, idx=indvar)
                measure_op = qecp.MeasureOp(extract_op.qubit)
                insert_op = qecp.InsertQubitOp(
                    in_codeblock=codeblock, idx=indvar, qubit=measure_op.out_qubit
                )

                tensor_insert_op = tensor.InsertOp(
                    measure_op.mres, dest=mres_tensor, indices=indvar
                )
                scf.YieldOp(tensor_insert_op.result, insert_op.out_codeblock)

            out_mres_tensor = for_each_qubit_op.results[0]
            out_codeblock = for_each_qubit_op.results[1]
            func.ReturnOp(out_mres_tensor, out_codeblock)

        measure_subroutine = func.FuncOp(
            name=f"measure_transversal_{self.qec_code.name}",
            function_type=(
                (codeblock_type,),
                (
                    builtin.TensorType(i1, shape=(n,)),
                    codeblock_type,
                ),
            ),
            region=Region(block),
            visibility="private",  # so that the `-symbol-dce` pass can remove if unused
        )

        return measure_subroutine

    def create_encode_subroutine(self) -> func.FuncOp:
        """Create a subroutine that takes in a codeblock, encodes it in the zero state for
        the QEC code (based on the tanner graph), and returns the encoded codeblock. This
        encoding procedure follows the example shown in arXiv: 0905.2794, Section VIII.A.
        It does not include Z-corrections; this is because the encode op is followed directly
        by a full cycle of error correction when lowering to the qecl dialect.

        The subroutine allocates auxiliary qubits for use in encoding based on the number of
        rows in the X tanner graph, and deallocates them once encoding is complete.

        Note that this method does not insert the subroutine into the module op. Instead it returns
        the built func.FuncOp object that can then be subsequently inserted where desired.
        """
        codeblock_type = qecp.PhysicalCodeblockType(self.qec_code.k, self.qec_code.n)
        input_types = (codeblock_type,)
        output_types = (codeblock_type,)

        block = Block(arg_types=input_types)

        with ImplicitBuilder(block):
            (codeblock,) = block.args

            # allocate auxiliary qubits
            aux_allocate_ops = (qecp.AllocAuxQubitOp() for row in self.qec_code.x_tanner)
            aux_qubits = [op.results[0] for op in aux_allocate_ops]

            # apply X-check gate+measurement pattern
            measure_ops, encoded_codeblock = self.check_pattern(
                aux_qubits, codeblock, check_type=CheckType.X
            )

            # deallocate the auxiliary qubits
            for meas_op in measure_ops:
                qecp.DeallocAuxQubitOp(meas_op.results[1])

            # return the encoded codeblock
            func.ReturnOp(encoded_codeblock)

        funcOp = func.FuncOp(
            name=f"encode_zero_{self.qec_code.name}",
            function_type=(input_types, output_types),
            visibility="private",
            region=Region([block]),
        )

        return funcOp

    def check_pattern(
        self,
        in_aux_qbs: Iterable[qecp.QecPhysicalQubitSSAValue],
        in_codeblock: qecp.PhysicalCodeBlockSSAValue,
        check_type: CheckType,
    ) -> tuple[Iterable[qecp.MeasureOp], qecp.PhysicalCodeBlockSSAValue]:
        """Contains the ops to perform a QEC check on the provided auxiliary qubits and codeblock.
        Intended to be called inside `builder.ImplicitBuilder` to add these operations to a block.

        This implementation uses the convention where all two-qubit gates are CNOTs - for example,
        see Figure 5a. and Figure 5d. in arXiv: 2304.08678

        This pattern includes measurement of the auxiliary qubits, and returns the MeasureOps, as
        well as the codeblock after the check pattern has been applied. It is not responsible for
        aux qubit allocation, aux qubit deallocation, or handling of measurement outputs (for
        example sending them to a decoder).

        Args:
            aux_qbs_in (Iterable[qecp.QecPhysicalQubitSSAValue]): The auxiliary qubits to be used
                in the check
            codeblock (qecp.PhysicalCodeBlockSSAValue): The codeblock of data-qubits to be used
                in the check
            check_type (CheckType): Which check pattern will be performed.

        Returns:
            Iterable[qecp.MeasureOp]: a list of the ops measuring the auxiliary qubits
            qecp.PhysicalCodeblockType: the codeblock after the check pattern has been applied
        """

        aux_qubits = in_aux_qbs
        tanner_graph, cnot_fn = self._get_cnot_and_tanner_graph(check_type)

        if check_type == CheckType.X:
            # auxiliary qubits are prepared in the |+> state
            hadamard_ops = [qecp.HadamardOp(qb) for qb in aux_qubits]
            aux_qubits = [h_op.results[0] for h_op in hadamard_ops]

        # extract data qubits
        extract_ops = [qecp.ExtractQubitOp(in_codeblock, i) for i in range(self.qec_code.n)]
        data_qubits = [ext_op.results[0] for ext_op in extract_ops]

        # apply CNOTs between data and auxiliary qubits based on tanner graph
        aux_qbs_out = []
        for aux_qb, row in zip(aux_qubits, tanner_graph):
            data_qbs_out = []
            for data_qb, val in zip(data_qubits, row, strict=True):
                if val:
                    aux_qb, data_qb = cnot_fn(aux_qb, data_qb)
                data_qbs_out.append(data_qb)
            data_qubits = data_qbs_out
            aux_qbs_out.append(aux_qb)

        # insert data qubits back into the codeblock
        codeblock = in_codeblock
        for i in range(self.qec_code.n):
            insert_op = qecp.InsertQubitOp(codeblock, i, data_qbs_out[i])
            codeblock = insert_op.results[0]

        if check_type == "X":
            # re-set auxiliary qubits
            hadamard_ops = [qecp.HadamardOp(aux) for aux in aux_qbs_out]
            aux_qbs_out = [h_op.results[0] for h_op in hadamard_ops]

        # measure auxiliary qubits
        measure_ops = [qecp.MeasureOp(qb) for qb in aux_qbs_out]

        return measure_ops, codeblock

    def _get_cnot_and_tanner_graph(self, check_type: CheckType) -> tuple[np.ndarray, Callable]:
        """Get the appropriate tanner graph and the function for applying CNOTs in an
        QEC check based on the check type."""

        if check_type == CheckType.X:
            # we use CNOT(aux, data) and X-tanner graph
            tanner_graph = self.qec_code.x_tanner

            def cnot_fn(aux_qb, data_qb):
                cnot_op = qecp.CnotOp(aux_qb, data_qb)
                aux_qb, data_qb = cnot_op.results
                return aux_qb, data_qb

        elif check_type == CheckType.Z:
            # we use CNOT(data, aux) and Z-tanner graph
            tanner_graph = self.qec_code.z_tanner

            def cnot_fn(aux_qb, data_qb):
                cnot_op = qecp.CnotOp(data_qb, aux_qb)
                data_qb, aux_qb = cnot_op.results
                return aux_qb, data_qb

        else:
            raise CompileError(
                f"Only CSS codes are supported, check_type must be X or Z but received {check_type}"
            )

        return tanner_graph, cnot_fn


convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
