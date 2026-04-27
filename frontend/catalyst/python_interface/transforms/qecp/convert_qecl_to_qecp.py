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

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
from xdsl import builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.builtin import IndexType, TensorType, i1, i32
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
from catalyst.python_interface.transforms.qecp.tanner_graph_lib import dense_tanner_graph_to_csc
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

        in_codeblock = cast(
            qecl.LogicalCodeBlockSSAValue | qecp.PhysicalCodeBlockSSAValue, op.in_codeblock
        )

        if (k := in_codeblock.type.k.value.data) != self.qec_code.k:
            raise CompileError(
                f"Circuit expressed in the qecl dialect with k={k} is not compatible with "
                f"lowering to a code with k={self.qec_code.k}"
            )

        callee = builtin.SymbolRefAttr(self.encode_subroutine.sym_name)
        arguments = (in_codeblock,)
        return_types = self.encode_subroutine.function_type.outputs.data
        callOp = func.CallOp(callee, arguments, return_types)

        rewriter.replace_op(op, callOp)


# MARK: QEC Cycle Op Pattern


@dataclass
class QecCycleOpConversion(RewritePattern):
    """Converts qecl.encode [zero] to the equivalent subroutine of qecp gates"""

    qec_code: QecCode
    qec_cycle_subroutine: func.FuncOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qecl.QecCycleOp, rewriter: PatternRewriter):
        """Rewrite pattern for `qecl.encode [zero]` op"""

        in_codeblock = cast(
            qecl.LogicalCodeBlockSSAValue | qecp.PhysicalCodeBlockSSAValue, op.in_codeblock
        )

        if (k := in_codeblock.type.k.value.data) != self.qec_code.k:
            raise CompileError(
                f"Circuit expressed in the qecl dialect with k={k} is not compatible with "
                f"lowering to a code with k={self.qec_code.k}"
            )

        callee = builtin.SymbolRefAttr(self.qec_cycle_subroutine.sym_name)
        arguments = (in_codeblock,)
        return_types = self.qec_cycle_subroutine.function_type.outputs.data
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

        module_block = op.regions[0].blocks.first
        assert module_block is not None, "Module has no block"

        tanner_x, tanner_z = self.insert_tanner_graph_ops_into_block(module_block)

        # Insert subroutines that implement the QEC protocols
        encode_funcop = self.create_encode_subroutine()
        module_block.add_op(encode_funcop)

        qec_cycle_funcop = self.create_qec_cycle_subroutine(tanner_x=tanner_x, tanner_z=tanner_z)
        module_block.add_op(qec_cycle_funcop)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CodeblockTypeConversion(qec_code=self.qec_code),
                    HyperRegisterTypeConversion(qec_code=self.qec_code),
                    AllocationConversion(),
                    DeallocationConversion(),
                    InsertBlockConversion(),
                    ExtractBlockConversion(),
                    EncodeOpConversion(qec_code=self.qec_code, encode_subroutine=encode_funcop),
                    QecCycleOpConversion(
                        qec_code=self.qec_code, qec_cycle_subroutine=qec_cycle_funcop
                    ),
                ]
            )
        ).rewrite_module(op)

    def insert_tanner_graph_ops_into_block(
        self, block: Block
    ) -> tuple[OpResult[qecp.TannerGraphType], OpResult[qecp.TannerGraphType]]:
        """Insert Tanner graph operations into the given block.

        The operations are inserted at the beginning of the block.

        Returns the X and Z Tanner graph SSA values from the `qecp.assemble_tanner` ops (we assume
        a CSS code here and therefore have separate X and Z Tanner graphs).
        """
        x_tanner_row_idx_array, x_tanner_col_ptr_array = dense_tanner_graph_to_csc(
            self.qec_code.x_tanner
        )
        x_tanner_row_idx_const_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(i32, shape=x_tanner_row_idx_array.shape),
                data=x_tanner_row_idx_array.tolist(),
            )
        )
        x_tanner_col_ptr_const_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(i32, shape=x_tanner_col_ptr_array.shape),
                data=x_tanner_col_ptr_array.tolist(),
            )
        )
        assemble_x_tanner_op = qecp.AssembleTannerGraphOp(
            row_idx=x_tanner_row_idx_const_op,
            col_ptr=x_tanner_col_ptr_const_op,
            tanner_graph_type=qecp.TannerGraphType(
                x_tanner_row_idx_array.shape[0], x_tanner_col_ptr_array.shape[0], i32
            ),
        )

        z_tanner_row_idx_array, z_tanner_col_ptr_array = dense_tanner_graph_to_csc(
            self.qec_code.z_tanner
        )
        z_tanner_row_idx_const_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(i32, shape=z_tanner_row_idx_array.shape),
                data=z_tanner_row_idx_array.tolist(),
            )
        )
        z_tanner_col_ptr_const_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(i32, shape=z_tanner_col_ptr_array.shape),
                data=z_tanner_col_ptr_array.tolist(),
            )
        )
        assemble_z_tanner_op = qecp.AssembleTannerGraphOp(
            row_idx=z_tanner_row_idx_const_op,
            col_ptr=z_tanner_col_ptr_const_op,
            tanner_graph_type=qecp.TannerGraphType(
                z_tanner_row_idx_array.shape[0], z_tanner_col_ptr_array.shape[0], i32
            ),
        )

        ops_to_insert = (
            x_tanner_row_idx_const_op,
            x_tanner_col_ptr_const_op,
            assemble_x_tanner_op,
            z_tanner_row_idx_const_op,
            z_tanner_col_ptr_const_op,
            assemble_z_tanner_op,
        )

        if block.first_op is None:
            block.add_ops(ops_to_insert)
        else:
            block.insert_ops_before(
                (
                    x_tanner_row_idx_const_op,
                    x_tanner_col_ptr_const_op,
                    assemble_x_tanner_op,
                    z_tanner_row_idx_const_op,
                    z_tanner_col_ptr_const_op,
                    assemble_z_tanner_op,
                ),
                block.first_op,
            )

        return assemble_x_tanner_op.tanner_graph, assemble_z_tanner_op.tanner_graph

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

        with builder.ImplicitBuilder(block):
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

    def create_qec_cycle_subroutine(
        self, tanner_x: qecp.TannerGraphSSAValue, tanner_z: qecp.TannerGraphSSAValue
    ) -> func.FuncOp:
        """FIXME

        Create a subroutine that takes in a codeblock, encodes it in the zero state for
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

        with builder.ImplicitBuilder(block):
            in_codeblock = cast(BlockArgument[qecp.PhysicalCodeblockType], block.args[0])

            # Apply X checks pattern for Z corrections
            x_out_codeblock = self._qec_cycle_css_pattern(in_codeblock, CheckType.X, tanner_x)

            # Apply Z checks pattern for X corrections
            z_out_codeblock = self._qec_cycle_css_pattern(x_out_codeblock, CheckType.Z, tanner_z)

            # Return the corrected codeblock
            func.ReturnOp(z_out_codeblock)

        funcOp = func.FuncOp(
            name=f"qec_cycle_{self.qec_code.name}",
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
    ) -> tuple[list[qecp.MeasureOp], qecp.PhysicalCodeBlockSSAValue]:
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

    def _qec_cycle_css_pattern(
        self,
        in_codeblock: qecp.PhysicalCodeBlockSSAValue,
        check_type: CheckType,
        tanner_graph: qecp.TannerGraphSSAValue,
    ) -> OpResult[qecp.PhysicalCodeblockType]:
        """TODO"""
        # Allocate auxiliary qubits for ESM checks
        aux_allocate_ops = (qecp.AllocAuxQubitOp() for row in self.qec_code.x_tanner)
        aux_qubits = [
            cast(OpResult[qecp.QecPhysicalQubitType], op.results[0]) for op in aux_allocate_ops
        ]

        # Apply gate+measurement pattern for the check
        measure_ops, out_codeblock = self.check_pattern(
            aux_qubits, in_codeblock, check_type=check_type
        )

        # Checks are done; deallocate the auxiliary qubits
        for x_meas_op in measure_ops:
            qecp.DeallocAuxQubitOp(x_meas_op.out_qubit)

        # Pack measurement results into a tensor for decoding
        pack_mres_tensor_op = tensor.FromElementsOp.build(
            operands=([meas_op.mres for meas_op in measure_ops],),
            result_types=(TensorType(i1, shape=(len(measure_ops),)),),
        )

        # Decode ESM syndrome
        decode_esm_op = qecp.DecodeEsmCssOp(
            tanner_graph=tanner_graph,
            esm=pack_mres_tensor_op.result,
            err_idx_type=TensorType(IndexType(), shape=(1,)),
        )

        # Apply correction
        err_idx = cast(OpResult[IndexType], decode_esm_op.err_idx)
        extract_err_qubit_op = qecp.ExtractQubitOp(codeblock=out_codeblock, idx=err_idx)
        err_qubit = extract_err_qubit_op.qubit

        match check_type:
            case CheckType.X:
                qecp.PauliZOp(in_qubit=err_qubit)
            case CheckType.Z:
                qecp.PauliZOp(in_qubit=err_qubit)
            case _:
                assert False, f"Unknown CheckType: '{check_type}'"

        insert_err_qubit_op = qecp.InsertQubitOp(
            in_codeblock=out_codeblock, idx=err_idx, qubit=err_qubit
        )

        # Return updated codeblock SSA value
        return insert_err_qubit_op.out_codeblock


convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
