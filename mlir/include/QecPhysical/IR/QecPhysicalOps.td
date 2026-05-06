// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QEC_PHYSICAL_OPS
#define QEC_PHYSICAL_OPS

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

include "QecPhysical/IR/QecPhysicalDialect.td"
include "QecPhysical/IR/QecPhysicalTypes.td"


//===----------------------------------------------------------------------===//
// QecPhysical dialect operations.
//===----------------------------------------------------------------------===//

// QecPhysical dialect base operation
class QecPhysical_Op<string mnemonic, list<Trait> traits = []> :
        Op<QecPhysical_Dialect, mnemonic, traits>;

def AllocOp : QecPhysical_Op<"alloc"> {
    let summary = "Allocate a physical hyper-register containing a sequence of physical codeblocks.";

    let results = (outs
        PhysicalHyperRegisterType:$hyper_reg
    );

    // The width of the hyper-register is inferred from the type.
    // The empty parentheses `()` are to mirror the syntax of the memref.alloc and similar ops.
    // Currently we do not support dynamic-width hyper-registers, but if we do in the future, the
    // value specifying the width can be placed in-between the parentheses.
    let assemblyFormat = [{
        `(` `)` attr-dict `:` qualified(type($hyper_reg))
    }];

    let hasCanonicalizeMethod = 1;
}

def DeallocOp : QecPhysical_Op<"dealloc"> {
    let summary = "Deallocate a physical hyper-register.";

    let arguments = (ins
        PhysicalHyperRegisterType:$hyper_reg
    );

    let assemblyFormat = [{
        $hyper_reg attr-dict `:` qualified(type($hyper_reg))
    }];

    let hasCanonicalizeMethod = 1;
}

def AllocAuxQubitOp : QecPhysical_Op<"alloc_aux"> {
    let summary = "Allocate a single auxiliary QEC physical qubit.";

    let results = (outs
        QecPhysicalQubitType:$qubit
    );

    let assemblyFormat = [{
        attr-dict `:` qualified(type($qubit))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
}

def DeallocAuxQubitOp : QecPhysical_Op<"dealloc_aux"> {
    let summary = "Deallocate a single auxiliary QEC physical qubit.";

    let arguments = (ins
        QecPhysicalQubitType:$qubit
    );

    let assemblyFormat = [{
        $qubit attr-dict `:` qualified(type($qubit))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
}

def ExtractCodeblockOp : QecPhysical_Op<"extract_block"> {
    let summary = "Extract a physical codeblock value from a hyper-register.";

    let arguments = (ins
        PhysicalHyperRegisterType:$hyper_reg,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        PhysicalCodeblockType:$codeblock
    );

    let assemblyFormat = [{
        $hyper_reg `[` ($idx^):($idx_attr)? `]` attr-dict `:` qualified(type($hyper_reg)) `->` qualified(type($codeblock))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def InsertCodeblockOp : QecPhysical_Op<"insert_block", [AllTypesMatch<["in_hyper_reg", "out_hyper_reg"]>]> {
    let summary = "Update the physical codeblock value of a hyper-register.";

    let arguments = (ins
        PhysicalHyperRegisterType:$in_hyper_reg,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr,
        PhysicalCodeblockType:$codeblock
    );

    let results = (outs
        PhysicalHyperRegisterType:$out_hyper_reg
    );

    let assemblyFormat = [{
        $in_hyper_reg `[` ($idx^):($idx_attr)? `]` `,` $codeblock attr-dict `:` qualified(type($in_hyper_reg)) `,` qualified(type($codeblock))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def ExtractQubitOp : QecPhysical_Op<"extract"> {
    let summary = "Extract a physical qubit value from a codeblock.";

    let description = [{
        This operation extracts a QEC physical qubit value from a physical codeblock. The qubit
        value is restricted to have the 'data' role; in other words, an auxiliary qubit cannot be
        extracted from a physical codeblock.
    }];

    let arguments = (ins
        PhysicalCodeblockType:$codeblock,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        QecPhysicalQubitType:$qubit
    );

    let assemblyFormat = [{
        $codeblock `[` ($idx^):($idx_attr)? `]` attr-dict `:` qualified(type($codeblock)) `->` qualified(type($qubit))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def InsertQubitOp : QecPhysical_Op<"insert", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "Update the physical qubit value of a codeblock.";

    let description = [{
        This operation updates the value of a QEC physical qubit in a physical codeblock. The qubit
        value is restricted to have the 'data' role; in other words, an auxiliary qubit cannot be
        inserted into a physical codeblock.
    }];

    let arguments = (ins
        PhysicalCodeblockType:$in_codeblock,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr,
        QecPhysicalQubitType:$qubit
    );

    let results = (outs
        PhysicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        $in_codeblock `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` qualified(type($in_codeblock)) `,` qualified(type($qubit))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

/**
 * Base class for single-qubit physical gate operations.
 *
 * An operation that inherits from this class represents a physical gate operation applied to a QEC
 * physical qubit. For example,
 *
 * ```mlir
 * %1 = qecp.hadamard %0 : !qecp.qubit<data>
 * ```
 *
 * represents a physical Hadamard operation applied to the physical data qubit `%0`. Single-qubit
 * physical gate operations can be applied to both data and auxiliary qubits.
 *
 * Adjoint operations are supported by adding the `adj` unit attribute. For example, a physical S†
 * gate operation is represented as follows:
 *
 * ```mlir
 * %1 = qecp.s %0 adj : !qecp.qubit<data>
 * ```
 */
class SingleQubitPhysicalGate_Op<string mnemonic, list<Trait> traits = []> :
        QecPhysical_Op<mnemonic, traits> {
    let arguments = (ins
        QecPhysicalQubitType:$in_qubit,
        UnitAttr:$adjoint
    );

    let results = (outs
        QecPhysicalQubitType:$out_qubit
    );

    let assemblyFormat = [{
        $in_qubit (`adj` $adjoint^)? attr-dict `:` qualified(type($in_qubit))
    }];
}

def IdentityOp : SingleQubitPhysicalGate_Op<"identity", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Identity gate operation.";
}

def PauliXOp : SingleQubitPhysicalGate_Op<"x", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Pauli X gate operation.";
}

def PauliYOp : SingleQubitPhysicalGate_Op<"y", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Pauli Y gate operation.";
}

def PauliZOp : SingleQubitPhysicalGate_Op<"z", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Pauli Z gate operation.";
}

def HadamardOp : SingleQubitPhysicalGate_Op<"hadamard", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Hadamard gate operation.";
}

def SOp : SingleQubitPhysicalGate_Op<"s", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical S (π/2 phase) gate operation.";
}

def CnotOp : QecPhysical_Op<"cnot", [
    AllTypesMatch<["in_ctrl_qubit", "out_ctrl_qubit"]>,
    AllTypesMatch<["in_trgt_qubit", "out_trgt_qubit"]>
]> {
    let summary = "A physical CNOT gate operation.";

    let arguments = (ins
        QecPhysicalQubitType:$in_ctrl_qubit,
        QecPhysicalQubitType:$in_trgt_qubit
    );

    let results = (outs
        QecPhysicalQubitType:$out_ctrl_qubit,
        QecPhysicalQubitType:$out_trgt_qubit
    );

    let assemblyFormat = [{
        $in_ctrl_qubit `,` $in_trgt_qubit attr-dict `:` qualified(type($in_ctrl_qubit)) `,` qualified(type($in_trgt_qubit))
    }];
}

/**
 * A physical Rot gate operation.
 *
 * ```mlir
 * %1 = qecp.rot (%phi, %theta, %omega) %0 : !qecp.qubit<data>
 * ```
 */
def RotOp: QecPhysical_Op<"rot", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical Rot gate operation.";

    let description = [{
        This operation is for physical noise injection only.
    }];

    let arguments = (ins
        F64:$phi,
        F64:$theta,
        F64:$omega,
        QecPhysicalQubitType:$in_qubit
    );

    let results = (outs
        QecPhysicalQubitType:$out_qubit
    );

    let assemblyFormat = [{
        `(` $phi `,` $theta `,` $omega `)` $in_qubit attr-dict `:` qualified(type($in_qubit))
    }];
}

def MeasureOp : QecPhysical_Op<"measure", [AllTypesMatch<["in_qubit", "out_qubit"]>]> {
    let summary = "A physical single-qubit projective measurement in the computational basis.";

    let arguments = (ins
        QecPhysicalQubitType:$in_qubit
    );

    let results = (outs
        I1:$mres,
        QecPhysicalQubitType:$out_qubit
    );

    let assemblyFormat = [{
        $in_qubit attr-dict `:` type($mres) `,` qualified(type($in_qubit))
    }];
}

def AssembleTannerGraphOp : QecPhysical_Op<"assemble_tanner"> {
    let summary = "Assemble a Tanner graph in CSC form from the given input arrays.";

    let arguments = (ins
        AnyTypeOf<[1DTensorOf<[AnyInteger]>, MemRefRankOf<[AnyInteger], [1]>]>:$row_idx,
        AnyTypeOf<[1DTensorOf<[AnyInteger]>, MemRefRankOf<[AnyInteger], [1]>]>:$col_ptr
    );

    let results = (outs
        TannerGraphType:$tanner_graph
    );

    let assemblyFormat = [{
        $row_idx `,` $col_ptr attr-dict `:` type($row_idx) `,` type($col_ptr) `->` qualified(type($tanner_graph))
    }];

    let extraClassDeclaration = [{
        bool isBufferized() {
            return mlir::isa<mlir::MemRefType>(getRowIdx().getType()) &&
                   mlir::isa<mlir::MemRefType>(getColPtr().getType());
        }
    }];

    let hasVerifier = 1;
}

def DecodeEsmCssOp : QecPhysical_Op<"decode_esm_css"> {
    let summary = [{
        Decode an ESM for a CSS code and return the index (indices) in the codeblock where the
        error(s) occurred."
    }];

    let description = [{
        This operation decodes an error-syndrome measurement (ESM), given as a one-dimensional
        tensor or memref of bits, for a CSS code specified by its Tanner graph. The operation
        returns the index (or indices) in the physical codeblock where the error(s) occurred.

        Importantly, this operation does not return the *type* of error that occurred (e.g. X, Z).
        It assumes that the QEC code is a CSS code and that separate Tanner graphs are used for X
        and Z checks. It is therefore up to the user of this operation to determine what type(s) of
        correction(s) to apply given the Tanner graph that was supplied.

        If no correctable error was detected, this operation returns an index value of -1.

        #### Example

        ```mlir
        // CSS Tanner graphs, separated for X and Z checks
        %tannerX = ... : !qecp.tanner_graph<8, 6, i32>
        %tannerZ = ... : !qecp.tanner_graph<8, 6, i32>

        // ESM
        %esm0, %aux01 = qecp.measure %aux00 : i1, !qecp.qubit<aux>
        %esm1, %aux11 = qecp.measure %aux10 : i1, !qecp.qubit<aux>
        %esm = tensor.from_elements %esm0, %esm1 : tensor<2xi1>

        // Decode ESM, return single index in codeblock where error occurred
        %err_idx = qecp.decode_esm_css(%tannerX : !qecp.tanner_graph<8, 6, i32>) %esm : tensor<2xi1> -> tensor<1xindex>
        ```

        .. note::

            The ``err_idx_in`` field is needed only after bufferization.
    }];

    let arguments = (ins
        AnyTypeOf<[1DTensorOf<[I1]>, MemRefRankOf<[I1], [1]>]>:$esm,
        TannerGraphType:$tanner_graph,
        Optional<MemRefRankOf<[Index], [1]>>:$err_idx_in
    );

    let results = (outs
        Optional<1DTensorOf<[Index]>>:$err_idx
    );

    let extraClassDeclaration = [{
        bool isBufferized() {
            return mlir::isa<mlir::MemRefType>(getEsm().getType()) && getResultTypes().empty();
        }
    }];

    let assemblyFormat = [{
        `(` $tanner_graph `:` qualified(type($tanner_graph)) `)` $esm
        ( `in` `(` $err_idx_in^ `:` type($err_idx_in) `)` )?
        attr-dict `:` type($esm) ( `->` type($err_idx)^ )?
    }];
}

def DecodePhysicalMeasurementOp : QecPhysical_Op<"decode_physical_meas"> {
    let summary = [{
        Decode physical measurement results and return the corresponding logical measurement.
    }];

    let description = [{
        This operation decodes the results of a transversal measurement operation acting on a
        physical codeblock and returns the corresponding logical measurement result(s) in the
        computational basis. The logical measurement results are returned as a one-dimensional
        tensor with shape=(k,), where k is the number of QEC logical qubits encoded by the physical
        codeblock.

        .. note::

            The ``logical_measurements_in`` field is needed only after bufferization.
    }];

    let arguments = (ins
        AnyTypeOf<[1DTensorOf<[I1]>, MemRefRankOf<[I1], [1]>]>:$physical_measurements,
        Optional<MemRefRankOf<[I1], [1]>>:$logical_measurements_in
    );

    let results = (outs
        Optional<1DTensorOf<[I1]>>:$logical_measurements
    );

    let extraClassDeclaration = [{
        bool isBufferized() {
            return mlir::isa<mlir::MemRefType>(getPhysicalMeasurements().getType()) &&
                   getResultTypes().empty();
        }
    }];

    let assemblyFormat = [{
        $physical_measurements
        ( `in` `(` $logical_measurements_in^ `:` type($logical_measurements_in) `)` )?
        attr-dict `:` type($physical_measurements) ( `->` type($logical_measurements)^ )?
    }];
}


#endif // QEC_PHYSICAL_OPS
