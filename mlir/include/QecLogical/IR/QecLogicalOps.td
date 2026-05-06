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

#ifndef QEC_LOGICAL_OPS
#define QEC_LOGICAL_OPS

include "mlir/IR/OpBase.td"

include "QecLogical/IR/QecLogicalAttrDefs.td"
include "QecLogical/IR/QecLogicalDialect.td"
include "QecLogical/IR/QecLogicalTypes.td"


//===----------------------------------------------------------------------===//
// QecLogical dialect operations.
//===----------------------------------------------------------------------===//

// QecLogical dialect base operation
class QecLogical_Op<string mnemonic, list<Trait> traits = []> :
        Op<QecLogical_Dialect, mnemonic, traits>;

def AllocOp : QecLogical_Op<"alloc"> {
    let summary = "Allocate a logical hyper-register containing a sequence of logical codeblocks.";

    let results = (outs
        LogicalHyperRegisterType:$hyper_reg
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

def DeallocOp : QecLogical_Op<"dealloc"> {
    let summary = "Deallocate a logical hyper-register.";

    let arguments = (ins
        LogicalHyperRegisterType:$hyper_reg
    );

    let assemblyFormat = [{
        $hyper_reg attr-dict `:` qualified(type($hyper_reg))
    }];

    let hasCanonicalizeMethod = 1;
}

def ExtractCodeblockOp : QecLogical_Op<"extract_block"> {
    let summary = "Extract a logical codeblock value from a hyper-register.";

    let arguments = (ins
        LogicalHyperRegisterType:$hyper_reg,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        LogicalCodeblockType:$codeblock
    );

    let assemblyFormat = [{
        $hyper_reg `[` ($idx^):($idx_attr)? `]` attr-dict `:` qualified(type($hyper_reg)) `->` qualified(type($codeblock))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def InsertCodeblockOp : QecLogical_Op<"insert_block", [AllTypesMatch<["in_hyper_reg", "out_hyper_reg"]>]> {
    let summary = "Update the logical codeblock value of a hyper-register.";

    let arguments = (ins
        LogicalHyperRegisterType:$in_hyper_reg,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr,
        LogicalCodeblockType:$codeblock
    );

    let results = (outs
        LogicalHyperRegisterType:$out_hyper_reg
    );

    let assemblyFormat = [{
        $in_hyper_reg `[` ($idx^):($idx_attr)? `]` `,` $codeblock attr-dict `:` qualified(type($in_hyper_reg)) `,` qualified(type($codeblock))
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def EncodeOp : QecLogical_Op<"encode", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "Encode a logical codeblock to the specified logical state.";

    let arguments = (ins
        LogicalCodeblockType:$in_codeblock,
        LogicalCodeblockInitStateAttr:$init_state
    );

    let results = (outs
        LogicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        `[` $init_state `]` $in_codeblock attr-dict `:` qualified(type($in_codeblock))
    }];
}

def NoiseOp : QecLogical_Op<"noise", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "Inject physical noise on elements of a logical codeblock.";

    let arguments = (ins
        LogicalCodeblockType:$in_codeblock
    );

    let results = (outs
        LogicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        $in_codeblock attr-dict `:` qualified(type($in_codeblock))
    }];
}

def QecCycleOp : QecLogical_Op<"qec", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "Perform a single cycle of a quantum error-correction protocol.";

    let arguments = (ins
        LogicalCodeblockType:$in_codeblock
    );

    let results = (outs
        LogicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        $in_codeblock attr-dict `:` qualified(type($in_codeblock))
    }];
}

/**
 * Base class for single-qubit logical gate operations.
 *
 * An operation that inherits from this class represents a logical gate operation applied to the
 * logical qubit at the provided index in the logical codeblock. For example,
 *
 * ```mlir
 * %1 = qecl.hadamard %0[ 1] : !qecl.codeblock<3>
 * ```
 *
 * represents a logical Hadamard operation applied to the logical qubit at index `1` in the
 * codeblock `%0`, which encodes k = 3 logical qubits.
 *
 * Adjoint operations are supported by adding the `adj` unit attribute. For example, a logical S†
 * gate operation is represented as follows:
 *
 * ```mlir
 * %1 = qecl.s %0[ 1] adj : !qecl.codeblock<3>
 * ```
 */
class SingleQubitLogicalGate_Op<string mnemonic, list<Trait> traits = []> :
        QecLogical_Op<mnemonic, traits> {
    let arguments = (ins
        LogicalCodeblockType:$in_codeblock,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr,
        UnitAttr:$adjoint
    );

    let results = (outs
        LogicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        $in_codeblock `[` ($idx^):($idx_attr)? `]` (`adj` $adjoint^)? attr-dict `:` qualified(type($in_codeblock))
    }];

    let hasVerifier = 1;
}

def IdentityOp : SingleQubitLogicalGate_Op<"identity", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical identity gate operation.";
}

def PauliXOp : SingleQubitLogicalGate_Op<"x", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical Pauli X gate operation.";
}

def PauliYOp : SingleQubitLogicalGate_Op<"y", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical Pauli Y gate operation.";
}

def PauliZOp : SingleQubitLogicalGate_Op<"z", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical Pauli Z gate operation.";
}

def HadamardOp : SingleQubitLogicalGate_Op<"hadamard", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical Hadamard gate operation.";
}

def SOp : SingleQubitLogicalGate_Op<"s", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical S (π/2 phase) gate operation.";
}

def CnotOp : QecLogical_Op<"cnot", [
    AttrSizedOperandSegments,
    AllTypesMatch<["in_ctrl_codeblock", "out_ctrl_codeblock", "in_trgt_codeblock", "out_trgt_codeblock"]>
]> {
    let summary = "A logical inter-codeblock CNOT gate operation.";

    let description = [{
        This operation represents a logical inter-codeblock CNOT gate operation applied to the
        logical qubits at the provided indices in the respective control and target logical
        codeblocks. For example,

        ```mlir
        %2, %3 = qecl.cnot %0[ 1], %1[ 2] : !qecl.codeblock<3>, !qecl.codeblock<3>
        ```

        represents a logical CNOT operation applied to the logical qubit at index `1` in the
        codeblock `%0` (the control qubit) and the logical qubit at index `2` in the codeblock `%1`
        (the target qubit), where both codeblocks encode k = 3 logical qubits.

        Note that this operation cannot represent an intra-codeblock CNOT operation—that is, a CNOT
        operation where the control and target qubits are encoded in the same logical codeblock (for
        k >= 2).
    }];

    let arguments = (ins
        LogicalCodeblockType:$in_ctrl_codeblock,
        Optional<Index>:$idx_ctrl,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_ctrl_attr,
        LogicalCodeblockType:$in_trgt_codeblock,
        Optional<Index>:$idx_trgt,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_trgt_attr
    );

    let results = (outs
        LogicalCodeblockType:$out_ctrl_codeblock,
        LogicalCodeblockType:$out_trgt_codeblock
    );

    let assemblyFormat = [{
        $in_ctrl_codeblock `[` ($idx_ctrl^):($idx_ctrl_attr)? `]` `,`
        $in_trgt_codeblock `[` ($idx_trgt^):($idx_trgt_attr)? `]`
        attr-dict `:` qualified(type($in_ctrl_codeblock)) `,` qualified(type($in_trgt_codeblock))
    }];

    let hasVerifier = 1;
}

def MeasureOp : QecLogical_Op<"measure", [AllTypesMatch<["in_codeblock", "out_codeblock"]>]> {
    let summary = "A logical single-qubit projective measurement in the computational basis.";

    let description = [{
        This operation represents a logical, projective computational-basis measurement of the
        logical qubit at the provided index in the logical codeblock. For example,

        ```mlir
        %mres, %1 = qecl.measure %0[ 1] : i1, !qecl.codeblock<3>
        ```

        represents a measurement of the logical qubit at index `1` in the codeblock `%0`, which
        encodes k = 3 logical qubits. The result of the measurement is returned as the value
        `%mres`.

        Note that unlike the `quantum.measure` operation, this operation does not currently support
        the `postselect` attribute to select the basis state of the qubit post-measurement.
    }];

    let arguments = (ins
        LogicalCodeblockType:$in_codeblock,
        Optional<Index>:$idx,
        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        I1:$mres,
        LogicalCodeblockType:$out_codeblock
    );

    let assemblyFormat = [{
        $in_codeblock `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($mres) `,` qualified(type($in_codeblock))
    }];

    let hasVerifier = 1;
}


#endif // QEC_LOGICAL_OPS
