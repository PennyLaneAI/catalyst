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


#endif // QEC_PHYSICAL_OPS
