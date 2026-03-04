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


#endif // QEC_LOGICAL_OPS
