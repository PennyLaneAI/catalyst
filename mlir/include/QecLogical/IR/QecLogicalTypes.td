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

#ifndef QEC_LOGICAL_TYPES
#define QEC_LOGICAL_TYPES

include "mlir/IR/AttrTypeBase.td"

include "QecLogical/IR/QecLogicalDialect.td"

//===----------------------------------------------------------------------===//
// QecLogical dialect types.
//===----------------------------------------------------------------------===//

class QecLogical_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QecLogical_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}


def LogicalCodeblockType : QecLogical_Type<"Codeblock", "codeblock"> {
    let summary = "A value-semantic logical codeblock";

    let description = [{
        A logical codeblock type is the fundamental unit of quantum information at the QEC logical
        layer of program representation. It represents a block of *k* logical qubits encoded by some
        number *n* of physical qubits for a general *((n, k, d))* QEC code, where *d* is the code
        distance.

        Unlike quantum registers, logical qubits cannot be extracted from or inserted into a logical
        codeblock. The reason for this design is that at the QEC physical layer below, it is
        generally not possible to extract the physical qubits that encode a single logical qubit in
        the codeblock for codes with k > 1. In general, operations applied on a single logical qubit
        in the logical codeblock are applied to the entire block of physical qubits.

        The logical codeblock type is code-agnostic, and in principle can be lowered to any physical
        QEC encoding representation for a given value of *k*, which specifies the *logical codeblock
        size*.

        #### Example:

        ```mlir
        // A logical codeblock that encodes one logical qubit (k = 1)
        !qecl.codeblock<1>
        ```
    }];

    let parameters = (ins
        TypeParameter<"int64_t", "Logical codeblock size">:$k
    );

    let assemblyFormat = "`<` $k `>`";

    // let genVerifyDecl = 1;
}

def LogicalHyperRegisterType : QecLogical_Type<"HyperRegister", "hyperreg"> {
    let summary = "A value-semantic logical hyper-register";

    let description = [{
        A logical hyper-register type is an abstraction that represents a sequence of logical
        codeblocks.

        The "dimensions" of a logical hyper-register are specified by its *width* (the number of
        logical codeblocks in the hyper-register), and the value *k*, the logical codeblock size
        (the number of logical qubits encoded in each codeblock).

        #### Example:

        ```mlir
        // A logical hyper-register containing three size-1 logical codeblocks (width = 3, k = 1)
        !qecl.hyperreg<3 x 1>
        ```
    }];

    let parameters = (ins
        TypeParameter<"int64_t", "Logical hyper-register width">:$width,
        TypeParameter<"int64_t", "Logical codeblock size">:$k
    );

    // TODO: By default, this assembly format expects whitespace before and after the 'x'.
    //       We could write custom printers/parsers as a workaround.
    let assemblyFormat = "`<` $width `x` $k `>`";

    // let genVerifyDecl = 1;
}


#endif // QEC_LOGICAL_TYPES
