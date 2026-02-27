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

#ifndef QEC_LOGICAL_DIALECT
#define QEC_LOGICAL_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// QecLogical dialect definition.
//===----------------------------------------------------------------------===//

def QecLogical_Dialect : Dialect {
    let summary = "A value-semantics dialect for logical QEC program representation.";
    let description = [{
        The QEC logical dialect is an experimental dialect for logical QEC program representation.
        In its current form, it is intended to sit one layer below the `quantum` dialect, which
        expresses the program in terms of abstract qubits and abstract operations on these qubits.
        At the QEC logical layer of program representation, abstract qubits are grouped into logical
        codeblocks and operations on the abstract qubits are represented as operations on these
        codeblocks. In addition, the QEC logical layer includes abstractions for codeblock encoding
        and cycles of quantum error correction. The layer below the QEC logical dialect is the QEC
        physical dialect, which realizes the operations on the logical codeblocks as operations on
        the QEC physical qubits that encode the logical codeblock.

        A program expressed at the QEC logical layer is QEC-code agnostic, and in principle can be
        lowered to a QEC physical layer for any QEC code. The only restriction is that the logical
        codeblock size, *k*, must be chosen when lowering an abstract program to the QEC logical
        layer. The QEC logical dialect can generally be used as the initial logical expression of
        any ((n, k, d)) QEC encoding system, however it is primarily intended for [[n, k, d]]
        stabilizer codes.

        Similar to the `quantum` dialect, the QEC logical dialect follows value semantics, meaning
        that logical codeblocks and other types are treated as immutable SSA values rather than
        memory buffers that can be modified in-place. Operations on these values consume and return
        new values representing the "updated" state. However, contrary to the behaviour of registers
        in the `quantum` dialect, where qubit values are extracted from a register, acted upon, and
        inserted back into the register, logical qubits cannot be extracted from or inserted into
        logical codeblocks. Instead, logical qubits within the codeblock are acted upon by
        operations that consume the codeblock as well as an index to the target qubit within the
        codeblock.

        #### Example

        [TODO: Fill in example once the basic ops are defined (future PR)]
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qecl";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::qecl";

    /// This dialect depends on the following list of other dialects.
    // let dependentDialects = [];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    let useDefaultTypePrinterParser = 1;
}


#endif // QEC_LOGICAL_DIALECT
