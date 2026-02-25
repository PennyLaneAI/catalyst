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

#ifndef QEC_PHYSICAL_DIALECT
#define QEC_PHYSICAL_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// QecPhysical dialect definition.
//===----------------------------------------------------------------------===//

def QecPhysical_Dialect : Dialect {
    let summary = "A value-semantics dialect for physical QEC program representation.";
    let description = [{
        The QEC physical dialect is an experimental dialect for physical QEC program representation.
        In its current form, it is intended to sit one layer below the QEC logical dialect, which
        expresses the program in terms of logical operations acting on logical codeblocks. At the
        QEC physical layer of program representation, logical codeblocks, which encode *k* logical
        qubits, are lowered to physical codeblocks consisting of *n* physical qubits. Similarly,
        operations on logical codeblocks are lowered to their corresponding operations on the
        physical codeblocks.

        A specific QEC code must be chosen to express a program at the QEC physical layer. In
        principle, from this layer a program can be compiled to target any hardware platform. The
        QEC physical dialect can generally be used as the final "physical" expression of any ((n, k,
        d)) QEC encoding system, however it is primarily intended for [[n, k, d]] stabilizer codes.
        Here, "physical" is still used in a somewhat abstract sense, since this layer of abstraction
        is still agnostic to the underlying physical hardware.

        Similar to the `quantum` dialect, the QEC physical dialect follows value semantics, meaning
        that physical codeblocks and other types are treated as immutable SSA values rather than
        memory buffers that can be modified in-place. Operations on these values consume and return
        new values representing the "updated" state.

        #### Example

        [TODO: Fill in example once the basic ops are defined (future PR)]
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qecp";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::qecp";

    /// This dialect depends on the following list of other dialects.
    // let dependentDialects = [];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    let useDefaultTypePrinterParser = 1;
}


#endif // QEC_PHYSICAL_DIALECT
