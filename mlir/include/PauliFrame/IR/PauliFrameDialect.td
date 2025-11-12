// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PAULI_FRAME_DIALECT
#define PAULI_FRAME_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PauliFrame dialect definition.
//===----------------------------------------------------------------------===//

def PauliFrame_Dialect : Dialect {
    let summary = "A dialect for Pauli frame tracking.";
    let description = [{
        The Pauli frame tracking dialect includes a set of abstractions and operations for
        interacting with an external Pauli frame tracking library. A *Pauli frame tracker* [1] is a
        system that tracks gates from the Pauli group in classical electronics instead of applying
        them to qubits on the device. Doing so reduces the number of quantum operations applied to
        the system, thus reducing the probability of errors.

        [1] Knill, E. Quantum computing with realistically noisy devices. Nature 434, 39-44 (2005).
        https://doi.org/10.1038/nature03350.

        > [!IMPORTANT]
        > The pauli_frame dialect is experimental and and will not maintain API stability between releases.
        > Use at your own risk.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "pauli_frame";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::pauli_frame";

    let dependentDialects = [
        "quantum::QuantumDialect"
    ];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    /// Uncomment the line below if defining attributes for the PauliFrame dialect
    // let useDefaultAttributePrinterParser = 1;

    /// Uncomment the line below if defining types for the PauliFrame dialect
    // let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// PauliFrame dialect types.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the PauliFrame dialect
// class PauliFrame_Type<string name, string typeMnemonic, list<Trait> traits = []>
//         : TypeDef<PauliFrame_Dialect, name, traits> {
//     let mnemonic = typeMnemonic;
// }


//===----------------------------------------------------------------------===//
// PauliFrame dialect base operation.
//===----------------------------------------------------------------------===//

class PauliFrame_Op<string mnemonic, list<Trait> traits = []> :
        Op<PauliFrame_Dialect, mnemonic, traits>;


#endif // PAULI_FRAME_DIALECT
