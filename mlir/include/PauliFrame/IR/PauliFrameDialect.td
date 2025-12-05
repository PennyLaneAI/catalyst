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

        A single Pauli record *Rq* tracks all the Pauli gates that are applied on qubit *q*. Every
        set of tracked Pauli gates can be reduced to one of the elements in the set {I, X, Z, XZ},
        hence a single Pauli record can be stored using two bits, encoded as X- and Z-parity bits:

            I = (0, 0),   X = (1, 0),   Z = (0, 1),   XZ = (1, 1)

        The operations in this dialect aim to represent the Pauli frame tracking components of the
        following five quantum processes required to maintain a system of universal quantum
        computation, as described in Ref. [2]:

            1. Initialization of a qubit to |0>:
                a. Set Pauli record of target qubit to I.
                b. Initialize target qubit to |0>.
            2. Measurement:
                a. Measure target qubit.
                b. Correct measurement result based on Pauli record.
            3. Pauli gates:
                a. Update Pauli record of target qubit accordingly (the Pauli gates themselves are
                   not physically applied to the qubit).
            4. Clifford gates:
                a. Update Pauli record(s) of target qubit(s).
                b. Apply Clifford gate on target qubit(s).
            5. Non-Clifford gates:
                a. Flush Pauli record(s) of target qubit(s). *Flushing* the Pauli record of a qubit
                   refers to physically applying the Pauli gates stored in the record on that qubit
                   and then resetting its record to I.
                b. Apply non-Clifford gate on target qubit(s).

        References
        ----------

        [1] Knill, E. Quantum computing with realistically noisy devices. Nature 434, 39-44 (2005).
        https://doi.org/10.1038/nature03350.

        [2] Riesebos, L., et al. Pauli Frames for Quantum Computer Architectures.
        DAC '17: Proceedings of the 54th Annual Design Automation Conference 76, 1-6 (2017).
        https://doi.org/10.1145/3061639.3062300.

        > [!IMPORTANT]
        > The pauli_frame dialect is experimental and will not maintain API stability between releases.
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
    let useDefaultAttributePrinterParser = 1;

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
