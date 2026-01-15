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

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "mlir/IR/AttrTypeBase.td"

include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Quantum dialect types.
//===----------------------------------------------------------------------===//

class Quantum_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QuantumDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : Quantum_Type<"Qubit", "bit"> {
    let summary = "A value-semantic qubit (state).";

    let parameters = (ins
        DefaultValuedParameter<"QubitLevel", "QubitLevel::Abstract">:$level
    );

    let assemblyFormat = "(`<` $level^ `>`)?";

    // This allows the ODS generator to "build" the type automatically
    // using the Abstract qubit level when it needs a default.
    let builderCall = "$_builder.getType<quantum::QubitType>(quantum::QubitLevel::Abstract)";

    let extraClassDeclaration = [{
        // Get an instance of the QubitType with the default 'Abstract' level.
        static QubitType get(::mlir::MLIRContext *context) {
            return get(context, quantum::QubitLevel::Abstract);
        }
    }];
}
def QuregType : Quantum_Type<"Qureg", "reg"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
}
def ObservableType : Quantum_Type<"Observable", "obs"> {
    let summary = "A quantum observable for use in measurements.";
}
def ResultType : Quantum_Type<"Result", "res"> {
    let summary = "A quantum measurement result.";
}

// // Hierarchical qubit types for FTQC workloads
// def LogicalQubitType : Quantum_Type<"LogicalQubit", "bit.logical"> {
//     let summary = "Logical qubit";
//     let description = [{
//         Values with logical qubit type represent qubits at the logical layer of program
//         representation. In other words, they represent qubit values allocated and acted on by the
//         user program, and are agnostic to any specific QEC or hardware architecture.
//     }];
// }

// def QECQubitType : Quantum_Type<"QECQubit", "bit.qec"> {
//     let summary = "QEC qubit";
//     let description = [{
//         Values with QEC qubit type represent qubits at the quantum-error-corrected (QEC) layer of
//         program representation.
//     }];
// }

// def PhysicalQubitType : Quantum_Type<"PhysicalQubit", "bit.physical"> {
//     let summary = "Physical qubit";
//     let description = [{
//         Values with physical qubit type represent qubits at the physical layer of program
//         representation.
//     }];
// }

#endif // QUANTUM_TYPES
