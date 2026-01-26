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
        DefaultValuedParameter<"QubitLevel", "QubitLevel::Abstract">:$level,
        DefaultValuedParameter<"QubitRole", "QubitRole::Null">:$role
    );

    let assemblyFormat = "(`<` $level^ (`,` $role^ )? `>`)?";

    // This allows the ODS generator to "build" the type automatically using the
    // 'Abstract' qubit level and 'Null' qubit role and when it needs a default.
    let builderCall = [{
        $_builder.getType<quantum::QubitType>(
            quantum::QubitLevel::Abstract, quantum::QubitRole::Null)
    }];

    let extraClassDeclaration = [{
        // Get an instance of the QubitType with the default 'Abstract' level and 'Null' role.
        static QubitType get(::mlir::MLIRContext *context) {
            return get(context, quantum::QubitLevel::Abstract, quantum::QubitRole::Null);
        }
    }];

    let genVerifyDecl = 1;
}
def QuregType : Quantum_Type<"Qureg", "reg"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";

    let parameters = (ins
        DefaultValuedParameter<"QubitLevel", "QubitLevel::Abstract">:$level
    );

    let assemblyFormat = "(`<` $level^ `>`)?";

    let builderCall = [{
        $_builder.getType<quantum::QuregType>(quantum::QubitLevel::Abstract)
    }];

    let extraClassDeclaration = [{
        // Get an instance of the QuregType with the default 'Abstract' level.
        static QuregType get(::mlir::MLIRContext *context) {
            return get(context, quantum::QubitLevel::Abstract);
        }
    }];
}
def ObservableType : Quantum_Type<"Observable", "obs"> {
    let summary = "A quantum observable for use in measurements.";
}
def ResultType : Quantum_Type<"Result", "res"> {
    let summary = "A quantum measurement result.";
}

#endif // QUANTUM_TYPES
