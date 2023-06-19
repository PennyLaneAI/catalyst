// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

def QuantumDialect : Dialect {
    let summary = "A quantum computing dialect with value semantics.";
    let description = [{
        The quantum dialect extends core MLIR with the necessary types and operations to form
        the core IR of the Catalyst compiler.

        At a minimum, a representation of quantum programs typically contains qubit management,
        quantum gate operations, and measurements. The chosen representation for this dialect is
        closely aligned with the operations in the [PennyLane](https://docs.pennylane.ai/en/stable/)
        quantum computing framework, but also allows for custom instruction sets.

        In this dialect, quantum operations are expressed using value semantics. That is abstract
        qubit values are consumed and produced by operations, instead of operations merely acting on
        qubit references. This exposes the dataflow of the quantum program in the SSA graph of the
        MLIR. See [QIRO: An SSA-based Quantum IR for Optimization](https://doi.org/10.1145/3491247).
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "quantum";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst::quantum";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Quantum dialect types.
//===----------------------------------------------------------------------===//

class Quantum_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QuantumDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : Quantum_Type<"Qubit", "bit"> {
    let summary = "A value-semantic qubit (state).";
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

#endif // QUANTUM_DIALECT
