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

#ifndef REF_QUANTUM_DIALECT
#define REF_QUANTUM_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// RefQuantum dialect definition.
//===----------------------------------------------------------------------===//

def RefQuantumDialect : Dialect {
    let summary = "Reference semantics quantum dialect.";
    let description = [{
        Here are words
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "ref_quantum";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::ref_quantum";

    // TODO: does it depend?
    // let dependentDialects = [
    //     "quantum::QuantumDialect"
    // ];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    // Do I even need attributes?
    // The only attr in regular quantum dialect is the named observable.
    // I probably also need it....
    let useDefaultAttributePrinterParser = 1;

    /// Uncomment the line below if defining types for the RefQuantum dialect
    // let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// RefQuantum dialect types.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the RefQuantum dialect
// class RefQuantum_Type<string name, string typeMnemonic, list<Trait> traits = []>
//         : TypeDef<RefQuantumDialect, name, traits> {
//     let mnemonic = typeMnemonic;
// }


//===----------------------------------------------------------------------===//
// RefQuantum dialect base operation.
//===----------------------------------------------------------------------===//

class RefQuantum_Op<string mnemonic, list<Trait> traits = []> :
        Op<RefQuantumDialect, mnemonic, traits>;


#endif // REF_QUANTUM_DIALECT
