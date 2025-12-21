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

#ifndef REF_QUANTUM_OPS
#define REF_QUANTUM_OPS

include "mlir/IR/EnumAttr.td"

// TODO: does it depend?
// include "Quantum/IR/QuantumDialect.td"
include "RefQuantum/IR/RefQuantumDialect.td"

//===----------------------------------------------------------------------===//
// RefQuantum dialect enums.
//===----------------------------------------------------------------------===//

def NamedObservable : I32EnumAttr<"NamedObservable",
    "Known named observables",
    [
        I32EnumAttrCase<"Identity", 0>,
        I32EnumAttrCase<"PauliX",   1>,
        I32EnumAttrCase<"PauliY",   2>,
        I32EnumAttrCase<"PauliZ",   3>,
        I32EnumAttrCase<"Hadamard", 4>,
    ]> {
    let cppNamespace = "catalyst::ref_quantum";
    let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// RefQuantum dialect traits.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// RefQuantum dialect attributes.
//===----------------------------------------------------------------------===//

def NamedObservableAttr : EnumAttr<RefQuantum_Dialect, NamedObservable, "named_observable">;

//===----------------------------------------------------------------------===//
// RefQuantum dialect operations.
//===----------------------------------------------------------------------===//


#endif // REF_QUANTUM_OPS
