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

#ifndef QUANTUM_ATTR_DEFS
#define QUANTUM_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Quantum dialect enums and attributes.
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
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

def NamedObservableAttr : EnumAttr<QuantumDialect, NamedObservable, "named_observable">;

def QubitLevel : I32EnumAttr<"QubitLevel",
    "Qubit levels in the hierarchical qubit representation",
    [
        I32EnumAttrCase<"Abstract", 0, "abstract">,
        I32EnumAttrCase<"Logical",  1, "logical">,
        I32EnumAttrCase<"QEC",      2, "qec">,
        I32EnumAttrCase<"Physical", 3, "physical">,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

def QubitRole : I32EnumAttr<"QubitRole",
    "Qubit roles for further specialization in the hierarchical qubit representation",
    [
        I32EnumAttrCase<"Null",   0, "null">,
        I32EnumAttrCase<"Data",   1, "data">,
        I32EnumAttrCase<"XCheck", 2, "xcheck">,
        I32EnumAttrCase<"ZCheck", 3, "zcheck">,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

#endif // QUANTUM_ATTR_DEFS
