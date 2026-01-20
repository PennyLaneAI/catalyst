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

#ifndef REFQUANTUM_TYPES
#define REFQUANTUM_TYPES

include "mlir/IR/AttrTypeBase.td"

include "RefQuantum/IR/RefQuantumDialect.td"

//===----------------------------------------------------------------------===//
// RefQuantum dialect types.
//===----------------------------------------------------------------------===//

class RefQuantum_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<RefQuantumDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def AllocationType : RefQuantum_Type<"Allocation", "allocation"> {
    let summary = "An allocation result.";
}

def QubitReferenceType : RefQuantum_Type<"QubitRef", "qubit_ref"> {
    let summary = "A reference to a qubit.";
}

#endif // REFQUANTUM_TYPES
