// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ION_DIALECT
#define ION_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Ion dialect.
//===----------------------------------------------------------------------===//

def Ion_Dialect : Dialect {
    let summary = "A trapped ions dialect with value semantics.";
    let description = [{
        The ion dialect extends core MLIR with the necessary types and operations to form
        the IR for trapped ions quantum computers.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "ion";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst::ion";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Ion dialect types.
//===----------------------------------------------------------------------===//

class Ion_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Ion_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def IonType : Ion_Type<"Ion", "ion"> {
    let summary = "A value-semantic ion.";
}

//===----------------------------------------------------------------------===//
// Ion dialect base operation.
//===----------------------------------------------------------------------===//

class Ion_Op<string mnemonic, list<Trait> traits = []> :
        Op<Ion_Dialect, mnemonic, traits>;

#endif // ION_DIALECT
