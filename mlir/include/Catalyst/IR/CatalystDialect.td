// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CATALYST_DIALECT
#define CATALYST_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Catalyst dialect.
//===----------------------------------------------------------------------===//

def Catalyst_Dialect : Dialect {
    let summary = "A dialect supporting core Catalyst functionality.";
    let description = [{
        The catalyst dialect serves as a standard library for the Catalyst compiler. It
        contains data structures that support core compiler functionality.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "catalyst";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Catalyst operation definition.
//===----------------------------------------------------------------------===//

class Catalyst_Op<string mnemonic, list<Trait> traits = []> :
        Op<Catalyst_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Catalyst dialect types.
//===----------------------------------------------------------------------===//

class Catalyst_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Catalyst_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def ArrayListType : Catalyst_Type<"ArrayList", "arraylist"> {
    let summary = "a dynamically resizable array";
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

#endif // CATALYST_DIALECT
