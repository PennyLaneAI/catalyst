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

#ifndef GRADIENT_DIALECT
#define GRADIENT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Gradient dialect definition.
//===----------------------------------------------------------------------===//

def Gradient_Dialect : Dialect {
    let name = "gradient";
    let summary = "A dialect for classical & quantum gradients.";
    let description = [{
        This dialect contains operations to represent gradients on classical
        and quantum functions, as well as lowering mechanisms to compute the
        gradients.
    }];
    let cppNamespace = "::catalyst::gradient";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Gradient operation definition.
//===----------------------------------------------------------------------===//

class Gradient_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gradient_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Gradient dialect types.
//===----------------------------------------------------------------------===//

class Gradient_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Gradient_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def ParameterVectorType : Gradient_Type<"ParameterVector", "vec"> {
    let summary = "A dynamically resizable array.";
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

#endif // GRADIENT_DIALECT
