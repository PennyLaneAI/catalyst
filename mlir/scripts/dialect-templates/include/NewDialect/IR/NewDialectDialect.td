// Copyright @@@year@@@ Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef @@@NEW_DIALECT@@@_DIALECT
#define @@@NEW_DIALECT@@@_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// @@@NewDialect@@@ dialect definition.
//===----------------------------------------------------------------------===//

def @@@NewDialect@@@_Dialect : Dialect {
    let summary = "Short summary of the dialect.";
    let description = [{
        Full description of the dialect.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "@@@new_dialect@@@";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::@@@new_dialect@@@";

    /// This dialect depends on the following list of other dialects.
    let dependentDialects = [
        // "quantum::QuantumDialect"
    ];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    let useDefaultAttributePrinterParser = 1;
    /// Uncomment the line below if defining types for the @@@NewDialect@@@ dialect
    // let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// @@@NewDialect@@@ dialect types.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the @@@NewDialect@@@ dialect
// class @@@NewDialect@@@_Type<string name, string typeMnemonic, list<Trait> traits = []>
//         : TypeDef<@@@NewDialect@@@_Dialect, name, traits> {
//     let mnemonic = typeMnemonic;
// }


//===----------------------------------------------------------------------===//
// @@@NewDialect@@@ dialect base operation.
//===----------------------------------------------------------------------===//

class @@@NewDialect@@@_Op<string mnemonic, list<Trait> traits = []> :
        Op<@@@NewDialect@@@_Dialect, mnemonic, traits>;


#endif // @@@NEW_DIALECT@@@_DIALECT
