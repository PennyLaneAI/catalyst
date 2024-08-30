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

#ifndef MITIGATION_DIALECT
#define MITIGATION_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Mitigation dialect definition.
//===----------------------------------------------------------------------===//

def Mitigation_Dialect : Dialect {
    let name = "mitigation";
    let summary = "A dialect for quantum mitigation.";
    let description = [{
        This dialect contains operations to represent mitigation on quantum functions,
        as well as lowering mechanisms to compute the mitigated expectation value.
    }];
    let cppNamespace = "::catalyst::mitigation";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    // let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Mitigation operation definition.
//===----------------------------------------------------------------------===//

class Mitigation_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mitigation_Dialect, mnemonic, traits>;

#endif // MITIGATION_DIALECT
