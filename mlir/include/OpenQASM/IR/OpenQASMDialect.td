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

#ifndef OPENQASM_DIALECT
#define OPENQASM_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OpenQASM dialect definition.
//===----------------------------------------------------------------------===//

def OpenQASM_Dialect : Dialect {
    let name = "openqasm";
    let summary = "A dialect for OpenQASM.";
    let description = [{
        This dialect contains operations to represent quantum circuits using
        OpenQASM (open quantum assembly language).
    }];
    let cppNamespace = "::catalyst::openqasm";
}

//===----------------------------------------------------------------------===//
// Base OpenQASM operation definition.
//===----------------------------------------------------------------------===//

class OpenQASM_Op<string mnemonic, list<Trait> traits = []> :
        Op<OpenQASM_Dialect, mnemonic, traits>;

#endif // OPENQASM_DIALECT
