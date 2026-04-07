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

#ifndef MBQC_DIALECT
#define MBQC_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MBQC dialect definition.
//===----------------------------------------------------------------------===//

def MBQC_Dialect : Dialect {
    let summary = "A dialect for measurement-based quantum computing.";
    let description = [{
        The measurement-based quantum computing (MBQC) dialect extends the Quantum dialect with
        operations and passes that are useful for expressing quantum workloads in the MBQC
        representation.

        See for example Raussendorf et al., Measurement-based quantum computation with cluster
        states, Phys. Rev. A 68, 022312 (2003),
        arXiv:[quant-ph/0301052](https://arxiv.org/abs/quant-ph/0301052)

        > [!IMPORTANT]
        > The MBQC dialect is experimental and and will not maintain API stability between releases.
        > Use at your own risk.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "mbqc";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::mbqc";

    let dependentDialects = [
        "quantum::QuantumDialect"
    ];

    /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
    let useDefaultAttributePrinterParser = 1;
    /// Uncomment the line below if defining types for the MBQC dialect
    // let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// MBQC dialect types.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the MBQC dialect
// class MBQC_Type<string name, string typeMnemonic, list<Trait> traits = []>
//         : TypeDef<MBQC_Dialect, name, traits> {
//     let mnemonic = typeMnemonic;
// }


//===----------------------------------------------------------------------===//
// MBQC dialect base operation.
//===----------------------------------------------------------------------===//

class MBQC_Op<string mnemonic, list<Trait> traits = []> :
        Op<MBQC_Dialect, mnemonic, traits>;


#endif // MBQC_DIALECT
