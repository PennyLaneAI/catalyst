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

#ifndef QEC_DIALECT
#define QEC_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// QEC dialect.
//===----------------------------------------------------------------------===//

def QECDialect : Dialect {
    let summary = "A dialect for quantum error correction circuits.";
    let description = [{
        The QEC dialect extends the Quantum dialect with operations and passes that may be useful
        in quantum error correction schemes, such as the Pauli Product Measurement framework.

        See also [A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery
        ](https://doi.org/10.22331/q-2019-03-05-128).
    }];

    let dependentDialects = [
        "catalyst::quantum::QuantumDialect"
    ];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qec";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst::qec";

    /// Use the default attribute printer/parser hooks, otherwise we would explicitly define them.
    let useDefaultAttributePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// QEC dialect enums.
//===----------------------------------------------------------------------===//

def LogicalInitKind : I32EnumAttr<"LogicalInitKind",
    "The initial state of a logical qubit such as |0⟩, |1⟩, |+⟩, |−⟩, |Y⟩, |-Y⟩, |m⟩, or |m̅⟩.",
    [
        I32EnumAttrCase<"zero",     0>,     // |0⟩ Non-magic state
        I32EnumAttrCase<"one",      1>,     // |1⟩ Non-magic state
        I32EnumAttrCase<"plus",     2>,     // |+⟩  = (|0⟩ + |1⟩) / sqrt(2)  Non-magic state
        I32EnumAttrCase<"minus",    3>,     // |-⟩  = (|0⟩ - |1⟩) / sqrt(2)  Non-magic state
        I32EnumAttrCase<"plus_i",   4>,     // |Y⟩  = (|0⟩ + i|1⟩) / sqrt(2) Non-magic / Magic state
        I32EnumAttrCase<"minus_i",  5>,     // |-Y⟩ = (|0⟩ - i|1⟩) / sqrt(2) Non-magic / Magic state
        I32EnumAttrCase<"magic",    6>,     // |m⟩  = |0⟩ + e^{iπ/4}|1⟩      Magic state
        I32EnumAttrCase<"magic_conj", 7>,   // |m̅⟩  = |0⟩ + e^{-iπ/4}|1⟩     Magic state
    ]> {
    let cppNamespace = "::catalyst::qec";
    let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// QEC dialect attributes.
//===----------------------------------------------------------------------===//

def PauliWord : TypedArrayAttrBase<StrAttr, "A product of Pauli operators, aka a Pauli word.">;
def LogicalInit : EnumAttr<QECDialect, LogicalInitKind, "enum">;

#endif // QEC_DIALECT
