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

// include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
// include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

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

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    // let useDefaultTypePrinterParser = 1;
    // let useDefaultAttributePrinterParser = 1;

}

//===----------------------------------------------------------------------===//
// QEC dialect enums.
//===----------------------------------------------------------------------===//

// def PauliEnum : I32EnumAttr<"Pauli",
//     "Kinds of Pauli operators",
//     [
//         I32EnumAttrCase<"I", 0>,
//         I32EnumAttrCase<"X", 1>,
//         I32EnumAttrCase<"Y", 2>,
//         I32EnumAttrCase<"Z", 3>,
//     ]> {
//     let cppNamespace = "catalyst::qec";
//     let genSpecializedAttr = 0;
// }

//===----------------------------------------------------------------------===//
// QEC dialect attributes.
//===----------------------------------------------------------------------===//

// def PauliLetter : EnumAttr<QECDialect, PauliEnum, "Single Pauli operator.">;
def PauliWord : TypedArrayAttrBase<StrAttr, "A product of Pauli operators, aka a Pauli word.">;


//===----------------------------------------------------------------------===//
// QEC dialect operations.
//===----------------------------------------------------------------------===//

class QEC_Op<string mnemonic, list<Trait> traits = []> : Op<QECDialect, mnemonic, traits>;

def PPRotationOp : QEC_Op<"ppr"> {
    let summary = "Pauli Product Rotation on qubits.";

    let arguments = (ins
        PauliWord:$pauli_product,
        F64:$theta,
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
      $pauli_product `(` $theta `)` $in_qubits attr-dict `:` type($out_qubits)
    }];
}

def PPMeasurementOp : QEC_Op<"ppm"> {
    let summary = "Pauli Product Measurement on qubits.";

    let arguments = (ins
        PauliWord:$pauli_product,
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        I1:$mres,
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
      $pauli_product $in_qubits attr-dict `:` type($out_qubits)
    }];
}

#endif // QEC_DIALECT
