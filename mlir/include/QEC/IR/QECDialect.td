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
include "mlir/IR/OpBase.td"

include "Quantum/IR/QuantumDialect.td"
include "QEC/IR/QECOpInterfaces.td"

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

}

//===----------------------------------------------------------------------===//
// QEC dialect attributes.
//===----------------------------------------------------------------------===//

def PauliWord : TypedArrayAttrBase<StrAttr, "A product of Pauli operators, aka a Pauli word.">;


//===----------------------------------------------------------------------===//
// QEC dialect operations.
//===----------------------------------------------------------------------===//

class QEC_Op<string mnemonic, list<Trait> traits = []> : Op<QECDialect, mnemonic, traits>;

def PPRotationOp : QEC_Op<"ppr", [QECOpInterface]> {
    let summary = "Pauli Product Rotation on qubits.";

    let description = [{
        The PPRotationOp represents a Pauli product rotation operation on a set of qubits.
        It applies a rotation of the form exp(iθP) where:
        - P is a Pauli product (specified by pauli_product)
        - θ is the rotation angle (specified by rotation_kind in fractions of π)
        
        The operation is characterized by:
        1. A Pauli product (e.g., "XIZ") specifying which Pauli operators to apply
        2. A rotation kind (in fractions of π) specifying the angle
        3. A list of input qubits to apply the rotation to
        
        The operation returns the same number of qubits as input.
        
        Example:
        ```mlir
        %result = qec.ppr "XIZ"(4) %q0, %q1, %q2 : (qreg, qreg, qreg) -> (qreg, qreg, qreg)
        ```
        This applies exp(iπ/2 * X⊗I⊗Z) to the three qubits.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product to apply (e.g., "XIZ")
        I16Attr:$rotation_kind,    // Rotation angle in fractions of π (e.g., 4 for π/2)
        Variadic<QubitType>:$in_qubits  // The qubits to apply the rotation to
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits  // The rotated qubits
    );

    let assemblyFormat = [{
      $pauli_product `(` $rotation_kind `)` $in_qubits attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;

    code extraBaseClassDeclaration = [{
        bool hasRotationKind(){
            return true;
        };
        bool isNonClifford(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 8 || rotationKind == -8;  // π/4 rotations are non-Clifford
        };
        bool isClifford(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 4 || rotationKind == -4;  // π/2 rotations are Clifford
        };
    }];
    let extraClassDeclaration = extraBaseClassDeclaration;
}

def PPMeasurementOp : QEC_Op<"ppm", [QECOpInterface]> {
    let summary = "Pauli Product Measurement on qubits.";

    let description = [{
        The PPMeasurementOp represents a Pauli product measurement operation.
        It measures a set of qubits in the basis specified by a Pauli product.
        
        The operation is characterized by:
        1. A Pauli product (e.g., "XIZ") specifying the measurement basis
        2. A list of input qubits to measure
        
        The operation returns:
        1. A measurement result (1-bit classical value)
        2. The post-measurement state of the qubits
        
        Example:
        ```mlir
        %result, %q0, %q1, %q2 = qec.ppm "XIZ" %q0, %q1, %q2 : (qreg, qreg, qreg) -> (i1, qreg, qreg, qreg)
        ```
        This measures the three qubits in the X⊗I⊗Z basis.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product specifying the measurement basis
        Variadic<QubitType>:$in_qubits  // The qubits to measure
    );

    let results = (outs
        I1:$mres,  // The measurement result (0 or 1)
        Variadic<QubitType>:$out_qubits  // The post-measurement state of the qubits
    );

    let assemblyFormat = [{
      $pauli_product $in_qubits attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;

    code extraBaseClassDeclaration = [{
        bool hasRotationKind(){
            return false;  // Measurements don't have rotation angles
        };
        uint16_t getRotationKind(){
            return 0;  // Default value for measurements
        };
    }];
    let extraClassDeclaration = extraBaseClassDeclaration;
}

#endif // QEC_DIALECT
