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

#ifndef QECOP_INTERFACES
#define QECOP_INTERFACES

include "mlir/IR/OpBase.td"

def QECOpInterface : OpInterface<"QECOpInterface"> {

    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered QEC Operations. These are characterized by operating on zero
        or more qubit values, and returning the same amount of qubit values.
    }];

    let cppNamespace = "::catalyst::qec";

    let methods = [
        InterfaceMethod<
            /*desc=*/"Get the input qubits for this operation.",
            /*retTy=*/"mlir::Operation::operand_range",
            /*methodName=*/"getInQubits"
        >,
        InterfaceMethod<
            /*desc=*/"Get the output qubits for this operation.",
            /*retTy=*/"mlir::Operation::result_range",
            /*methodName=*/"getOutQubits"
        >,
        InterfaceMethod<
            /*desc=*/"Get the Pauli product for this operation.",
            /*retTy=*/" ::mlir::ArrayAttr",
            /*methodName=*/"getPauliProduct"
        >,
        InterfaceMethod<
            /*desc=*/"Get the Pauli product for this operation.",
            /*retTy=*/" ::mlir::ArrayAttr",
            /*methodName=*/"getPauliProductAttr"
        >,
        InterfaceMethod<
            /*desc=*/"Get the rotation kind for this operation.",
            /*retTy=*/"uint16_t",
            /*methodName=*/"getRotationKind"
        >,
        InterfaceMethod<
            /*desc=*/"Set the Pauli product for this operation.",
            /*retTy=*/"void",
            /*methodName=*/"setPauliProductAttr", (ins " ::mlir::ArrayAttr":$attr)
        >,
        InterfaceMethod<
            /*desc=*/"Set the rotation kind for this operation.",
            /*retTy=*/"void",
            /*methodName=*/"setRotationKind", (ins "uint16_t":$attrValue)
        >
    ];


}

#endif // QECOP_INTERFACES
