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

#ifndef PBCOP_INTERFACES
#define PBCOP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PBC Operation Interface
//===----------------------------------------------------------------------===//

def PBCOpInterface : OpInterface<"PBCOpInterface"> {

    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered PBC Operations. These are characterized by operating on one or more qubit values,
        and returning the same amount of qubit values.
    }];

    let cppNamespace = "::catalyst::pbc";

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
            /*retTy=*/"::mlir::ArrayAttr",
            /*methodName=*/"getPauliProduct"
        >,
        InterfaceMethod<
            /*desc=*/"Set the Pauli product for this operation.",
            /*retTy=*/"void",
            /*methodName=*/"setPauliProductAttr", (ins "const ::mlir::ArrayAttr&":$propValue)
        >
    ];

}

def DecomposableGate : OpInterface<"DecomposableGate"> {
    let description = [{
        This interface provides a generic way to query decomposition data from a quantum operation.
        An operation must implement this interface in order to compatible with the `graph-decomposition` pass.
        Note that implementing this interface does not guarantee that an operation will decomposed successfully, only that it will be registered with the graph.
    }];

    let cppNamespace = "::catalyst::pbc";

    let methods = [
        InterfaceMethod<
            "Return the name of the corresponding frontend operator (i.e. the rule supplier).", "std::string", "getOperatorName"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to shape and dtype of dynamic data.",
            "llvm::StringMap<llvm::SmallVector<mlir::Type>>", "getDynamicShape"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to number of wires (excluding control wires).",
            "llvm::StringMap<size_t>", "getWireLens"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to static data values.",
            "mlir::DictionaryAttr", "getStaticData"
        >,
        InterfaceMethod<
            "Return a string representation of any additional data an operator may need to provide"
            " for uniqueness (ex. UID), or emptystring if not applicable.",
            "std::string", "getExtraData", (ins), [{}], [{ return ""; }]
        >,
        InterfaceMethod<
            "Return the operation's ID for use with the graph. "
            "This should be computed by the following convention:\n"
            "```\n"
            "name + {map params to list of types} + {map wire args to number of wires} + {static data} + [extra data]\n"
            "```\n"
            "Note that extra data is currently specific to `OperatorOp`s, and should only appear when present."
            "Each of these maps should be sorted lexicographically by key."
            "In general, this should be computed by the default here and not overridden.",
            "std::string", "getGraphOpId", (ins), [{}], [{ return ::catalyst::pbc::defaultGetGraphOpId($_op.getOperation()); }]
        >
    ];
}

#endif // PBCOP_INTERFACES
