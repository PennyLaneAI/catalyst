// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QREF_INTERFACES
#define QREF_INTERFACES

include "mlir/IR/OpBase.td"

def QuantumRegion : OpInterface<"QuantumRegion"> {
    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered quantum regions. These are characterized by operating on a single
        qubit register reference.
    }];

    let cppNamespace = "::catalyst::qref";

    let methods = [
        InterfaceMethod<
            "Return the quantum register operand.",
            "mlir::Value", "getRegisterOperand", (ins), /*methodBody=*/[{
                return $_op->getOperand(0);
            }]
        >
    ];

    let verify = [{

        if ($_op->getNumOperands() != 1) {
            return $_op->emitOpError("must have exactly one operand (a quantum register)");
        }

        return mlir::success();
    }];
}

def QuantumOperation : OpInterface<"QuantumOperation"> {
    let description = [{
        A base class for all quantum operations that can be considered actions on qubits.
        The actions do not have to be unitary. For example, the SetState operations also falls
        under this class.
    }];

    let cppNamespace = "::catalyst::qref";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered input qubits (including controls).",
            "std::vector<mlir::Value>", "getQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered input qubits (including controls).",
            "void", "setQubitOperands", (ins "mlir::ValueRange":$replacements)
        >
    ];
}

def QuantumGate : OpInterface<"QuantumGate", [QuantumOperation]> {
    let description = [{
        A base class for all unitary quantum operations.
        These operations can be inverted and controlled.
    }];

    let cppNamespace = "::catalyst::qref";

    let methods = [
        InterfaceMethod<
            "Return operands which are considered non-controlled input qubit reference values.",
            "mlir::ValueRange", "getNonCtrlQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered non-controlled input qubit reference values.",
            "void", "setNonCtrlQubitOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return all operands which are considered controlling input qubit reference values.",
            "mlir::ValueRange", "getCtrlQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered controlling input qubit reference values.",
            "void", "setCtrlQubitOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return all operands which are considered controlling input boolean values.",
            "mlir::ValueRange", "getCtrlValueOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered controlling input boolean values.",
            "void", "setCtrlValueOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return adjoint flag.",
            "bool", "getAdjointFlag"
        >,
        InterfaceMethod<
            "Set adjoint flag.",
            "void", "setAdjointFlag", (ins "bool":$adjoint)
        >
    ];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        if (gate.getCtrlValueOperands().size() != gate.getCtrlQubitOperands().size()) {
            return $_op->emitError() <<
                "number of controlling qubits in input (" <<
                gate.getCtrlQubitOperands().size() << ") " <<
                "and controlling values (" <<
                gate.getCtrlValueOperands().size() <<
                ") must be the same";
        }

        llvm::SmallPtrSet<mlir::Value, 4> seenQubits;
        for (mlir::Value qubit : gate.getQubitOperands()) {
            if (!seenQubits.insert(qubit).second) {
                return $_op->emitError() << "all qubits on a quantum gate must be " <<
                "distinct (including controls)";
            }
        }

        return mlir::success();
    }];
}

def ParametrizedGate : OpInterface<"ParametrizedGate", [QuantumGate]> {
    let description = [{
        This interface provides a generic way to interact with parametrized
        quantum instructions. These are quantum operations with arbitrary
        classical gate parameters.
    }];

    let cppNamespace = "::catalyst::qref";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered gate parameters.",
            "mlir::ValueRange", "getAllParams"
        >,
        InterfaceMethod<
            "Return the param operand at the requested index.",
            "mlir::Value", "getParam", (ins "size_t":$idx), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                mlir::ValueRange allParams = mlir::cast<ConcreteOp>($_op).getAllParams();
                assert(idx < allParams.size() && "parameter index out of range");
                return allParams[idx];
            }]
        >,
        InterfaceMethod<
            "Return the starting index at which to find gate parameter operands.",
            "size_t", "getParamOperandIdx", (ins), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return 0; }]
        >,
    ];
}


#endif // QREF_INTERFACES
