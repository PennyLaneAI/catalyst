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

#ifndef QUANTUM_INTERFACES
#define QUANTUM_INTERFACES

include "mlir/IR/OpBase.td"

def QuantumRegion : OpInterface<"QuantumRegion"> {
    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered quantum regions. These are characterized by operating on a single
        qubit register, and returning a new register value.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return the quantum register operand.",
            "mlir::Value", "getRegisterOperand", (ins), /*methodBody=*/[{
                return $_op->getOperand(0);
            }]
        >,
        InterfaceMethod<
            "Return the quantum register result.",
            "mlir::Value", "getRegisterResult", (ins), /*methodBody=*/[{
                return $_op->getResult(0);
            }]
        >,
    ];

    let verify = [{

        if ($_op->getNumOperands() != 1)
            return $_op->emitOpError("must have exactly one operand (quantum register)");

        if ($_op->getNumResults() != 1)
            return $_op->emitOpError("must have exactly one result (quantum register)");

        return mlir::success();
    }];
}

def QuantumOperation : OpInterface<"QuantumOperation"> {
    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered quantum logic gates. These are characterized by operating on zero
        or more qubit values, and returning the same amount of qubit values.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered input qubit values (including controls).",
            "std::vector<mlir::Value>", "getQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered input qubit values (including controls).",
            "void", "setQubitOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return all results which are considered output qubit values (including controls).",
            "std::vector<mlir::Value>", "getQubitResults"
        >
    ];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        if (gate.getQubitOperands().size() != gate.getQubitResults().size())
            return $_op->emitError() <<
                "number of qubits in input (" << gate.getQubitOperands().size() << ") " <<
                "and output (" << gate.getQubitResults().size() << ") must be the same";

        return mlir::success();
    }];
}

def QuantumGate : OpInterface<"QuantumGate", [QuantumOperation]> {
    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered 'unitary' gates, which in addition to the properties of the
        "QuantumOperation" can be inverted and controlled.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return operands which are considered non-controlled input qubit values.",
            "mlir::ValueRange", "getNonCtrlQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered non-controlled input qubit values.",
            "void", "setNonCtrlQubitOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return results which are considered non-controlled output qubit values.",
            "mlir::ValueRange", "getNonCtrlQubitResults"
        >,
        InterfaceMethod<
            "Return all operands which are considered controlling input qubit values.",
            "mlir::ValueRange", "getCtrlQubitOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered controlling input qubit values.",
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
            "Return all operands which are considered controlling output qubit values.",
            "mlir::ValueRange", "getCtrlQubitResults"
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

        if (gate.getCtrlValueOperands().size() != gate.getCtrlQubitOperands().size())
            return $_op->emitError() <<
                "number of controlling qubits in input (" <<
                gate.getCtrlQubitOperands().size() << ") " <<
                "and controlling values (" <<
                gate.getCtrlValueOperands().size() <<
                ") must be the same";

        if (gate.getCtrlQubitOperands().size() != gate.getCtrlQubitResults().size())
            return $_op->emitError() <<
                "number of controlling qubits in input (" <<
                gate.getCtrlQubitOperands().size() << ") " <<
                "and output (" <<
                gate.getCtrlQubitResults().size() <<
                ") must be the same";

        return mlir::success();
    }];
}

def StaticGate : OpInterface<"StaticGate", [QuantumGate]> {
    let description = [{
        This interface provides a generic way to interact with quantum
        instructions with static parameters (known at runtime). These parameters
        are specified by a set of constant literals in the form of an array attribute.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered gate parameters.",
            "mlir::DenseF64ArrayAttr", "getAllParams"
        >,
    ];
}

def ParametrizedGate : OpInterface<"ParametrizedGate", [QuantumGate]> {
    let description = [{
        This interface provides a generic way to interact with parametrized
        quantum instructions. These are quantum operations with arbitrary
        classical gate parameters.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered gate parameters.",
            "std::variant<mlir::ValueRange, mlir::SmallVector<double>>", "getAllParams"
        >,
        InterfaceMethod<
            "Return all gate parameters as ssa values no matter dynamic or static",
            "mlir::ValueRange", "getSSAParams"
        >,
        InterfaceMethod<
            "Return the starting index at which to find gate parameter operands.",
            "size_t", "getParamOperandIdx", (ins), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return 0; }]
        >,
    ];
}

def DifferentiableGate : OpInterface<"DifferentiableGate", [ParametrizedGate]> {
    let description = [{
        This interface provides a generic way to interact with differentiable
        quantum instructions. These are quantum operations with differentiable
        gate parameters (must be scalar F64 values).
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered differentiable gate parameters.",
            "mlir::ValueRange", "getDiffParams", (ins), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                return $_op.getSSAParams();
            }]
        >,
        InterfaceMethod<
            "Return the starting index at which to find differentiable operands in the Operation*."
            "Differentiable gate parameter operands do not need to be stored in a single ODS "
            "argument or be located in a particular position, but are assumed to be contiguous.",
            "size_t", "getDiffOperandIdx", (ins), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
               return $_op.getParamOperandIdx();
            }]
        >,
    ];
}

def MeasurementProcess : OpInterface<"MeasurementProcess"> {
    let description = [{
        This interface provides a generic way to interact with quantum measurement processes.
        These are instructions that represent some information extraction procedure on the
        underlying statevector. On simulators, they may be implemented by direct manipulation
        of the statevector, but on hardware they typically turn into procedures over the quantum
        program as a whole, usually requiring many executions of the same circuit.
    }];

    let cppNamespace = "::catalyst::quantum";
}

#endif // QUANTUM_INTERFACES
