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

#ifndef REFQUANTUM_INTERFACES
#define REFQUANTUM_INTERFACES

include "mlir/IR/OpBase.td"

def QuantumOperation : OpInterface<"QuantumOperation"> {
    let description = [{
        A base class for all quantum operations that can be considered actions on wires.
        The actions do not have to be unitary. For example, the SetState operations also falls
        under this class.
    }];

    let cppNamespace = "::catalyst::ref_quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered input wires (including controls).",
            "std::vector<mlir::Value>", "getWireOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered input wires (including controls).",
            "void", "setWireOperands", (ins "mlir::ValueRange":$replacements)
        >
    ];
}

def QuantumGate : OpInterface<"QuantumGate", [QuantumOperation]> {
    let description = [{
        A base class for all unitary quantum operations.
        These operations can be inverted and controlled.
    }];

    let cppNamespace = "::catalyst::ref_quantum";

    let methods = [
        InterfaceMethod<
            "Return operands which are considered non-controlled input wire values.",
            "mlir::ValueRange", "getNonCtrlWireOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered non-controlled input wire values.",
            "void", "setNonCtrlWireOperands", (ins "mlir::ValueRange":$replacements)
        >,
        InterfaceMethod<
            "Return all operands which are considered controlling input wire values.",
            "mlir::ValueRange", "getCtrlWireOperands"
        >,
        InterfaceMethod<
            "Set all operands which are considered controlling input wire values.",
            "void", "setCtrlWireOperands", (ins "mlir::ValueRange":$replacements)
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

        if (gate.getCtrlValueOperands().size() != gate.getCtrlWireOperands().size()) {
            return $_op->emitError() <<
                "number of controlling wires in input (" <<
                gate.getCtrlWireOperands().size() << ") " <<
                "and controlling values (" <<
                gate.getCtrlValueOperands().size() <<
                ") must be the same";
        }

        // STL methods to check duplicates will all complain about `mlir::Value` not having a
        // comparison method defined, since they all use map/set, which is hash-based
        // So we just do it manually
        std::vector<mlir::Value> wireOperands = gate.getWireOperands();
        for (size_t i=0; i < wireOperands.size(); i++) {
            for (size_t j=i+1; j < wireOperands.size(); j++) {
                if (wireOperands[i] == wireOperands[j]) {
                    return $_op->emitError() << "all wires on a quantum gate must be " <<
                    "distinct (including controls)";
                }
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

    let cppNamespace = "::catalyst::ref_quantum";

    let methods = [
        InterfaceMethod<
            "Return all operands which are considered gate parameters.",
            "mlir::ValueRange", "getAllParams"
        >,
        InterfaceMethod<
            "Return the param operand at the requested index.",
            "mlir::Value", "getParam", (ins "size_t":$idx), /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return mlir::cast<ConcreteOp>($_op).getAllParams()[idx]; }]
        >,
    ];
}


#endif // REFQUANTUM_INTERFACES
