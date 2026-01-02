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

#ifndef REF_QUANTUM_OPS
#define REF_QUANTUM_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
// TODO: I probably need bufferization. Remove if ended up not using it.
// Or do I? If this dialect is only supposed to be at a high level,
// i.e. connection to lower parts of the pipeline are done via
//   ref dialect ---> value dialect ---> bufferization ---> ....
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"

// TODO: does it depend?
// Ideally, I want the two dialects to be cleanly separated.
include "Quantum/IR/QuantumDialect.td"
include "Quantum/IR/QuantumEnums.td"
include "RefQuantum/IR/RefQuantumDialect.td"
include "RefQuantum/IR/RefQuantumInterfaces.td"

//===----------------------------------------------------------------------===//
// RefQuantum dialect operations.
//===----------------------------------------------------------------------===//

// -----

class Memory_Op<string mnemonic, list<Trait> traits = []> : RefQuantum_Op<mnemonic, traits>;

// TODO: (dynamic) alloc and deallocs
// Note that extracts and inserts are not a thing in reference semantics

// -----

class Gate_Op<string mnemonic, list<Trait> traits = []> :
        RefQuantum_Op<mnemonic, traits # [QuantumOperation]> {

    code extraBaseClassDeclaration = [{
        std::vector<mlir::Value> getWireOperands() {
            std::vector<mlir::Value> values;
            values.insert(values.end(), getWires().begin(), getWires().end());
            return values;
        }

        void setWireOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange wires = getWiresMutable();
            assert(wires.size() == replacements.size() && "must provide values for all wires");
            wires.assign(replacements);
        }
    }];

    let extraClassDeclaration = extraBaseClassDeclaration;
}

class UnitaryGate_Op<string mnemonic, list<Trait> traits = []> :
        Gate_Op<mnemonic, traits # [QuantumGate, Unitary]> {

    code extraBaseClassDeclaration = [{
        std::vector<mlir::Value> getWireOperands() {
            std::vector<mlir::Value> values;
            values.insert(values.end(), getWires().begin(), getWires().end());
            values.insert(values.end(), getCtrlWires().begin(), getCtrlWires().end());
            return values;
        }

        void setWireOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange wires = getWiresMutable();
            mlir::MutableOperandRange ctrls = getCtrlWiresMutable();
            assert(wires.size() + ctrls.size() == replacements.size() &&
                   "must provide values for all wires (including controls)");

            wires.assign(replacements.take_front(wires.size()));
            ctrls.assign(replacements.take_back(ctrls.size()));
        }

        mlir::ValueRange getNonCtrlWireOperands() {
            return getWires();
        }

        void setNonCtrlWireOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange wires = getWiresMutable();
            assert(wires.size() == replacements.size() &&
                   "must provide values for all non-ctrl wire values");
            wires.assign(replacements);
        }

        mlir::ValueRange getCtrlWireOperands() {
            return getCtrlWires();
        }

        void setCtrlWireOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange ctrlWires = getCtrlWiresMutable();
            assert(ctrlWires.size() == replacements.size() &&
                   "must provide values for all ctrl wire values");
            ctrlWires.assign(replacements);
        }

        mlir::ValueRange getCtrlValueOperands() {
            return getCtrlValues();
        }

        void setCtrlValueOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange ctrlValues = getCtrlValuesMutable();
            assert(ctrlValues.size() == replacements.size() &&
                   "must provide values for all control values");
            ctrlValues.assign(replacements);
        }

        bool getAdjointFlag() {
            return getAdjoint();
        }

        void setAdjointFlag(bool adjoint) {
            if (adjoint) {
               (*this)->setAttr("adjoint", mlir::UnitAttr::get(this->getContext()));
            } else {
               (*this)->removeAttr("adjoint");
            }
        };
    }];

    let extraClassDeclaration = extraBaseClassDeclaration;
}

def CustomOp : UnitaryGate_Op<"custom", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "A generic quantum gate on n qubits with m floating point parameters.";
    let description = [{
    }];

    let arguments = (ins
        Variadic<F64>:$params,
        Variadic<I64>:$wires,  // Perhaps add a static version? But not super high priority on my list. We have arith.constant.
        StrAttr:$gate_name,
        UnitAttr:$adjoint,
        Variadic<I64>:$ctrl_wires,  // Ditto regarding static.
        Variadic<I1>:$ctrl_values
    );

    // TODO: add convenience builders

    let assemblyFormat = [{
        $gate_name `(` $params `)` $wires (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($wires) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getParams();
        }
    }];

    // TODO: anything needed here?
    // let hasCanonicalizeMethod = 1;
    // let hasVerifier = 1;
}

// // -----

// // Observable ops are not meaningful on their own: their purpose is to be sent into a measurement
// // Hence they are Pure, i.e. removable if no users

class Observable_Op<string mnemonic, list<Trait> traits = []> :
        RefQuantum_Op<mnemonic, traits # [Pure]>;

def NamedObsOp : Observable_Op<"namedobs"> {
    let summary = "Define a Named observable for use in measurements";
    let description = [{
        The `ref_quantum.namedobs` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents one of 5 named observables
        {Identity, PauliX, PauliY, PauliZ, Hadamard} on a qubit. The arguments are a wire to
        measure as well as an encoding operator for the qubit as an integer between 0-4.
    }];

    let arguments = (ins
        I64:$wire
        //NamedObservableAttr:$type
    );

    let results = (outs
        ObservableType:$obs
    );

    // let assemblyFormat = [{
    //     $wire `[` $type `]` attr-dict  `:` type(results)
    // }];
    let assemblyFormat = [{
        $wire attr-dict `:` type(results)
    }];
}

// -----

// class Measurement_Op<string mnemonic, list<Trait> traits = []> :
//         Quantum_Op<mnemonic, traits # [MeasurementProcess]>;

#endif // REF_QUANTUM_OPS
