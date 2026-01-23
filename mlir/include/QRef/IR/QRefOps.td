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

#ifndef QREF_OPS
#define QREF_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
// TODO: I probably need bufferization. Remove if ended up not using it.
// Or do I? If this dialect is only supposed to be at a high level,
// i.e. connection to lower parts of the pipeline are done via
//   ref dialect ---> value dialect ---> bufferization ---> ....
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"

include "Quantum/IR/QuantumAttrDefs.td"
include "Quantum/IR/QuantumDialect.td"
include "Quantum/IR/QuantumTypes.td"
include "QRef/IR/QRefDialect.td"
include "QRef/IR/QRefInterfaces.td"
include "QRef/IR/QRefTypes.td"

//===----------------------------------------------------------------------===//
// QRef dialect operations.
//===----------------------------------------------------------------------===//

// -----

class Memory_Op<string mnemonic, list<Trait> traits = []> : QRef_Op<mnemonic, traits>;

def AllocOp : Memory_Op<"alloc"> {
    let summary = "Allocate n qubits.";
    let description = [{
        This operation represents the allocations of some number of qubits.

        To distinguish from the quantum registers in the value semantics quantum dialect, which are
        lowered to arrays of qubit IDs in runtime, the result of this operation in reference
        semantics is an SSA value of `Allocation` type, whose entire purpose is to serve as an
        operand to the `MakeReference` operation to produce a qubit reference value.
    }];

    let arguments = (ins
        Optional<I64>:$nqubits,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$nqubits_attr
    );

    let results = (outs
        AllocationType:$allocation
    );

    let assemblyFormat = [{
        `(` ($nqubits^):($nqubits_attr)? `)` attr-dict `:` type(results)
    }];

    // TODO
    // let hasCanonicalizeMethod = 1;
}

def DeallocOp : Memory_Op<"dealloc"> {
    let summary = "Deallocate an allocation.";
    let description = [{
    }];

    let arguments = (ins
        AllocationType:$allocation
    );

    let assemblyFormat = [{
        $allocation attr-dict `:` type(operands)
    }];

    // TODO
    // let hasCanonicalizeMethod = 1;
}

def MakeReferenceOp : Memory_Op<"make_reference", [NoMemoryEffect]> {
    let summary = "Produce a qubit reference SSA value from an allocation and an index.";
    let description = [{
    }];

    let arguments = (ins
        AllocationType:$allocation,
        Optional<I64>:$idx,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        QRefQubitType:$wire
    );

    let assemblyFormat = [{
        $allocation `,` ($idx^):($idx_attr)? attr-dict `:` type(operands) `->` type(results)
    }];

    // TODO
    // let hasCanonicalizeMethod = 1;
    // let hasVerifier = 1;
    // let hasFolder = 1;
}

// -----

class Gate_Op<string mnemonic, list<Trait> traits = []> :
        QRef_Op<mnemonic, traits # [QuantumOperation]> {

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

def SetStateOp : Gate_Op<"set_state"> {
    let summary = "Set state to a complex vector.";
    let description = [{
        This operation is useful for simulators implementing state preparation.
        Instead of decomposing state preparation into multiple operations, this
        operation shortcuts all of that into a single operation.

        .. note::
            This op is not bufferizable at the moment, and must take in the state as a tensor
            instead of a memref. To execute reference semantics quantum dialect, please convert
            to the value semantics quantum dialect, where ops are bufferizable and lowerable to
            LLVM IR.
    }];

    let arguments = (ins
        1DTensorOf<[Complex<F64>]>:$in_state,
        Variadic<QRefQubitType>:$wires
    );

    let assemblyFormat = [{
        `(` $in_state `)` $wires attr-dict `:` type(operands)
    }];

}


def SetBasisStateOp : Gate_Op<"set_basis_state"> {
    let summary = "Set basis state.";
    let description = [{
        This operation is useful for simulators implementing set basis state.
        Instead of decomposing basis state into multiple operations, this
        operation shortcuts all of that into a single operation.
        This signature matches the one in pennylane-lightning which expects
        only a single integer as opposed to a binary digit.

        .. note::
            This op is not bufferizable at the moment, and must take in the basis state as a tensor
            instead of a memref. To execute reference semantics quantum dialect, please convert
            to the value semantics quantum dialect, where ops are bufferizable and lowerable to
            LLVM IR.
    }];

    let arguments = (ins
        1DTensorOf<[I1]>:$basis_state,
        Variadic<QRefQubitType>:$wires
    );

    let assemblyFormat = [{
        `(` $basis_state`)` $wires attr-dict `:` type(operands)
    }];
}


def CustomOp : UnitaryGate_Op<"custom", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "A generic quantum gate on n qubits with m floating point parameters.";
    let description = [{
    }];

    let arguments = (ins
        Variadic<F64>:$params,
        Variadic<QRefQubitType>:$wires,
        StrAttr:$gate_name,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
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

def PauliRotOp : UnitaryGate_Op<"paulirot", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply a Pauli Product Rotation";
    let description = [{
        The `qref.paulirot` operation applies a rotation around a Pauli product
        operator to the state-vector.
        The arguments are the rotation angle `angle`, a string representing the
        Pauli product operator, and a set of qubits the operation acts on.
        Note that this operation is currently not excutable. There isn't a valid
        lowering path to the LLVM IR.
    }];

    let arguments = (ins
        F64:$angle,
        PauliWord:$pauli_product,
        Variadic<QRefQubitType>:$wires,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        $pauli_product `(` $angle `)` $wires (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($wires) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `angle` is the 0th operand
        }
    }];

    let hasVerifier = 1;
}

def GlobalPhaseOp : UnitaryGate_Op<"gphase", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Global Phase.";

    let description = [{
        Applies global phase to the current system.
    }];

    let arguments = (ins
        F64:$params,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $params `)` (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($params) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `params` is the 0-th operand
        }

        // Simulate missing operands and results for the default impl of the quantum gate interface.
        mlir::OperandRange getWires() {
            return {getOperands().begin(), getOperands().begin()};
        }
        mlir::MutableOperandRange getWiresMutable() {
            return mlir::MutableOperandRange(getOperation(), 0, 0);
        }
    }];
}

def MultiRZOp : UnitaryGate_Op<"multirz", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply an arbitrary multi Z rotation";
    let description = [{
        The `qref.multirz` operation applies an arbitrary multi Z rotation to the state-vector.
        The arguments are the rotation angle `theta` and a set of wires the operation acts on.

        .. note::
            This operation is one of the few quantum operations that is not applied via
            ``qref.custom``. The reason for this is that its quantum dialect counterpart
            needs to be handled in a special way during the lowering due to its C function being
            variadic on the number of qubits.
    }];

    let arguments = (ins
        F64:$theta,
        Variadic<QRefQubitType>:$wires,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $theta `)` $wires (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($wires) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `theta` is the 0-th operand
        }
    }];

    // TODO
    // let hasCanonicalizeMethod = 1;
}

def PCPhaseOp : UnitaryGate_Op<"pcphase", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply a projector-controlled phase gate";
    let description = [{
        This gate is built from simpler gates like `PhaseShift` and `PauliX` and acts on a group
        of wires and takes a rotation angle.
        It also takes another number, an integer called `dim`, which defines a specific part
        of the quantum state. The gate then applies a positive phase shift to a portion of the
        state defined by `dim`. At the same time, it applies a negative phase shift to the rest
        of the state.

        .. note::
            This operation is one of the few quantum operations that is not applied via
            ``quantum.custom``. The reason for this is that its quantum dialect counterpart needs
            to be handled in a special way during the lowering due to its C function being variadic
            on the number of qubits.

        .. note::
            `dim` is currently captured as a float number for compatibility with
            runtime and device integration.

    }];

    let arguments = (ins
        F64:$theta,
        F64:$dim,
        Variadic<QRefQubitType>:$wires,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $theta `,` $dim `)` $wires (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($wires) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `theta` is the 0-th operand
        }
    }];

    // TODO
    // let hasCanonicalizeMethod = 1;
}

def QubitUnitaryOp : UnitaryGate_Op<"unitary", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply an arbitrary fixed unitary matrix";
    let description = [{
        The `qref.unitary` operation applies an arbitrary fixed unitary matrix to the
        state-vector. The arguments are a set of qubits and a 2-dim matrix of complex numbers
        that represents a Unitary matrix of size 2^(number of qubits) * 2^(number of qubits).

        .. note::
            This op is not bufferizable at the moment, and must take in the matrix as a tensor
            instead of a memref. To execute reference semantics quantum dialect, please convert
            to the value semantics quantum dialect, where ops are bufferizable and lowerable to
            LLVM IR.
    }];

    let arguments = (ins
        2DTensorOf<[Complex<F64>]>:$matrix,
        Variadic<QRefQubitType>:$wires,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_wires,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $matrix `:` type($matrix) `)` $wires (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_wires^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($wires) (`ctrls` type($ctrl_wires)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `matrix` is the first operand
        }
    }];

    let hasVerifier = 1;
}
// -----

// Observable ops are not meaningful on their own: their purpose is to be sent into a measurement
// Hence they are Pure, i.e. removable if no users

class Observable_Op<string mnemonic, list<Trait> traits = []> :
        QRef_Op<mnemonic, traits # [Pure]>;

def NamedObsOp : Observable_Op<"namedobs"> {
    let summary = "Define a Named observable for use in measurements";
    let description = [{
        The `qref.namedobs` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents one of 5 named observables
        {Identity, PauliX, PauliY, PauliZ, Hadamard} on a qubit. The arguments are a wire to
        measure as well as an encoding operator for the qubit as an integer between 0-4.
    }];

    let arguments = (ins
        QRefQubitType:$wire,
        NamedObservableAttr:$type
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        $wire `[` $type `]` attr-dict  `:` type(results)
    }];
}

// -----

// class Measurement_Op<string mnemonic, list<Trait> traits = []> :
//         Quantum_Op<mnemonic, traits # [MeasurementProcess]>;

#endif // QREF_OPS
