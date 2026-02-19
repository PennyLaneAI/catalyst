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
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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
        This operation represents the allocation of some number of qubits.
    }];

    let arguments = (ins
        Optional<I64>:$nqubits,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$nqubits_attr
    );

    let results = (outs
        QRefQuregType:$qreg
    );

    let assemblyFormat = [{
        `(` ($nqubits^):($nqubits_attr)? `)` attr-dict `:` type(results)
    }];

    let hasCanonicalizeMethod = 1;
    let hasVerifier = 1;
}

def AllocQubitOp : Memory_Op<"alloc_qb"> {
    let summary = "Allocate a single qubit.";
    let description = [{
        This operation represents the allocation of a single qubit.
    }];

    let results = (outs
        QRefQubitType:$qubit
    );

    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let builders = [
        OpBuilder<(ins),[{
            AllocQubitOp::build($_builder, $_state,
            /*qubit=*/ $_builder.getType<QubitType>()
            );
        }]>,
    ];

    code extraBaseClassDeclaration = [{
        mlir::OpResult getOutQubit() {
            return getODSResults(0).front();
        }
    }];

    let extraClassDeclaration = extraBaseClassDeclaration;
}

def DeallocQubitOp : Memory_Op<"dealloc_qb"> {
    let summary = "Deallocate a single qubit.";
    let description = [{
        This operation represents the deallocation of a single qubit.
    }];

    let arguments = (ins
        QRefQubitType:$qubit
    );

    let assemblyFormat = [{
        $qubit attr-dict `:` type(operands)
    }];
}

def DeallocOp : Memory_Op<"dealloc"> {
    let summary = "Deallocate a batch of qubits.";
    let description = [{
        This operation represents the allocation of some number of qubits.
    }];

    let arguments = (ins
        QRefQuregType:$qreg
    );

    let assemblyFormat = [{
        $qreg attr-dict `:` type(operands)
    }];

    let hasCanonicalizeMethod = 1;
}

def GetOp : Memory_Op<"get", [NoMemoryEffect]> {
    let summary = "Produce a qubit reference SSA value from an allocation and an index.";
    let description = [{
        This operation produces a qubit reference SSA value from an allocation and an index.

        It is analogous to the quantum.extract operation in the value semantics quantum dialect, but
        with two crucial differences:
        1. In the value semantics dialect, an extracted qubit value represents an abstract data edge
        not a real value in memory; however here in the reference semantics dialect, a qubit value
        represents a concrete, classical qubit ID.
        2. Unlike the quantum.extract operation, the qubit values do not have to be inserted back
        into the register they come from, since the circuit topology is not represented via SSA
        dataflow.
    }];

    let arguments = (ins
        QRefQuregType:$qreg,
        Optional<I64>:$idx,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        QRefQubitType:$qubit
    );

    let assemblyFormat = [{
        $qreg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type(operands) `->` type(results)
    }];
}

// -----

class Gate_Op<string mnemonic, list<Trait> traits = []> :
        QRef_Op<mnemonic, traits # [QuantumOperation]> {

    code extraBaseClassDeclaration = [{
        std::vector<mlir::Value> getQubitOperands() {
            std::vector<mlir::Value> values;
            values.insert(values.end(), getQubits().begin(), getQubits().end());
            return values;
        }

        void setQubitOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange qubits = getQubitsMutable();
            assert(qubits.size() == replacements.size() && "must provide values for all qubits");
            qubits.assign(replacements);
        }
    }];

    let extraClassDeclaration = extraBaseClassDeclaration;
}

class UnitaryGate_Op<string mnemonic, list<Trait> traits = []> :
        Gate_Op<mnemonic, traits # [QuantumGate, Unitary]> {

    code extraBaseClassDeclaration = [{
        std::vector<mlir::Value> getQubitOperands() {
            std::vector<mlir::Value> values;
            values.insert(values.end(), getQubits().begin(), getQubits().end());
            values.insert(values.end(), getCtrlQubits().begin(), getCtrlQubits().end());
            return values;
        }

        void setQubitOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange qubits = getQubitsMutable();
            mlir::MutableOperandRange ctrls = getCtrlQubitsMutable();
            assert(qubits.size() + ctrls.size() == replacements.size() &&
                   "must provide values for all qubits (including controls)");

            qubits.assign(replacements.take_front(qubits.size()));
            ctrls.assign(replacements.take_back(ctrls.size()));
        }

        mlir::ValueRange getNonCtrlQubitOperands() {
            return getQubits();
        }

        void setNonCtrlQubitOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange qubits = getQubitsMutable();
            assert(qubits.size() == replacements.size() &&
                   "must provide values for all non-ctrl qubit values");
            qubits.assign(replacements);
        }

        mlir::ValueRange getCtrlQubitOperands() {
            return getCtrlQubits();
        }

        void setCtrlQubitOperands(mlir::ValueRange replacements) {
            mlir::MutableOperandRange ctrlQubits = getCtrlQubitsMutable();
            assert(ctrlQubits.size() == replacements.size() &&
                   "must provide values for all ctrl qubit values");
            ctrlQubits.assign(replacements);
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
        }
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
        Variadic<QRefQubitType>:$qubits
    );

    let assemblyFormat = [{
        `(` $in_state `)` $qubits attr-dict `:` type(operands)
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
        Variadic<QRefQubitType>:$qubits
    );

    let assemblyFormat = [{
        `(` $basis_state`)` $qubits attr-dict `:` type(operands)
    }];
}


def CustomOp : UnitaryGate_Op<"custom", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "A generic quantum gate on n qubits with m floating point parameters.";
    let description = [{
        This operation represents a generic quantum gate on an arbitrary number of qubits, with
        an arbitrary number of floating point parameters.
    }];

    let arguments = (ins
        Variadic<F64>:$params,
        Variadic<QRefQubitType>:$qubits,
        StrAttr:$gate_name,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        $gate_name `(` $params `)` $qubits (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($qubits) (`ctrls` type($ctrl_qubits)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getParams();
        }
    }];

    let hasCanonicalizeMethod = 1;
    let hasVerifier = 1;
}

def PauliRotOp : UnitaryGate_Op<"paulirot", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply a Pauli Product Rotation";
    let description = [{
        The `qref.paulirot` operation applies a rotation around a Pauli product
        operator to the state-vector.
        The arguments are the rotation angle `angle`, a string representing the
        Pauli product operator, and a set of qubits the operation acts on.
    }];

    let arguments = (ins
        F64:$angle,
        PauliWord:$pauli_product,
        Variadic<QRefQubitType>:$qubits,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        $pauli_product `(` $angle `)` $qubits (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($qubits) (`ctrls` type($ctrl_qubits)^ )?
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
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $params `)` (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($params) (`ctrls` type($ctrl_qubits)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `params` is the 0-th operand
        }

        // Simulate missing operands and results for the default impl of the quantum gate interface.
        mlir::OperandRange getQubits() {
            return {getOperands().begin(), getOperands().begin()};
        }
        mlir::MutableOperandRange getQubitsMutable() {
            return mlir::MutableOperandRange(getOperation(), 0, 0);
        }
    }];
}

def MultiRZOp : UnitaryGate_Op<"multirz", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply an arbitrary multi Z rotation";
    let description = [{
        The `qref.multirz` operation applies an arbitrary multi Z rotation to the state-vector.
        The arguments are the rotation angle `theta` and a set of qubits the operation acts on.

        .. note::
            This operation is one of the few quantum operations that is not applied via
            ``qref.custom``. The reason for this is that its quantum dialect counterpart
            needs to be handled in a special way during the lowering due to its C function being
            variadic on the number of qubits.
    }];

    let arguments = (ins
        F64:$theta,
        Variadic<QRefQubitType>:$qubits,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $theta `)` $qubits (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($qubits) (`ctrls` type($ctrl_qubits)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `theta` is the 0-th operand
        }
    }];

    let hasCanonicalizeMethod = 1;
}

def PCPhaseOp : UnitaryGate_Op<"pcphase", [ParametrizedGate, AttrSizedOperandSegments]> {
    let summary = "Apply a projector-controlled phase gate";
    let description = [{
        This gate is built from simpler gates like `PhaseShift` and `PauliX` and acts on a group
        of qubits and takes a rotation angle.
        It also takes another number, an integer called `dim`, which defines a specific part
        of the quantum state. The gate then applies a positive phase shift to a portion of the
        state defined by `dim`. At the same time, it applies a negative phase shift to the rest
        of the state.

        .. note::
            This operation is one of the few quantum operations that is not applied via
            ``qref.custom``. The reason for this is that its quantum dialect counterpart needs
            to be handled in a special way during the lowering due to its C function being variadic
            on the number of qubits.

        .. note::
            `dim` is currently captured as a float number for compatibility with
            runtime and device integration.

    }];

    let arguments = (ins
        F64:$theta,
        F64:$dim,
        Variadic<QRefQubitType>:$qubits,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $theta `,` $dim `)` $qubits (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($qubits) (`ctrls` type($ctrl_qubits)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `theta` is the 0-th operand
        }
    }];

    let hasCanonicalizeMethod = 1;
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
        Variadic<QRefQubitType>:$qubits,
        UnitAttr:$adjoint,
        Variadic<QRefQubitType>:$ctrl_qubits,
        Variadic<I1>:$ctrl_values
    );

    let assemblyFormat = [{
        `(` $matrix `:` type($matrix) `)` $qubits (`adj` $adjoint^)? attr-dict
        ( `ctrls` `(` $ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $ctrl_values^ `)` )?
        `:` type($qubits) (`ctrls` type($ctrl_qubits)^ )?
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(0);  // `matrix` is the first operand
        }
    }];

    let hasVerifier = 1;
}

// -----

class Region_Op<string mnemonic, list<Trait> traits = []> :
        QRef_Op<mnemonic, traits # []>;

def AdjointOp : Region_Op<"adjoint", [QuantumRegion, NoTerminator]> {
    let summary = "Calculate the adjoint of the enclosed operations";

    let regions = (region SizedRegion<1>:$region);

    let arguments = (ins
        QRefQuregType:$qreg
    );

    let assemblyFormat = [{
        `(` $qreg `)` attr-dict `:` type(operands) $region
    }];

    let hasVerifier = 1;
}

// -----

// Observable ops are not meaningful on their own: their purpose is to be sent into a measurement
// Hence they are Pure, i.e. removable if no users

class Observable_Op<string mnemonic, list<Trait> traits = []> :
        QRef_Op<mnemonic, traits # [Pure]>;

def ComputationalBasisOp : Observable_Op<"compbasis", [AttrSizedOperandSegments]> {
    let summary = "Define a pseudo-observable of the computational basis for use in measurements";
    let description = [{
        The `qref.compbasis` operation defines a quantum observable to be used by other
        operations such as measurement processes. The specific observable defined here is a
        "pseudo" observable to represent measurements in the computational basis.
        The only arguments are either the list of qubit references to measure, or the quantum
        register reference to measure. When the argument is a register, the measurement occurs
        on all qubits in the register.

        Example:

        ```mlir
        func.func @foo(%q0: !qref.bit, %q1: !qref.bit, %r: !qref.reg<5>)
        {
            %obs_q = qref.compbasis qubits %q0, %q1 : !quantum.obs
            %obs_r = qref.compbasis (qreg %r : !qref.reg<5>) : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        Variadic<QRefQubitType>:$qubits,
        Optional<QRefQuregType>:$qreg
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        (`qubits` $qubits^)? (`(` `qreg` $qreg `:` type($qreg)^ `)`)? attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

def NamedObsOp : Observable_Op<"namedobs"> {
    let summary = "Define a Named observable for use in measurements";
    let description = [{
        The `qref.namedobs` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents one of 5 named observables
        {Identity, PauliX, PauliY, PauliZ, Hadamard} on a qubit. The arguments are a qubit to
        measure as well as an encoding operator for the qubit as an integer between 0-4.
    }];

    let arguments = (ins
        QRefQubitType:$qubit,
        NamedObservableAttr:$type
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        $qubit `[` $type `]` attr-dict  `:` type(results)
    }];
}

def HermitianOp : Observable_Op<"hermitian"> {
    let summary = "Define a Hermitian observable for use in measurements";
    let description = [{
        The `qref.hermitian` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents the Hermitian observable on a
        set of qubits. The arguments are a set of qubits to measure as well as a row-major flatten
        matrix of complex numbers that represents a Hermitian matrix that must be of size
        2^(number of qubits) * 2^(number of qubits).

        .. note::
            This op is not bufferizable at the moment, and must take in the Hermitian matrix as a
            tensor instead of a memref. To execute reference semantics quantum dialect, please
            convert to the value semantics quantum dialect, where ops are bufferizable and lowerable
            to LLVM IR.
    }];

    let arguments = (ins
        2DTensorOf<[Complex<F64>]>:$matrix,
        Variadic<QRefQubitType>:$qubits
    );

    let results = (outs ObservableType:$obs);

    let assemblyFormat = [{
        `(` $matrix `:` type($matrix) `)` $qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

// -----

def MeasureOp : QRef_Op<"measure"> {
    let summary = "A single-qubit projective measurement in the computational basis.";
    let description = [{
        This operation represents a single-qubit computational basis measurement.
        Note that in PennyLane, measurements are performed in the 0/1 convention, instead of the
        1/-1 convention.
    }];

    let arguments = (ins
        QRefQubitType:$in_qubit,
        OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>>:$postselect
    );

    let results = (outs
        I1:$mres
    );

    let assemblyFormat = [{
        $in_qubit (`postselect` $postselect^)? attr-dict `:` type(results)
    }];
}

#endif // QREF_OPS
