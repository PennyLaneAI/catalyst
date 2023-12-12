// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "Quantum/IR/QuantumDialect.td"
include "Quantum/IR/QuantumInterfaces.td"

//===----------------------------------------------------------------------===//
// Quantum dialect enums.
//===----------------------------------------------------------------------===//

def NamedObservable : I32EnumAttr<"NamedObservable",
    "Known named observables",
    [
        I32EnumAttrCase<"Identity", 0>,
        I32EnumAttrCase<"PauliX",   1>,
        I32EnumAttrCase<"PauliY",   2>,
        I32EnumAttrCase<"PauliZ",   3>,
        I32EnumAttrCase<"Hadamard", 4>,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}


//===----------------------------------------------------------------------===//
// Quantum dialect traits.
//===----------------------------------------------------------------------===//

def Unitary : NativeOpTrait<"UnitaryTrait">;
def Hermitian : NativeOpTrait<"HermitianTrait">;

def QuantumMemory : Resource<"QuantumMemory">;

//===----------------------------------------------------------------------===//
// Quantum dialect attributes.
//===----------------------------------------------------------------------===//

def NamedObservableAttr : EnumAttr<QuantumDialect, NamedObservable, "named_observable">;

//===----------------------------------------------------------------------===//
// Quantum dialect operations.
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<Trait> traits = []> : Op<QuantumDialect, mnemonic, traits>;

def InitializeOp : Quantum_Op<"init"> {
    let summary = "Initialize the quantum runtime.";

    let assemblyFormat = [{
        attr-dict
    }];
}

def FinalizeOp : Quantum_Op<"finalize"> {
    let summary = "Teardown the quantum runtime.";

    let assemblyFormat = [{
        attr-dict
    }];
}

def DeviceInitOp : Quantum_Op<"device"> {
    let summary = "Initialize a quantum device.";

    let arguments = (ins
        StrAttr:$lib,
        StrAttr:$name,
        StrAttr:$kwargs
    );

    let assemblyFormat = [{
      `[` $lib `,` $name `,` $kwargs `]` attr-dict
    }];
}

def DeviceReleaseOp : Quantum_Op<"drelease"> {
    let summary = "Release the active quantum device.";

    let assemblyFormat = [{
        attr-dict
    }];
}

// -----

class Memory_Op<string mnemonic, list<Trait> traits = []> : Quantum_Op<mnemonic, traits>;

// The operation has no memory effect and is pure, therefore two similar alloc would be removed
// by a CSE pass. This needs to be changed in the future.
def AllocOp : Memory_Op<"alloc", [NoMemoryEffect]> {
    let summary = "Allocate n qubits into a quantum register.";
    let description = [{
    }];

    let arguments = (ins
        Optional<I64>:$nqubits,
        OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$nqubits_attr
    );

    let results = (outs
        QuregType:$qreg
    );

    let assemblyFormat = [{
        `(` ($nqubits^):($nqubits_attr)? `)` attr-dict `:` type(results)
    }];
}

def DeallocOp : Memory_Op<"dealloc"> {
    let summary = "Deallocate a quantum register.";
    let description = [{
    }];

    let arguments = (ins
        QuregType:$qreg
    );

    let assemblyFormat = [{
        $qreg attr-dict `:` type(operands)
    }];

    let hasCanonicalizeMethod = 1;
}

def ExtractOp : Memory_Op<"extract", [NoMemoryEffect]> {
    let summary = "Extract a qubit value from a register.";
    let description = [{
    }];

    let arguments = (ins
        QuregType:$qreg,
        Optional<I64>:$idx,
        OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$idx_attr
    );

    let results = (outs
        QubitType:$qubit
    );

    let assemblyFormat = [{
        $qreg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($qreg) `->` type(results)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def InsertOp : Memory_Op<"insert", [NoMemoryEffect]> {
    let summary = "Update the qubit value of a register.";
    let description = [{
    }];

    let arguments = (ins
        QuregType:$in_qreg,
        Optional<I64>:$idx,
        OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$idx_attr,
        QubitType:$qubit
    );

    let results = (outs
        QuregType:$out_qreg
    );

    let assemblyFormat = [{
        $in_qreg `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` type($in_qreg) `,` type($qubit)
    }];

    let hasCanonicalizeMethod = 1;
    let hasVerifier = 1;
    let hasFolder = 1;
}

// -----

class Gate_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits # [NoMemoryEffect, QuantumGate, Unitary]> {

    code extraBaseClassDeclaration = [{
        mlir::ValueRange getQubitOperands() {
            return getInQubits();
        }

        mlir::ValueRange getQubitResults() {
            return getOutQubits();
        }

        bool getAdjointFlag() {
            return getAdjoint().has_value() ? getAdjoint().value() : false;
        }
        void setAdjointFlag(bool adjoint) {
            setAdjoint(adjoint);
        };
    }];

    let extraClassDeclaration = extraBaseClassDeclaration;
}

def CustomOp : Gate_Op<"custom", [AttrSizedOperandSegments, DifferentiableGate]> {
    let summary = "A generic quantum gate on n qubits with m floating point parameters.";
    let description = [{
    }];

    let arguments = (ins
        Variadic<F64>:$params,
        Variadic<QubitType>:$in_qubits,
        StrAttr:$gate_name,
        OptionalAttr<UnitAttr>:$adjoint
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        $gate_name `(` $params `)` $in_qubits attr-dict `:` type($out_qubits)
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getParams();
        }
    }];
}

def MultiRZOp : Gate_Op<"multirz", [DifferentiableGate]> {
    let summary = "Apply an arbitrary multi Z rotation";
    let description = [{
        The `quantum.multirz` operation applies an arbitrary multi Z rotation to the state-vector.
        The arguments are the rotation angle `theta` and a set of qubits the operation acts on.
    }];

    let arguments = (ins
        F64:$theta,
        Variadic<QubitType>:$in_qubits,
        OptionalAttr<UnitAttr>:$adjoint
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        `(` $theta `)` $in_qubits attr-dict `:` type($out_qubits)
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(getParamOperandIdx());
        }
    }];
}

def QubitUnitaryOp : Gate_Op<"unitary", [ParametrizedGate]> {
    let summary = "Apply an arbitrary fixed unitary matrix";
    let description = [{
        The `quantum.unitary` operation applies an arbitrary fixed unitary matrix to the
        state-vector. The arguments are a set of qubits and a 2-dim matrix of complex numbers
        that represents a Unitary matrix of size 2^(number of qubits) * 2^(number of qubits).
    }];

    let arguments = (ins
        AnyTypeOf<[
            2DTensorOf<[Complex<F64>]>, MemRefRankOf<[Complex<F64>], [2]>
        ]>:$matrix,
        Variadic<QubitType>:$in_qubits,
        OptionalAttr<UnitAttr>:$adjoint
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        `(` $matrix `:` type($matrix) `)` $in_qubits attr-dict `:` type($out_qubits)
    }];

    let extraClassDeclaration = extraBaseClassDeclaration # [{
        mlir::ValueRange getAllParams() {
            return getODSOperands(getParamOperandIdx());
        }
    }];

    let hasVerifier = 1;
}

// -----

class Region_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits # [NoMemoryEffect]>;

def AdjointOp : Region_Op<"adjoint", [QuantumRegion, SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "Calculate the adjoint of the enclosed operations";

    let regions = (region SizedRegion<1>:$region);

    let arguments = (ins
        QuregType:$qreg
    );

    let results = (outs
        QuregType:$out_qreg
    );

    let assemblyFormat = [{
        `(` $qreg `)` attr-dict `:` type(operands) $region
    }];

    let hasVerifier = 1;
}

def YieldOp : Quantum_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["AdjointOp"]>]> {
    let summary = "Return results from quantum program regions";

    let arguments = (ins
        Variadic<AnyTypeOf<[QuregType]>>:$results
    );

    let assemblyFormat = [{
        attr-dict ($results^ `:` type($results))?
    }];

    let builders = [
        OpBuilder<(ins), [{ /* nothing to do */ }]>
    ];
}

// -----

class Observable_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits # [Pure]>;

def ComputationalBasisOp : Observable_Op<"compbasis"> {
    let summary = "Define a pseudo-obeservable of the computational basis for use in measurements";
    let description = [{
        The `quantum.compbasis` operation defines a quantum observable to be used by other
        operations such as measurement processes. The specific observable defined here is a
        "pseudo" observable to represent measurements in the computational basis.
        The only arguments are the list of qubits to measure.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit)
        {
            %res = quantum.compbasis %q0, %q1 : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        Variadic<QubitType>:$qubits
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        $qubits attr-dict `:` type(results)
    }];
}

def NamedObsOp : Observable_Op<"namedobs"> {
    let summary = "Define a Named observable for use in measurements";
    let description = [{
        The `quantum.namedobs` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents one of 5 named observables
        {Identity, PauliX, PauliY, PauliZ, Hadamard} on a qubit. The arguments are a qubit to
        measure as well as an encoding operator for the qubit as an integer between 0-4.

        Example:

        ```mlir
        func.func @foo(%q: !quantum.bit)
        {
            %res = quantum.namedobs %q[4] : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        QubitType:$qubit,
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
        The `quantum.hermitian` operation defines a quantum observable to be used by measurement
        processes. The specific observable defined here represents the Hermitian observable on a
        set of qubits. The arguments are a set of qubits to measure as well as a row-major flatten
        matrix of complex numbers that represents a Hermitian matrix that must be of size
        2^(number of qubits) * 2^(number of qubits).
    }];

    let arguments = (ins
        AnyTypeOf<[
            2DTensorOf<[Complex<F64>]>, MemRefRankOf<[Complex<F64>], [2]>
        ]>:$matrix,
        Variadic<QubitType>:$qubits
    );

    let results = (outs ObservableType:$obs);

    let assemblyFormat = [{
        `(` $matrix `:` type($matrix) `)` $qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

def TensorOp : Observable_Op<"tensor"> {
    let summary = "Define a tensor product of observables for use in measurements";
    let description = [{
        The `quantum.tensor` operation defines a quantum observable to be used by other
        operations such as measurement processes. The specific observable defined here represents
        the tensor product of observables on a set of qubits. The arguments are a set of
        `quantum.Observable` generated by `quantum.namedobs` and `quantum.hermitian`.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit, %m: tensor<2x2xcomplex<f64>>)
        {
            %obs1 = quantum.namedobs %q0[4] : !quantum.obs
            %obs2 = quantum.hermitian(%m: tensor<2x2xcomplex<f64>>) %q1 : !quantum.obs
            %res = quantum.tensorprod %obs1, %obs2 : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        Variadic<ObservableType>:$terms
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        $terms attr-dict `:` type(results)
    }];
}

def HamiltonianOp : Observable_Op<"hamiltonian"> {
    let summary = "Define a Hamiltonian observable for use in measurements";
    let description = [{
        The `quantum.hamiltonian` operation defines a quantum observable to be used by other
        operations such as measurement processes. The specific observable defined here represents
        the hamiltonian of observables. The arguments are a set of coefficients and a set of
        `quantum.Observable` generated by `quantum.namedobs`, `quantum.hermitian`,
        or `quantum.tensorobs`.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit, %c: tensor<2xf64>) {
            %obs1 = quantum.namedobs %q0[4] : !quantum.obs
            %obs2 = quantum.namedobs %q1[2] : !quantum.obs
            %res = quantum.hamiltonian(%c: tensor<2xf64>) %obs1, %obs2 : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        AnyTypeOf<[
            1DTensorOf<[F64]>, MemRefRankOf<[F64], [1]>
        ]>:$coeffs,
        Variadic<ObservableType>:$terms
    );

    let results = (outs
        ObservableType:$obs
    );

    let assemblyFormat = [{
        `(` $coeffs `:` type($coeffs) `)` $terms attr-dict `:` type(results)
    }];
}

// -----

class Measurement_Op<string mnemonic, list<Trait> traits = []> :
        Quantum_Op<mnemonic, traits # [MeasurementProcess]>;

def MeasureOp : Quantum_Op<"measure"> {
    let summary = "A single-qubit projective measurement in the computational basis.";
    let description = [{
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$mres,
        QubitType:$out_qubit
    );

    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def SampleOp : Measurement_Op<"sample"> {
    let summary = "Sample eigenvalues from the given observable for the current state";
    let description = [{
        The `quantum.sample` operation represents the measurement process of sampling eigenvalues
        from an observable on the current quantum state. Given the nature of the operation, an
        attribute specifying the shot number, i.e. the number of samples to draw, must be specified.
        The only SSA argument is an observable that must be defined by an operation in the local
        scope.

        Note that the return value type depends on the type of observable provided. Computational
        basis samples are returned as a 2D array of shape (shot number, number of qubits), with all
        other obversables the output is a 1D array of lenth equal to the shot number.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit)
        {
            %obs1 = quantum.compbasis %q0, %q1 : !quantum.obs
            %samples = quantum.samples %obs1 {shots=1000} : tensor<1000xf64>

            %obs2 = quantum.pauli %q0[3], %q1[1] : !quantum.obs
            %samples2 = quantum.samples %obs2 {shots=1000} : tensor<1000x2xf64>

            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        Optional<
           AnyTypeOf<[
            MemRefRankOf<[F64], [1]>,
            MemRefRankOf<[F64], [2]>
           ]>
        >:$in_data,
        I64Attr:$shots
    );

    let results = (outs
        Optional<
            AnyTypeOf<[
                1DTensorOf<[F64]>,
                2DTensorOf<[F64]>
            ]>
        >:$samples
    );

    let assemblyFormat = [{
        $obs ( `in` `(` $in_data^ `:` type($in_data) `)` )? attr-dict ( `:` type($samples)^ )?
    }];

    let extraClassDeclaration = [{
        bool isBufferized() {
            return getResultTypes().empty();
        }
    }];

    let hasVerifier = 1;
}

def CountsOp : Measurement_Op<"counts", [SameVariadicOperandSize, SameVariadicResultSize]> {
    let summary = "Compute sample counts for the given observable for the current state";
    let description = [{
        The `quantum.counts` operation represents the measurement process of sampling eigenvalues
        from an observable on the current quantum state and counting the frequency of each
        eigenvalue. Given the nature of the operation, an attribute specifying the shot number,
        i.e. the number of samples to draw, must be specified.
        The only SSA argument is an observable that must be defined by an operation in the local
        scope.

        Note that the "counts dictionary" is returned as two separate arrays of the same length, one
        array for the eigenvalues, and one for count of each eigenvalue. When operating in the
        computational basis, the "eigenvalues" are the possible bitstrings one could measure on the
        given qubits, encoded as (floating-point) integers.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit)
        {
            %obs = quantum.compbasis %q0, %q1 : !quantum.obs
            %counts = quantum.counts %obs {shots=1000} : tensor<4xf64>, tensor<4xi64>

            %obs2 = quantum.pauli %q0[3], %q1[1] : !quantum.obs
            %counts2 = quantum.counts %obs2 {shots=1000} : tensor<2xf64>, tensor<2xi64>

            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        Optional<MemRefRankOf<[F64], [1]>>:$in_eigvals,
        Optional<MemRefRankOf<[I64], [1]>>:$in_counts,
        I64Attr:$shots
    );

    let results = (outs
        Optional<1DTensorOf<[F64]>>:$eigvals,
        Optional<1DTensorOf<[I64]>>:$counts
    );

    let assemblyFormat = [{
        $obs
        ( `in` `(` $in_eigvals^ `:` type($in_eigvals) `,` $in_counts `:` type($in_counts) `)` )?
        attr-dict ( `:` type($eigvals)^ `,` type($counts) )?
    }];

    let extraClassDeclaration = [{
        bool isBufferized() {
            return getResultTypes().empty();
        }
    }];

    let hasVerifier = 1;
}

def ExpvalOp : Measurement_Op<"expval"> {
    let summary = "Compute the expectation value of the given observable for the current state";
    let description = [{
        The `quantum.expval` operation represents the measurement process of computing the
        expectation value of an observable on the current quantum state. While this quantity can
        be computed analytically on simulators, an optional attribute specifiying the shot number,
        i.e. the number of samples to draw, can be specified for hardware execution or shot noise
        simulation.
        The only SSA argument is an observable that must be defined by an operation in the local
        scope.

        Example:

        ```mlir
        func.func @foo(%q: !quantum.bit)
        {
            %obs = quantum.namedobs %q[4] : !quantum.obs
            %expval = quantum.expval %obs : f64

            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        OptionalAttr<I64Attr>:$shots
    );

    let results = (outs
        F64:$expval
    );

    let assemblyFormat = [{
        $obs attr-dict `:` type(results)
    }];
}

def VarianceOp : Measurement_Op<"var"> {
    let summary = "Compute the variance of the given observable for the current state";
    let description = [{
        The `quantum.var` operation represents the measurement process of computing the variance of
        an observable on the current quantum state. While this quantity can be computed analytically
        on simulators, an optional attribute specifiying the shot number, i.e. the number of samples
        to draw, can be specified for hardware execution or shot noise simulation.
        The only SSA argument is an observable that must be defined by an operation in the local
        scope.

        Example:

        ```mlir
        func.func @foo(%q: !quantum.bit)
        {
            %obs = quantum.namedobs %q[4] : !quantum.obs
            %var = quantum.var %obs : f64

            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        OptionalAttr<I64Attr>:$shots
    );

    let results = (outs
        F64:$variance
    );

    let assemblyFormat = [{
        $obs attr-dict `:` type(results)
    }];
}

def ProbsOp : Measurement_Op<"probs"> {
    let summary = "Compute computational basis probabilities for the current state";
    let description = [{
        The `quantum.probs` operation represents the measurement process of computing probabilities
        for measurement outcomes in the computational basis for a set of qubits.
        Marginal probabilities are supported, that is the provided qubits do not need to make up the
        entire statevector.
        The result array contains one element for each possible bitstring, i.e. 2^n where n is the
        number of qubits.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit)
        {
            %probs = quantum.probs %q0, %q1 : tensor<4xf64>
            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        Optional<MemRefRankOf<[F64], [1]>>:$state_in
    );

    let results = (outs
        Optional<1DTensorOf<[F64]>>:$probabilities
    );

    let assemblyFormat = [{
        $obs ( `in` `(` $state_in^ `:` type($state_in) `)` )? attr-dict ( `:` type($probabilities)^ )?
    }];

    let extraClassDeclaration = [{
        bool isBufferized() {
            return getResultTypes().empty();
        }
    }];

    let hasVerifier = 1;
}

def StateOp : Measurement_Op<"state"> {
    let summary = "Return the current statevector";
    let description = [{
        The `quantum.state` operation represents the measurement process of returning the current
        statevector in the computational basis. Typically reserved for simulator devices, although
        in principle also achievable on hardware via tomography techniques.
        While marginal states are supported, the operation is only well-defined if the provided
        qubits are not entangled with the rest of the quantum state.
        The result array contains one (complex) element for each possible bitstring, i.e. 2^n where
        n is the number of qubits.

        Example:

        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit)
        {
            %state = quantum.state %q0, %q1 : tensor<4xcomplex<f64>>
            func.return
        }
        ```
    }];

    let arguments = (ins
        ObservableType:$obs,
        Optional<MemRefRankOf<[Complex<F64>], [1]>>:$state_in
    );

    let results = (outs
        Optional<1DTensorOf<[Complex<F64>]>> :$state
    );

    let extraClassDeclaration = [{
        bool isBufferized() {
            return getResultTypes().empty();
        }
    }];

    let assemblyFormat = [{
        $obs ( `in` `(` $state_in^ `:` type($state_in) `)` )? attr-dict ( `:` type($state)^ )?
    }];

    let hasVerifier = 1;
}

#endif // QUANTUM_OPS
