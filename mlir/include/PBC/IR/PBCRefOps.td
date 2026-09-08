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

include "QRef/IR/QRefTypes.td"

def RefPPMeasurementOp : PBC_Op<"ref.ppm",
    [DeclareOpInterfaceMethods<ResourceQuantumOpInterface, ["getResourceDetailedName"]>]> {
    let summary = "Pauli Product Measurement on qubit references.";

    let description = [{
        Identical to the pbc.ppm operation, except in reference semantics.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,
        UnitAttr:$negated,  // Pauli words can carry a negative sign, which flips the result
        Variadic<QubitRefType>:$qubits
    );

    let results = (outs
        I1:$mres
    );

    let assemblyFormat = [{
        $pauli_product (`(``-``)` $negated^)? $qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

def RefPPRotationOp : PBC_Op<"ref.ppr", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ResourceQuantumOpInterface, ["getResourceAdjointFlag"]>]> {
    let summary = "Pauli Product Rotation on qubit references.";

    let description = [{
        Identical to the pbc.ppr operation, except in reference semantics.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product to apply (e.g., ["X", "I", "Z"])
        ConfinedAttr<SI8Attr, [RotationKindConstraint]>:$rotation_kind,  // angle in fractions of π
        Variadic<QubitRefType>:$qubits,  // The qubits to apply the rotation to
        Optional<I1>:$condition
    );

    let assemblyFormat = [{
        $pauli_product `(` $rotation_kind `)` $qubits attr-dict (`cond` `(` $condition^ `)`)?
    }];

    let hasVerifier = 1;
}

def RefSelectPPMeasurementOp : PBC_Op<"ref.select.ppm",
    [DeclareOpInterfaceMethods<ResourceQuantumOpInterface>]> {
    let summary = "Multiplexed Pauli product measurement on qubit references.";

    let description = [{
        Identical to the pbc.select.ppm operation, except in reference semantics.
    }];

    let arguments = (ins
        I1:$select_switch,
        PauliWord:$pauli_product_0,
        UnitAttr:$negated_0,
        PauliWord:$pauli_product_1,
        UnitAttr:$negated_1,
        Variadic<QubitRefType>:$qubits  // The qubits to measure
    );

    let results = (outs
        I1:$mres  // The measurement result (0 or 1)
    );

    let assemblyFormat = [{
        ` ` `(` $select_switch `?` $pauli_product_0 (`(``-``)` $negated_0^)? `:` $pauli_product_1 (`(``-``)` $negated_1^)? `)` $qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

def RefFabricateOp : PBC_Op<"ref.fabricate",
    [DeclareOpInterfaceMethods<ResourceQuantumOpInterface>, DeclareOpInterfaceMethods<ResourceAllocQubitOpInterface>]> {
    let summary = "Fabricate auxiliary qubit references from qubit factories.";

    let description = [{
        Identical to the pbc.fabricate operation, except in reference semantics.
    }];

    let arguments = (ins
        LogicalInit:$init_state
    );

    let results = (outs
        Variadic<QubitRefType>:$out_qubits
    );

    let assemblyFormat = [{
        $init_state attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;
}

def RefPrepareStateOp : PBC_Op<"ref.prepare",
    [DeclareOpInterfaceMethods<ResourceQuantumOpInterface>, DeclareOpInterfaceMethods<ResourceAllocQubitOpInterface>]> {
    let summary = "Prepare auxiliary qubit references in a given state.";

    let description = [{
        Identical to the pbc.prepare operation, except in reference semantics.
    }];

    let arguments = (ins
        LogicalInit:$init_state
    );

    let results = (outs
        Variadic<QubitRefType>:$out_qubits
    );

    let assemblyFormat = [{
        $init_state attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;
}
