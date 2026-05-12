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

def RefPPMeasurementOp : PBC_Op<"ref.ppm"> {
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
