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

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "mlir/IR/AttrTypeBase.td"

include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Quantum dialect types.
//===----------------------------------------------------------------------===//

class Quantum_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QuantumDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : Quantum_Type<"Qubit", "bit"> {
    let summary = "A value-semantic qubit (state).";
}
def QuregType : Quantum_Type<"Qureg", "reg"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
}
def ObservableType : Quantum_Type<"Observable", "obs"> {
    let summary = "A quantum observable for use in measurements.";
}
def ResultType : Quantum_Type<"Result", "res"> {
    let summary = "A quantum measurement result.";
}

//===----------------------------------------------------------------------===//
// DO-QAOA types.
//===----------------------------------------------------------------------===//

def PartitionType : Quantum_Type<"Partition", "partition"> {
    let summary = "Frozen-qubit partition descriptor for DO-QAOA.";
    let description = [{
        Carries the number of qubits in the original circuit and the number of
        frozen (hotspot) qubits m, which determines the 2^m sub-problems
        produced by divide-and-conquer QAOA (Section 1.1).
    }];
    let parameters = (ins "unsigned":$numQubits, "unsigned":$m);
    let assemblyFormat = "`<` $numQubits `,` $m `>`";
}

def ClusterMapType : Quantum_Type<"ClusterMap", "cluster_map"> {
    let summary = "Landscape cluster assignment map for DO-QAOA.";
    let description = [{
        Represents the result of clustering 2^m sub-problem energy landscapes
        into K similarity groups. K=1 is the common case for sparse graphs
        with connectivity parameter s > sc ~= 0.6 (Section 2.1).
    }];
    let parameters = (ins "unsigned":$k);
    let assemblyFormat = "`<` $k `>`";
}

def CircuitRefType : Quantum_Type<"CircuitRef", "circuit_ref"> {
    let summary = "Reference to a representative DO-QAOA sub-circuit.";
    let description = [{
        An opaque index identifying which sub-circuit within a landscape
        cluster has been selected as the representative for full variational
        optimisation. Produced by quantum.select_representative.
    }];
}

def ParamsType : Quantum_Type<"Params", "params"> {
    let summary = "QAOA variational parameter buffer for DO-QAOA transfer.";
    let description = [{
        Holds the variational parameters theta = (gamma_1..gamma_p,
        beta_1..beta_p) produced by optimising the representative sub-circuit.
        Used as input and output of quantum.bias_transfer.
    }];
}

def BitstringType : Quantum_Type<"Bitstring", "bitstring"> {
    let summary = "Optimal bitstring solution selected by DO-QAOA.";
    let description = [{
        Binary assignment of graph nodes corresponding to the sub-circuit
        with the minimum expectation value across all 2^m sub-problem
        executions. Produced by quantum.aggregate_min.
    }];
}

#endif // QUANTUM_TYPES
