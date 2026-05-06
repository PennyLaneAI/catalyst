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

#ifndef QUANTUM_ATTR_DEFS
#define QUANTUM_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Quantum dialect enums and attributes.
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

def NamedObservableAttr : EnumAttr<QuantumDialect, NamedObservable, "named_observable">;

def QubitLevel : I32EnumAttr<"QubitLevel",
    "Qubit levels in the hierarchical qubit representation",
    [
        I32EnumAttrCase<"Abstract", 0, "abstract">,
        I32EnumAttrCase<"Logical",  1, "logical">,
        I32EnumAttrCase<"QEC",      2, "qec">,
        I32EnumAttrCase<"Physical", 3, "physical">,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

def QubitRole : I32EnumAttr<"QubitRole",
    "Qubit roles for further specialization in the hierarchical qubit representation",
    [
        I32EnumAttrCase<"Null",   0, "null">,
        I32EnumAttrCase<"Data",   1, "data">,
        I32EnumAttrCase<"XCheck", 2, "xcheck">,
        I32EnumAttrCase<"ZCheck", 3, "zcheck">,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

def PauliWord : TypedArrayAttrBase<StrAttr, "A product of Pauli operators, aka a Pauli word.">;

//===----------------------------------------------------------------------===//
// DO-QAOA graph metadata attributes.
//
// Two representations are provided:
//   DenseGraphAttr  — for small graphs: stores the full NxN f64 weight matrix
//                     as a DenseElementsAttr (tensor<NxNxf64>).
//   SparseGraphAttr — for large graphs: COO (Coordinate) encoding storing only
//                     non-zero edge weights J_ij as three parallel arrays.
//===----------------------------------------------------------------------===//

class Quantum_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QuantumDialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def DenseGraphAttr : Quantum_Attr<"DenseGraph", "dense_graph"> {
    let summary = "Dense NxN adjacency/weight matrix for DO-QAOA (small graphs).";
    let description = [{
        Stores the symmetric edge-weight matrix J_ij as a rank-2
        `tensor<NxNxf64>` DenseElementsAttr. Suitable for graphs with up to
        ~64 nodes where the full matrix fits in L2 cache. The diagonal is
        always zero (no self-loops). Off-diagonal entry [i,j] holds J_ij.
    }];
    let parameters = (ins
        "uint32_t":$numNodes,
        AttrParameter<"::mlir::DenseElementsAttr",
                      "Symmetric NxN f64 weight matrix">:$weights
    );
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
    let dependentDialects = ["mlir::BuiltinDialect"];
}

def SparseGraphAttr : Quantum_Attr<"SparseGraph", "sparse_graph"> {
    let summary = "COO sparse adjacency/weight encoding for DO-QAOA (large graphs).";
    let description = [{
        Stores only non-zero edge weights J_ij in COO (Coordinate) format.
        rowIndices and colIndices are flat int32 arrays of length numEdges
        (upper-triangle only: i < j). weights is a 1-D DenseElementsAttr of
        shape [numEdges] and element type f64. Suitable for sparse graphs
        where the number of edges E << N^2.
    }];
    let parameters = (ins
        "uint32_t":$numNodes,
        "uint32_t":$numEdges,
        ArrayRefParameter<"int32_t", "COO row indices (i < j)">:$rowIndices,
        ArrayRefParameter<"int32_t", "COO col indices (i < j)">:$colIndices,
        AttrParameter<"::mlir::DenseElementsAttr",
                      "1-D f64 edge weights J_ij">:$weights
    );
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
    let dependentDialects = ["mlir::BuiltinDialect"];
}

#endif // QUANTUM_ATTR_DEFS
