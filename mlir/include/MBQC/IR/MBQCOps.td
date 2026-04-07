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

#ifndef MBQC_OPS
#define MBQC_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"
include "Quantum/IR/QuantumTypes.td"

include "MBQC/IR/MBQCDialect.td"

//===----------------------------------------------------------------------===//
// MBQC dialect enums.
//===----------------------------------------------------------------------===//

def MeasurementPlane : I32EnumAttr<"MeasurementPlane",
    "Planes in the Bloch sphere representation with support for arbitrary-basis measurements",
    [
        I32EnumAttrCase<"XY", 0>,
        I32EnumAttrCase<"YZ", 1>,
        I32EnumAttrCase<"ZX", 2>,
    ]> {
    let cppNamespace = "catalyst::mbqc";
    let genSpecializedAttr = 0;
}


//===----------------------------------------------------------------------===//
// MBQC dialect traits.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// MBQC dialect attributes.
//===----------------------------------------------------------------------===//

def MeasurementPlaneAttr : EnumAttr<MBQC_Dialect, MeasurementPlane, "measurement_plane">;


//===----------------------------------------------------------------------===//
// MBQC dialect operations.
//===----------------------------------------------------------------------===//

def MeasureInBasisOp : MBQC_Op<"measure_in_basis"> {
    let summary = "A parametric single-qubit projective measurement in an arbitrary basis.";
    let description = [{
        A parametric single-qubit projective measurement is equivalent to the `quantum.measure`
        operation except that it supports measurements in an arbitrary basis, rather than the
        computational basis only. Specifically, the measurement basis is parameterized by a plane
        along the major axes of the Bloch sphere (either XY, YZ or ZX) and a rotation angle about
        that plane.

        The convention we have adopted to denote the measurement plane is with a two-character
        string (one of "XY", "YZ" or "ZX"), where the first character indicates the axis along which
        the basis vector is aligned for a rotation angle of 0, and the second character indicates
        the other axis that forms the plane. For instance, the measurement basis defined by the XY
        plane with a rotation angle of 0 is the Pauli-X basis, with orthonormal basis vector |+> and
        |->, which point along the +x and -x axes of the Bloch sphere, respectively. For more
        details, see the documentation for the [`qml.ftqc.measure_arbitrary_basis()`
        ](https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.measure_arbitrary_basis.html)
        function.
    }];

    let arguments = (ins
        QubitType:$in_qubit,
        MeasurementPlaneAttr:$plane,
        F64:$angle,
        OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>>:$postselect
    );

    let results = (outs
        I1:$mres,
        QubitType:$out_qubit
    );

    let assemblyFormat = [{
        `[` $plane `,` $angle `]` $in_qubit (`postselect` $postselect^)? attr-dict `:` type(results)
    }];
}

def GraphStatePrepOp : MBQC_Op<"graph_state_prep"> {
    let summary = "Allocate resources for a new graph state.";
    let description = [{
        This operation allocates the resources for a new quantum *graph state* for use in MBQC
        workloads. A graph state is a highly entangled group of qubits arranged according to an
        arbitrary graph structure, where each qubit is represented by a vertex of the graph and
        where there is an edge between every interacting pair of neighbouring qubits.

        A graph state requires three parameters to be fully expressed:

        1. A graph representing the qubit connectivity.
        2. The qubit initial state. All qubits in the graph state are assumed to be initialized
           to the same state. This parameter is expressed as the name of the one-qubit gate that
           would have to be applied to the :math:`|0\rangle` state to realize the desired state.
        3. The type of entangling interaction along each edge. All interactions between
           neighbouring qubits are assumed to be the same. This parameter is expressed as the name
           of the two-qubit gate that would have to be applied to the pairs of neighbouring qubits
           to realize the desired entangled state.

        A *densely packed adjacency matrix* is used to represent the graph structure. This
        representations is a more compact form of the familiar adjacency matrix graph
        representation, where only the lower-triangular part of the matrix is stored (since the
        graph is undirected) and where the diagonal elements are excluded (since there are no qubit
        self-interactions). The matrix values are then packed into a flat array. For example, the
        following graph structure,

        .. code-block::

            0 -- 1 -- 2 -- 3

        has the adjacency matrix representation (with row and column indices shown for reference):

        .. code-block::

                 0  1  2  3

            0    0  1  0  0
            1    1  0  1  0
            2    0  1  0  1
            3    0  0  1  0

        The equivalent densely packed adjacency matrix has the lower-triangular part of the above
        matrix packed into a flat array using row-major order as follows (with extra space added
        between sequences of values to indicate the rows of the matrix):

        .. code-block::

            [1,   0, 1,   0, 0, 1]

        In this representation, the number of vertices, :math:`N`, in the graph is related to the
        number of elements, :math:`m`, in the densely packed adjacency matrix via

        .. math::

            m = N(N - 1) / 2,

        or inversely,

        .. math::

            N = (1 + \sqrt{1 + 8m}) / 2.
    }];

    let arguments = (ins
        AnyTypeOf<[
            1DTensorOf<[I1]>, MemRefRankOf<[I1], [1]>
        ]>:$adj_matrix,
        StrAttr:$init_op,
        StrAttr:$entangle_op
    );

    let results = (outs
        QuregType:$qreg
    );

    let assemblyFormat = [{
        `(` $adj_matrix `:` type($adj_matrix) `)` `[` `init` $init_op `,` `entangle` $entangle_op `]` attr-dict `:` type(results)
    }];
}

#endif // MBQC_OPS
