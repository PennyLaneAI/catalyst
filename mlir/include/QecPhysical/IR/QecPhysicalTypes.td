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

#ifndef QEC_PHYSICAL_TYPES
#define QEC_PHYSICAL_TYPES

include "mlir/IR/AttrTypeBase.td"

include "QecPhysical/IR/QecPhysicalDialect.td"

//===----------------------------------------------------------------------===//
// QecPhysical dialect types.
//===----------------------------------------------------------------------===//

class QecPhysical_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QecPhysical_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}


def QecPhysicalQubitType : QecPhysical_Type<"QecPhysicalQubit", "qubit"> {
    let summary = "A value-semantic QEC physical qubit";

    let parameters = (ins
        TypeParameter<"QecPhysicalQubitRole", "Qubit role">:$role
    );

    let assemblyFormat = "`<` $role `>`";
}

def PhysicalCodeblockType : QecPhysical_Type<"Codeblock", "codeblock"> {
    let summary = "A value-semantic physical codeblock";

    let description = [{
        A physical codeblock type is an abstraction around a collection of QEC physical qubits that
        encode the logical codeblock. More specifically, it represents a block of *n* physical
        qubits that encode *k* logical qubits for a general *((n, k, d))* QEC code, where *d* is the
        code distance.

        Unlike logical codeblocks, but like quantum registers, physics qubits *can* be extracted
        from or inserted into a physical codeblock. The reason for this design is that while at the
        QEC logical layer above, where it is generally not possible to extract a single logical
        qubit in the codeblock for codes with k > 1, it *is* possible to act on individual physical
        qubits in the physical codeblock, since these objects represent real quantum resources.

        #### Example:

        ```mlir
        // A physical codeblock that encodes one logical qubit with 7 physical qubits (k = 1, n = 7)
        !qecp.codeblock<1 x 7>
        ```
    }];

    let parameters = (ins
        TypeParameter<"int64_t", "Number of logical qubits encoded">:$k,
        TypeParameter<"int64_t", "Physical codeblock size">:$n
    );

    let assemblyFormat = "`<` $k `x` $n `>`";
}

def PhysicalHyperRegisterType : QecPhysical_Type<"HyperRegister", "hyperreg"> {
    let summary = "A value-semantic physical hyper-register";

    let description = [{
        A physical hyper-register type is an abstraction that represents a sequence of physical
        codeblocks.

        The "dimensions" of a physical hyper-register are specified by its *width* (the number of
        physical codeblocks in the hyper-register), the value *k* (the number of logical qubits
        encoded in each codeblock), and the value *n*, the physical codeblock size (the number of
        physical qubits in the codeblock that encode the *k* logical qubits).

        #### Example:

        ```mlir
        // A physical hyper-register containing three size-7 physical codeblock, where each
        // codeblock encode one logical qubit (width = 3, k = 1, n = 7)
        !qecp.hyperreg<3 x 1 x 7>
        ```
    }];

    let parameters = (ins
        TypeParameter<"int64_t", "Physical hyper-register width">:$width,
        TypeParameter<"int64_t", "Number of logical qubits encoded">:$k,
        TypeParameter<"int64_t", "Physical codeblock size">:$n
    );

    // TODO: By default, this assembly format expects whitespace before and after the 'x's.
    //       We could write custom printers/parsers as a workaround.
    let assemblyFormat = "`<` $width `x` $k `x` $n `>`";
}

def TannerGraphType : QecPhysical_Type<"TannerGraph", "tanner_graph"> {
    let summary = "A Tanner graph represented by its adjacency matrix in CSC form";

    let description = [{
        A Tanner graph, in the context of QEC, is a multipartite graph that specifies a quantum
        error-correcting code. A Tanner graph can be partitioned into some number, N, of independent
        sets representing the data qubits of the QEC code and the N - 1 sets representing the
        auxiliary qubits that participate in each class of parity checking (e.g. for CSS codes,
        there are two independent sets of auxiliary qubits for X-parity and Z-parity checks).

        This implementation represents a Tanner graph by its adjacency matrix in Compressed Sparse
        Column (CSC) form. In this representation, the adjacency matrix is fully defined by
        one-dimensional arrays of row indices and column pointers. The "data" array does not need to
        be specified since Tanner graphs are unweighted, therefore the data array is always all 1s.

        #### Example:

        Consider the the following adjacency matrix:

        ```
        ┌           ┐
        │ 0 0 0 1 0 │
        │ 0 0 0 1 1 │
        │ 0 0 0 0 1 │
        │ 1 1 0 0 0 │
        │ 0 1 1 0 0 │
        └           ┘
        ```

        This matrix represents the Tanner graph for three data qubits labelled by column indices
        0, 1, 2 and two auxiliary qubits labelled by column indices 3, 4. A value of '1' indicates
        an edge between vertices. The CSC representation of this adjacency matrix is as follows:

        ```
        row_idx = [3, 3, 4, 4, 0, 1, 1, 2]
        col_ptr = [0, 1, 3, 4, 6, 8]
        ```

        In MLIR, the `tanner_graph` type is fully specified by the sizes of these two arrays (8 and
        6, respectively), and the array element type (e.g. i32):

        ```mlir
        !qecp.tanner_graph<8, 6, i32>
        ```
    }];

    let parameters = (ins
        TypeParameter<"int64_t", "Size of row index array">:$rowIdxSize,
        TypeParameter<"int64_t", "Size of column pointer array">:$colPtrSize,
        TypeParameter<"Type", "Element type">:$elementType
    );

    let assemblyFormat = "`<` $rowIdxSize `,` $colPtrSize `,` $elementType `>`";
}

#endif // QEC_PHYSICAL_TYPES
