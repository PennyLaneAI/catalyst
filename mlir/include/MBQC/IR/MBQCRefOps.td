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

def RefMeasureInBasisOp : MBQC_Op<"ref.measure_in_basis"> {
    let summary = "A parametric single-qubit projective measurement in an arbitrary basis.";
    let description = [{
        Identical to the `mbqc.measure_in_basis` operation, except in reference semantics.
    }];

    let arguments = (ins
        QubitRefType:$in_qubit,
        MeasurementPlaneAttr:$plane,
        F64:$angle,
        OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>>:$postselect
    );

    let results = (outs
        I1:$mres
    );

    let assemblyFormat = [{
        `[` $plane `,` $angle `]` $in_qubit (`postselect` $postselect^)? attr-dict `:` type(results)
    }];
}

def RefGraphStatePrepOp : MBQC_Op<"ref.graph_state_prep"> {
    let summary = "Allocate resources for a new graph state.";
    let description = [{
        Identical to mbqc.graph_state_prep operation, except in reference semantics.
    }];

    let arguments = (ins
        AnyTypeOf<[
            1DTensorOf<[I1]>, MemRefRankOf<[I1], [1]>
        ]>:$adj_matrix,
        StrAttr:$init_op,
        StrAttr:$entangle_op
    );

    let results = (outs
        QuregRefType:$qreg
    );

    let assemblyFormat = [{
        `(` $adj_matrix `:` type($adj_matrix) `)` `[` `init` $init_op `,` `entangle` $entangle_op `]` attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}
