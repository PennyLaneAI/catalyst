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

include "MBQC/IR/MBQCDialect.td"

//===----------------------------------------------------------------------===//
// MBQC dialect enums.
//===----------------------------------------------------------------------===//

def BlochPlane : I32EnumAttr<"BlochPlane",
    "Planes in the Bloch sphere representation with support for arbitrary-basis measurements",
    [
        I32EnumAttrCase<"XY", 0>,
        I32EnumAttrCase<"ZX", 1>,
        I32EnumAttrCase<"YZ", 2>,
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

def BlochPlaneAttr : EnumAttr<MBQC_Dialect, BlochPlane, "bloch_plane">;


//===----------------------------------------------------------------------===//
// MBQC dialect operations.
//===----------------------------------------------------------------------===//

def MeasureInBasisOp : MBQC_Op<"measure_in_basis"> {
    let summary = "A parametric single-qubit projective measurement in an arbitrary basis.";
    let description = [{
        A parametric single-qubit projective measurement is equivalent to the `quantum.measure`
        operation except that it supports measurements in an arbitrary basis, rather than the
        computational basis only. Specifically, the measurement basis is parameterized by a plane
        along the major axes of the Bloch sphere (either XY, ZX or YZ) and a rotation angle about
        that plane.
    }];

    let arguments = (ins
        QubitType:$in_qubit,
        BlochPlaneAttr:$plane,
        F64:$angle,
        OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>>:$postselect
    );

    let results = (outs
        I1:$mres,
        QubitType:$out_qubit
    );

    let assemblyFormat = [{
        `[` $plane `,` $angle `]` $in_qubit attr-dict `:` type(results)
    }];
}

#endif // MBQC_OPS
