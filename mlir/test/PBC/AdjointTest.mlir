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

// RUN: quantum-opt --adjoint-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=ADJOINT
// RUN: quantum-opt --ppr-to-ppm --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=PPM

// ADJOINT-LABEL: @workflow
// PPM-LABEL: @workflow
func.func private @workflow(%r: !quantum.reg) -> !quantum.reg attributes {} {
  // ADJOINT-NOT: quantum.adjoint
  // PPM-NOT: quantum.adjoint
  %r_out = quantum.adjoint(%r) : !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %0 = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit

    // ADJOINT: pbc.ppr ["Y"](4)
    // ADJOINT: pbc.ppr ["X"](-2)
    // PPM: pbc.ppm ["Y", "Y"](-)
    // PPM: pbc.ppm ["X"]
    // PPM: pbc.ppr ["X"](-2)
    %1 = pbc.ppr ["X"](2) %0 : !quantum.bit
    %2 = pbc.ppr ["Y"](-4) %1 : !quantum.bit

    %3 = quantum.insert %arg0[0], %2 : !quantum.reg, !quantum.bit
    quantum.yield %3 : !quantum.reg
  }
  return %r_out : !quantum.reg
}

// -----

// PPM-LABEL: @workflow_adjoint_ppr_to_ppm
func.func @workflow_adjoint_ppr_to_ppm() {
  %0 = quantum.alloc(1) : !quantum.reg
  // PPM-NOT: quantum.adjoint
  %1 = quantum.adjoint (%0) : !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %qb = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit
    // PPM: pbc.ppm ["Z", "Y"]
    // PPM: pbc.ppm ["X"]
    // PPM: pbc.ppr ["Z"](2)
    %out = pbc.ppr ["Z"](4) %qb : !quantum.bit
    %inserted = quantum.insert %arg0[0], %out : !quantum.reg, !quantum.bit
    quantum.yield %inserted : !quantum.reg
  }
  quantum.dealloc %1 : !quantum.reg
  return
}
