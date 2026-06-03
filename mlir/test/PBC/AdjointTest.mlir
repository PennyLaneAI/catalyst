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

// RUN: quantum-opt --adjoint-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:      @workflow
func.func private @workflow(%r: !quantum.reg) -> !quantum.reg attributes {} {
  // CHECK-NOT: quantum.adjoint
  %r_out = quantum.adjoint(%r) : !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %0 = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit

    // CHECK: pbc.ppr ["Y"](4)
    // CHECK: pbc.ppr ["X"](-2)
    %1 = pbc.ppr ["X"](2) %0 : !quantum.bit
    %2 = pbc.ppr ["Y"](-4) %1 : !quantum.bit

    %3 = quantum.insert %arg0[0], %2 : !quantum.reg, !quantum.bit
    quantum.yield %3 : !quantum.reg
  }
  return %r_out : !quantum.reg
}
