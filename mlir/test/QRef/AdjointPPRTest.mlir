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

// CHECK-LABEL: @workflow
// CHECK-NOT: qref.adjoint
// CHECK: [[Q:%.+]] = qref.get %{{.*}}[ 0]
// CHECK: pbc.ref.ppr ["Y"](4) [[Q]]
// CHECK: [[Q2:%.+]] = qref.get %{{.*}}[ 0]
// CHECK: pbc.ref.ppr ["X"](-2) [[Q2]]
func.func private @workflow(%r: !qref.reg<?>) {
  qref.adjoint {
    %q = qref.get %r[0] : !qref.reg<?> -> !qref.bit
    pbc.ref.ppr ["X"](2) %q
    pbc.ref.ppr ["Y"](-4) %q
  }
  return
}
