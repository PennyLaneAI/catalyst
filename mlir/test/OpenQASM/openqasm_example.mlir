// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --translate-to-openqasm --split-input-file --verify-diagnostics | FileCheck %s

module {
    openqasm.allocate(2)
    openqasm.ry(0.7) 0
    openqasm.cnot 0, 1
}

// CHECK:   OPENQASM 3.0;
// CHECK:   include "stdgates.inc";

// CHECK:   qubit[2] q;
// CHECK:   ry(7.000000e-01) q[0];
// CHECK:   cx q[0], q[1];

// CHECK:   ------------------------------
// CHECK:   ORIGINAL IR:
// CHECK:   ------------------------------

// CHECK:   module {
  // CHECK:    openqasm.allocate(2)
  // CHECK:    openqasm.ry(0.69999999999999996) 0
  // CHECK:    openqasm.cnot 0, 1
// CHECK:   }
