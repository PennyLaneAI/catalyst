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

// RUN: quantum-opt %s | quantum-opt | FileCheck %s
// RUN: quantum-opt --mlir-print-op-generic %s | quantum-opt | FileCheck %s

// CHECK-LABEL: @escaped_name
func.func @escaped_name(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: quantum.operator "Quoted \22name\22"() qubits(%{{.*}})
  %out = quantum.operator "Quoted \"name\""() qubits(%q)
  return %out : !quantum.bit
}

// CHECK-LABEL: @quoted_map_keys
func.func @quoted_map_keys(%p0: f64, %p1: i64, %q0: !quantum.bit, %q1: !quantum.bit)
    -> (!quantum.bit, !quantum.bit) {
  // CHECK: quantum.operator "QuotedKeys"(%{{.*}}: f64, %{{.*}}: i64) qubits(%{{.*}}, %{{.*}})
  // CHECK-NEXT: param_map = {"1" = [1], "my-param" = [0]} qubit_map = {"wire pair" = [0, 1]}
  %out:2 = quantum.operator "QuotedKeys"(%p0: f64, %p1: i64) qubits(%q0, %q1)
      param_map = {"my-param" = [0], "1" = [1]}
      qubit_map = {"wire pair" = [0, 1]}
  return %out#0, %out#1 : !quantum.bit, !quantum.bit
}
