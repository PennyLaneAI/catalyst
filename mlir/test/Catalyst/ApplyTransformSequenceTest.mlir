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

// RUN: quantum-opt %s --apply-transform-sequence --split-input-file --verify-diagnostics | FileCheck %s

// This file tests that the --apply-transform-sequence:
// 1. Correctly removes the transformer module from the payload module
// 2. Correctly applies the transformer module
// (using -remove-chained-self-inverse as the example)
// 3. Silently passes on modules with no transform program

module @workflow {

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "remove-chained-self-inverse" to %arg0 {options = "func-name=f"} : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }

  func.func private @f(%arg0: tensor<f64>) -> !quantum.bit {
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %extracted = tensor.extract %c_0[] : tensor<i64>
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }


  func.func private @g(%arg0: tensor<f64>) -> !quantum.bit {
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %extracted = tensor.extract %c_0[] : tensor<i64>
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }

}

// CHECK-LABEL: workflow
// CHECK-NOT: module attributes {transform.with_named_sequence}
// CHECK-NOT: transform.named_sequence @__transform_main
// CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f"}
// CHECK-NOT: transform.yield

// CHECK-LABEL: f
// CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit

// CHECK-LABEL: g
// CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit

// -----

// CHECK-LABEL: @empty_workflow
module @empty_workflow {

}
