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

// RUN: quantum-opt %s --transform-cleanup --split-input-file --verify-diagnostics | FileCheck %s

module @workflow attributes {transform.with_named_sequence} {
  func.func public @f(%arg0: tensor<f64>) -> (tensor<f64>) {
    return %arg0 : tensor<f64>
  }

  func.func public @g(%arg0: tensor<f64>) -> (tensor<f64>) {
    return %arg0 : tensor<f64>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.op<"func.func">){
    %0 = transform.apply_registered_pass "some-pass" to %arg0 {options = "func-name=f"}: (!transform.op<"func.func">) -> !transform.op<"func.func">
    %1 = transform.apply_registered_pass "some-other-pass" to %0 {options = "func-name=g"}: (!transform.op<"func.func">) -> !transform.op<"func.func">
    transform.yield
  }
}

// CHECK: module @workflow {
// CHECK-NOT: module @workflow attributes {transform.with_named_sequence} {
// CHECK-NOT: transform.named_sequence @__transform_main
// CHECK-NOT: {{%.+}} = transform.apply_registered_pass "some-pass" to {{%.+}} {options = "func-name=f"}
// CHECK-NOT: {{%.+}} = transform.apply_registered_pass "some-other-pass" to {{%.+}} {options = "func-name=g"}
// CHECK-NOT: transform.yield