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

// RUN: catalyst --tool=opt --catalyst-pipeline="pipe(user-transform-stage)" --mlir-timing --mlir-timing-display=list %s -o /dev/null 2>&1 | FileCheck %s

// Each pass of the transform sequence has its own timer.

// CHECK-DAG: transform_cse
// CHECK-DAG: transform_canonicalize
// CHECK-DAG: transform_symbol-dce
module @program {
  module @circuit {
    func.func public @circuit() attributes {quantum.node} {
      return
    }

    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "cse" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "canonicalize" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %2 = transform.apply_registered_pass "symbol-dce" to %1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield
      }
    }
  }
}
