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

// RUN: quantum-opt %s --split-input-file --lower-runtime-dispatch --dispatch-executor-targets | FileCheck %s

// -----

// The kernel launch and the symbol call share the single session.
// CHECK-LABEL: func.func @main
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK-NOT: executor.open
// CHECK-DAG: executor.send_binary %[[S]]("/tmp/target.o")
// CHECK-DAG: executor.launch %[[S]]("compute", "/tmp/target.o")
// CHECK-DAG: executor.call %[[S]]("foo")
module @jit_shared {
  func.func @setup() {
    return
  }
  func.func @teardown() {
    return
  }
  func.func @main() {
    catalyst.runtime_call fn("foo") () {
        c_params = [], c_result = "void", dispatch = "ADDR:PORT"
    }
    catalyst.launch_kernel @target::@compute() : () -> ()
    return
  }
  module @target attributes {catalyst.object_file = "/tmp/target.o", catalyst.dispatch = {address = "ADDR:PORT"}} {
    func.func public @compute() {
      return
    }
  }
}

// -----

// Calls addressed to different executors keep their own sessions.
// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[A:.*]] = executor.open("host:1") : !executor.session
// CHECK-DAG: %[[B:.*]] = executor.open("host:2") : !executor.session
// CHECK-DAG: executor.call %[[A]]("first")
// CHECK-DAG: executor.call %[[B]]("second")
module @jit_two_executors {
  func.func @setup() {
    return
  }
  func.func @main() {
    catalyst.runtime_call fn("first") () {
        c_params = [], c_result = "void", dispatch = "host:1"
    }
    catalyst.runtime_call fn("second") () {
        c_params = [], c_result = "void", dispatch = "host:2"
    }
    return
  }
}
