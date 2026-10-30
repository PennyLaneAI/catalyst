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

// RUN: quantum-opt %s --split-input-file --lower-runtime-dispatch --verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func.func @bound
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK: %[[ARG:.*]] = tensor.from_elements %arg0 : tensor<1xi32>
// CHECK: %[[R:.*]] = executor.call %[[S]]("foo") (%[[ARG]]) {num_input_args = 1 : i32}
// CHECK-SAME: !executor.session, (tensor<1xi32>) -> tensor<1xi32>
// CHECK: %[[OUT:.*]] = tensor.extract %[[R]]
// CHECK: return %[[OUT]]
// CHECK-NOT: catalyst.runtime_call
module @jit_bound {
  func.func @setup() {
    return
  }
  func.func @bound(%arg0: i32) -> i32 {
    %0 = catalyst.runtime_call fn("foo") (%arg0) {
        c_params = ["i32"],
        c_result = "i32",
        dispatch = "ADDR:PORT"
    } : (i32) -> (i32)
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: func.func @fields
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK: %[[SESS:.*]] = tensor.from_elements %arg0 : tensor<1xi64>
// CHECK: %[[KEY:.*]] = arith.constant dense<{{.*}}> : tensor<256xi8>
// CHECK: %[[N:.*]] = tensor.from_elements %arg1 : tensor<1xi32>
// CHECK: executor.call %[[S]]("collect") (%[[SESS]], %[[KEY]], %[[N]]) {num_input_args = 3 : i32}
// CHECK-SAME: (tensor<1xi64>, tensor<256xi8>, tensor<1xi32>) -> (tensor<1xi32>, tensor<64xi8>)
module @jit_fields {
  func.func @setup() {
    return
  }
  func.func @fields(%arg0: i64, %arg1: i32) -> tensor<64xi8> {
    %status, %reply = catalyst.runtime_call fn("collect") (%arg0, %arg1) {
        c_params = ["ptr", "str", "out", "u32"],
        c_result = "i32",
        c_strings = ["decoder-0"],
        dispatch = "ADDR:PORT"
    } : (i64, i32) -> (i32) outs(tensor<64xi8>)
    return %reply : tensor<64xi8>
  }
}

// -----

// CHECK-LABEL: func.func @void_call
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK: executor.call %[[S]]("reset") () {num_input_args = 0 : i32}
// CHECK-SAME: !executor.session, () -> ()
module @jit_void {
  func.func @setup() {
    return
  }
  func.func @void_call() {
    catalyst.runtime_call fn("reset") () {
        c_params = [],
        c_result = "void",
        dispatch = "ADDR:PORT"
    }
    return
  }
}

// -----

// Two calls to the same executor in one function share a single session.
// CHECK-LABEL: func.func @shared_session
// CHECK: executor.open("ADDR:PORT")
// CHECK-NOT: executor.open
module @jit_shared_session {
  func.func @setup() {
    return
  }
  func.func @shared_session() {
    catalyst.runtime_call fn("first") () {
        c_params = [], c_result = "void", dispatch = "ADDR:PORT"
    }
    catalyst.runtime_call fn("second") () {
        c_params = [], c_result = "void", dispatch = "ADDR:PORT"
    }
    return
  }
}

// -----

// An empty dispatch binds to the program's single executor.
// CHECK-LABEL: func.func @inherit
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK: executor.call %[[S]]("foo") ()
module @jit_inherit {
  func.func @setup() {
    return
  }
  func.func @inherit() {
    catalyst.runtime_call fn("foo") () {
        c_params = [], c_result = "void", dispatch = ""
    }
    return
  }
  module @target attributes {catalyst.object_file = "/tmp/target.o", catalyst.dispatch = {address = "ADDR:PORT"}} {
    func.func public @compute() {
      return
    }
  }
}

// -----

// A call that names no executor cannot be routed when the program targets more than one.
module @jit_ambiguous {
  func.func @setup() {
    return
  }
  func.func @ambiguous() {
    // expected-error @below {{ambiguous executor: the program targets 2 executors}}
    catalyst.runtime_call fn("foo") () {
        c_params = [], c_result = "void", dispatch = ""
    }
    return
  }
  module @t1 attributes {catalyst.object_file = "/tmp/t1.o", catalyst.dispatch = {address = "host:1"}} {
    func.func public @c1() {
      return
    }
  }
  module @t2 attributes {catalyst.object_file = "/tmp/t2.o", catalyst.dispatch = {address = "host:2"}} {
    func.func public @c2() {
      return
    }
  }
}

// -----

// Without any executor in the program there is no address to bind an empty dispatch to.
module @jit_unaddressed {
  func.func @unaddressed() {
    // expected-error @below {{dispatch has no executor address}}
    catalyst.runtime_call fn("foo") () {
        c_params = [], c_result = "void", dispatch = ""
    }
    return
  }
}
