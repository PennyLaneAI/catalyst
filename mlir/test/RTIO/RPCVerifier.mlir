// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// -----

// Valid: no args, no return
// CHECK-LABEL: module @valid_no_args
module @valid_no_args {
  func.func @__kernel__() {
    rtio.rpc @foo
    return
  }
}

// -----

// Valid: with args
// CHECK-LABEL: module @valid_with_args
module @valid_with_args {
  func.func @__kernel__(%key: i64, %idx: i64, %val: f64) {
    rtio.rpc @bar (%key, %idx, %val : i64, i64, f64)
    return
  }
}

// -----

// Valid: sync RPC with return value
// CHECK-LABEL: module @valid_with_return
module @valid_with_return {
  func.func @__kernel__() -> i64 {
    %x = rtio.rpc @get_value -> i64
    return %x : i64
  }
}

// -----

// Valid: async RPC with args
// CHECK-LABEL: module @valid_async
module @valid_async {
  func.func @__kernel__(%x: i64) {
    rtio.rpc @send_data async (%x : i64)
    return
  }
}

// -----

// Invalid: async RPC cannot have return values
module @invalid_async_with_return {
  func.func @__kernel__() -> i64 {
    // expected-error@+1 {{async RPC cannot have return values}}
    %x = rtio.rpc @get_value async -> i64
    return %x : i64
  }
}
