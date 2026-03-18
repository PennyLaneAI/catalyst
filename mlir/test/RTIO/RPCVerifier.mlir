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

// Valid: no args, tag n:
// CHECK-LABEL: module @valid_no_args
module @valid_no_args {
  func.func @__kernel__() {
    rtio.rpc @foo tag("n:")
    return
  }
}


// -----

// Valid: args match tag n:IIf
// CHECK-LABEL: module @valid_with_args
module @valid_with_args {
  func.func @__kernel__(%key: i64, %idx: i64, %val: f64) {
    rtio.rpc @bar tag("n:IIf") (%key, %idx, %val : i64, i64, f64)
    return
  }
}

// -----

// Valid: sync RPC with return value
// CHECK-LABEL: module @valid_with_return
module @valid_with_return {
  func.func @__kernel__() -> i64 {
    %x = rtio.rpc @get_value tag("I:") -> i64
    return %x : i64
  }
}


// -----

module @invalid_tag_format {
  func.func @__kernel__() {
    // expected-error@+1 {{tag must be in format '<return>:<args>' (e.g. n:IIf)}}
    rtio.rpc @foo tag("n")
    return
  }
}

// -----

module @invalid_arg_count {
  func.func @__kernel__(%x: i64) {
    // expected-error@+1 {{tag has 2 arg type code(s) but 1 argument(s) provided}}
    rtio.rpc @foo tag("n:II") (%x : i64)
    return
  }
}

// -----

module @invalid_type_mismatch {
  func.func @__kernel__(%x: f64) {
    // expected-error@+1 {{argument 0 has type 'f64' which is incompatible with tag code 'I'}}
    rtio.rpc @foo tag("n:I") (%x : f64)
    return
  }
}
