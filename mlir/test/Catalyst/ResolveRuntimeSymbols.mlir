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

// RUN: quantum-opt %s --split-input-file --resolve-runtime-symbols --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @symbols_are_kept
func.func @symbols_are_kept(%arg0: i32) {
  // CHECK: fn("native_add")
  %0 = catalyst.runtime_call fn("native_add") (%arg0, %arg0) {
      c_params = ["i32", "i32"],
      c_result = "i32"
  } : (i32, i32) -> (i32)

  // CHECK: fn("_ZN4demo11mangled_addEii")
  %1 = catalyst.runtime_call fn("_ZN4demo11mangled_addEii") (%arg0, %arg0) {
      c_params = ["i32", "i32"],
      c_result = "i32"
  } : (i32, i32) -> (i32)
  return
}

// -----

// CHECK-LABEL: @dispatched_calls_are_kept
func.func @dispatched_calls_are_kept(%arg0: i32) {
  // CHECK: fn("demo::add")
  %0 = catalyst.runtime_call fn("demo::add") (%arg0, %arg0) {
      c_params = ["i32", "i32"],
      c_result = "i32",
      dispatch = "192.0.2.1:9000"
  } : (i32, i32) -> (i32)
  return
}

// -----

func.func @no_library(%arg0: i32) {
  // expected-error@+1 {{cannot resolve the C++ function 'demo::add' because no library was specified}}
  %0 = catalyst.runtime_call fn("demo::add") (%arg0, %arg0) {
      c_params = ["i32", "i32"],
      c_result = "i32"
  } : (i32, i32) -> (i32)
  return
}

// -----

module attributes {catalyst.runtime_artifacts = ["/nonexistent/libdemo.so"]} {
  func.func @unreadable_library(%arg0: i32) {
    // expected-error@+1 {{cannot read the symbols of '/nonexistent/libdemo.so'}}
    %0 = catalyst.runtime_call fn("demo::add(int, int)") (%arg0, %arg0) {
        c_params = ["i32", "i32"],
        c_result = "i32"
    } : (i32, i32) -> (i32)
    return
  }
}
