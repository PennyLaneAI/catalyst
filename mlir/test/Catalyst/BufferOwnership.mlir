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

// RUN: quantum-opt --annotate-buffer-ownership --split-input-file %s | FileCheck %s

// Callers must pick up newly proven ownership using their cached graphs. Both
// scalar-returning and void functions still need annotations on their calls.
module @forwarded_results {
  func.func @leaf(%input: memref<4xf64>) -> memref<4xf64> {
    %result = memref.alloc() : memref<4xf64>
    return %result : memref<4xf64>
  }

  // CHECK-LABEL: func.func @middle
  // CHECK: call @leaf({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @middle(%input: memref<4xf64>) -> memref<?xf64> {
    %result = call @leaf(%input) : (memref<4xf64>) -> memref<4xf64>
    %cast = memref.cast %result : memref<4xf64> to memref<?xf64>
    return %cast : memref<?xf64>
  }

  // CHECK-LABEL: func.func @outer
  // CHECK: call @middle({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @outer(%input: memref<4xf64>) -> memref<?xf64> {
    %result = call @middle(%input) : (memref<4xf64>) -> memref<?xf64>
    return %result : memref<?xf64>
  }

  // CHECK-LABEL: func.func @scalar_caller
  // CHECK: call @outer({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @scalar_caller(%input: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %result = call @outer(%input) : (memref<4xf64>) -> memref<?xf64>
    %value = memref.load %result[%c0] : memref<?xf64>
    return %value : f64
  }

  // CHECK-LABEL: func.func @void_caller
  // CHECK: call @outer({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @void_caller(%input: memref<4xf64>) {
    %result = call @outer(%input) : (memref<4xf64>) -> memref<?xf64>
    memref.dealloc %result : memref<?xf64>
    return
  }
}

// -----

// A cycle can propagate ownership established by a local allocation, but a
// recursive call alone cannot prove that its result owns an allocation.
module @recursive_results {
  // CHECK-LABEL: func.func @allocating
  // CHECK: call @forwarding({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @allocating(%input: memref<4xf64>, %recurse: i1) -> memref<4xf64> {
    scf.if %recurse {
      %false = arith.constant false
      %temporary = func.call @forwarding(%input, %false) : (memref<4xf64>, i1) -> memref<4xf64>
      memref.dealloc %temporary : memref<4xf64>
    }
    %result = memref.alloc() : memref<4xf64>
    return %result : memref<4xf64>
  }

  // CHECK-LABEL: func.func @forwarding
  // CHECK: call @allocating({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @forwarding(%input: memref<4xf64>, %recurse: i1) -> memref<4xf64> {
    %result = call @allocating(%input, %recurse) : (memref<4xf64>, i1) -> memref<4xf64>
    return %result : memref<4xf64>
  }

  // CHECK-LABEL: func.func @unknown_recursive
  // CHECK: call @unknown_recursive({{.*}}) : (memref<4xf64>) -> memref<4xf64>
  func.func @unknown_recursive(%input: memref<4xf64>) -> memref<4xf64> {
    %result = call @unknown_recursive(%input) : (memref<4xf64>) -> memref<4xf64>
    return %result : memref<4xf64>
  }

  // CHECK-LABEL: func.func @recursive_caller
  // CHECK: call @forwarding({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  // CHECK: call @unknown_recursive({{.*}}) : (memref<4xf64>) -> memref<4xf64>
  func.func @recursive_caller(%input: memref<4xf64>, %recurse: i1) {
    %fresh = call @forwarding(%input, %recurse) : (memref<4xf64>, i1) -> memref<4xf64>
    %unknown = call @unknown_recursive(%input) : (memref<4xf64>) -> memref<4xf64>
    return
  }
}

// -----

// Do not cut off traversal at unowned call results: they can still alias another
// returned buffer. Likewise, unowned views participate in the shared-origin check.
module @result_aliases {
  func.func private @external(memref<4xf64>) -> memref<4xf64>

  func.func @external_alias() -> (memref<4xf64>, memref<4xf64>) {
    %base = memref.alloc() : memref<4xf64>
    %alias = call @external(%base) : (memref<4xf64>) -> memref<4xf64>
    return %base, %alias : memref<4xf64>, memref<4xf64>
  }

  func.func @base_and_view() -> (memref<4xf64>, memref<2xf64, strided<[1], offset: 1>>) {
    %base = memref.alloc() : memref<4xf64>
    %view = memref.subview %base[1][2][1] : memref<4xf64> to memref<2xf64, strided<[1], offset: 1>>
    return %base, %view : memref<4xf64>, memref<2xf64, strided<[1], offset: 1>>
  }

  func.func @mixed(%input: memref<4xf64>) -> (memref<4xf64>, memref<4xf64>, memref<4xf64>, i32) {
    %first = memref.alloc() : memref<4xf64>
    %second = memref.alloc() : memref<4xf64>
    %c0 = arith.constant 0 : i32
    return %first, %input, %second, %c0 : memref<4xf64>, memref<4xf64>, memref<4xf64>, i32
  }

  // CHECK-LABEL: func.func @alias_caller
  // CHECK: call @external_alias() : () -> (memref<4xf64>, memref<4xf64>)
  // CHECK: call @base_and_view() : () -> (memref<4xf64>, memref<2xf64, strided<[1], offset: 1>>)
  // CHECK: call @mixed({{.*}}) {catalyst.owned_buffer_results = array<i32: 0, 2>}
  func.func @alias_caller(%input: memref<4xf64>) {
    %a:2 = call @external_alias() : () -> (memref<4xf64>, memref<4xf64>)
    %b:2 = call @base_and_view() : () -> (memref<4xf64>, memref<2xf64, strided<[1], offset: 1>>)
    %c:4 = call @mixed(%input) : (memref<4xf64>) -> (memref<4xf64>, memref<4xf64>, memref<4xf64>, i32)
    return
  }
}

// -----

// The cached graph retains control-flow edges, including loop-carried cycles,
// and ownership must hold at every return.
module @control_flow {
  func.func @maybe_borrowed(%input: memref<4xf64>, %condition: i1) -> memref<4xf64> {
    cf.cond_br %condition, ^fresh, ^borrowed
  ^fresh:
    %result = memref.alloc() : memref<4xf64>
    return %result : memref<4xf64>
  ^borrowed:
    return %input : memref<4xf64>
  }

  func.func @always_fresh(%condition: i1) -> memref<4xf64> {
    cf.cond_br %condition, ^left, ^right
  ^left:
    %first = memref.alloc() : memref<4xf64>
    return %first : memref<4xf64>
  ^right:
    %second = memref.alloc() : memref<4xf64>
    return %second : memref<4xf64>
  }

  func.func @loop_owned(%count: index) -> memref<4xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %base = memref.alloc() : memref<4xf64>
    %result = scf.for %i = %c0 to %count step %c1 iter_args(%buffer = %base) -> (memref<4xf64>) {
      scf.yield %buffer : memref<4xf64>
    }
    return %result : memref<4xf64>
  }

  // CHECK-LABEL: func.func @control_flow_caller
  // CHECK: call @maybe_borrowed({{.*}}) : (memref<4xf64>, i1) -> memref<4xf64>
  // CHECK: call @always_fresh({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  // CHECK: call @loop_owned({{.*}}) {catalyst.owned_buffer_results = array<i32: 0>}
  func.func @control_flow_caller(%input: memref<4xf64>, %condition: i1, %count: index) {
    %a = call @maybe_borrowed(%input, %condition) : (memref<4xf64>, i1) -> memref<4xf64>
    %b = call @always_fresh(%condition) : (i1) -> memref<4xf64>
    %c = call @loop_owned(%count) : (index) -> memref<4xf64>
    return
  }
}

// -----

// Even when there is nothing to infer, the final walk must remove stale
// annotations from calls to external and scalar-returning functions.
module @no_owned_results {
  func.func private @external() -> memref<4xf64>

  func.func @scalar() -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }

  // CHECK-LABEL: func.func @unowned_caller
  // CHECK: call @external() : () -> memref<4xf64>
  // CHECK: call @scalar() : () -> i32
  func.func @unowned_caller() {
    %buffer = call @external() {catalyst.owned_buffer_results = array<i32: 0>} : () -> memref<4xf64>
    %value = call @scalar() {catalyst.owned_buffer_results = array<i32: 0>} : () -> i32
    return
  }
}
