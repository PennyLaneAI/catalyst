// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --cp-global-memref --split-input-file %s | FileCheck %s

// Test that functions which do not return a memref are unaffected by the transformation.

// CHECK-LABEL: @wrapper
// CHECK-NOT: llvm.copy_memref
func.func public @wrapper() -> f64 attributes {llvm.emit_c_interface} {
  %0 = arith.constant 0.0 : f64
  func.return %0 : f64
}

// -----

// Test that functions which do return a memref are affected by the transformation

// CHECK-LABEL: @wrapper
// CHECK-SAME: llvm.copy_memref
func.func public @wrapper() -> memref<1xf64> attributes {llvm.emit_c_interface} {
  %0 = arith.constant 0.0 : f64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf64>
  %c0 = arith.constant 0 : index
  memref.store %0, %alloc[%c0] : memref<1xf64>
  func.return %alloc : memref<1xf64>
}

// -----

// Test that the transformation actually happened
// CHECK-LABEL: @wrapper
func.func public @wrapper() -> memref<1xf64> attributes {llvm.emit_c_interface} {
  %0 = arith.constant 0.0 : f64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf64>
  %c0 = arith.constant 0 : index
  memref.store %0, %alloc[%c0] : memref<1xf64>
  // CHECK: [[deadbeef:%.+]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: [[struct:%.+]] = builtin.unrealized_conversion_cast %alloc
  // CHECK: [[ptr:%.+]] = llvm.extractvalue [[struct]][0]
  // CHECK: [[int:%.+]] = llvm.ptrtoint [[ptr]]
  // CHECK: [[cmp:%.+]] = llvm.icmp "eq" [[deadbeef]], [[int]]
  // CHECK: [[res:%.+]] = scf.if [[cmp]]
  // CHECK-NEXT: [[new:%.+]] = memref.alloc
  // CHECK-NEXT: memref.copy %alloc, [[new]]
  // CHECK-NEXT: scf.yield [[new]]
  // CHECK-NEXT: else
  // CHECK-NEXT: scf.yield %alloc
  // CHECK: return [[res]]
  func.return %alloc : memref<1xf64>
}


// -----

// Test that transformation supports dynamic shapes.
// CHECK-LABEL: @wrapper
// CHECK-SAME: [[arg:%.+]]:
// CHECK-SAME: llvm.copy_memref
func.func public @wrapper(%x: memref<3x?x?xf64>) -> memref<3x?x?xf64> attributes {llvm.emit_c_interface} {
  // CHECK: [[res:%.+]] = scf.if [[cmp]]
  // CHECK-NEXT: [[d0:%.+]] = memref.dim
  // CHECK-NEXT: [[d1:%.+]] = memref.dim
  // CHECK-NEXT: [[new:%.+]] = memref.alloc([[d0]], [[d1]])
  // CHECK-NEXT: memref.copy [[arg]], [[new]]
  // CHECK-NEXT: scf.yield [[new]]
  // CHECK-NEXT: else
  // CHECK-NEXT: scf.yield [[arg]]
  // CHECK: return [[res]]
  func.return %x : memref<3x?x?xf64>
}


