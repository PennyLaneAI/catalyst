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

// RUN: quantum-opt --pass-pipeline='builtin.module(annotate-buffer-ownership,func.func(buffer-deallocation))' %s | FileCheck %s

module {
  func.func @make(%input: memref<4xf64>) -> memref<4xf64> {
    %result = memref.alloc() : memref<4xf64>
    memref.copy %input, %result : memref<4xf64> to memref<4xf64>
    return %result : memref<4xf64>
  }

  // Skipping inference for a scalar-returning caller must still allow the
  // deallocation pass to manage its owned call result independently of its input.
  // CHECK-LABEL: func.func @caller
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[TMP:%.+]] = memref.alloc() : memref<4xf64>
  // CHECK: [[RESULT:%.+]] = call @make([[TMP]]) : (memref<4xf64>) -> memref<4xf64>
  // CHECK-NEXT: memref.dealloc [[TMP]] : memref<4xf64>
  // CHECK-NEXT: [[VALUE:%.+]] = memref.load [[RESULT]]{{\[}}[[C0]]] : memref<4xf64>
  // CHECK-NEXT: memref.dealloc [[RESULT]] : memref<4xf64>
  // CHECK-NEXT: return [[VALUE]] : f64
  func.func @caller(%input: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() : memref<4xf64>
    memref.copy %input, %tmp : memref<4xf64> to memref<4xf64>
    %result = call @make(%tmp) : (memref<4xf64>) -> memref<4xf64>
    %value = memref.load %result[%c0] : memref<4xf64>
    return %value : f64
  }
}
