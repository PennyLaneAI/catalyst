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

// RUN: quantum-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{unknown-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

// A whole tensor is stored as one contiguous run of elements, whatever its rank and element type.
// The block operations take the collapsed rank-1 view of its buffer, which is a view and not a
// copy: no element is loaded or stored by the bufferization itself.

// CHECK-LABEL: func.func @push_pop_complex_matrix
// CHECK: [[list:%.+]] = catalyst.list_init : <complex<f64>>
// CHECK: [[buffer:%.+]] = bufferization.to_buffer %{{.*}} : tensor<2x2xcomplex<f64>> to memref<2x2xcomplex<f64>>
// CHECK: [[block:%.+]] = memref.collapse_shape [[buffer]]
// CHECK-SAME: memref<2x2xcomplex<f64>> into memref<4xcomplex<f64>>
// CHECK: catalyst.list_push_block [[block]], [[list]] : memref<4xcomplex<f64>>, <complex<f64>>
// The destination buffer comes from bufferizing tensor.empty, so it is owned and freed like any
// other allocation and the popped elements outlive the list.
// CHECK: [[dest:%.+]] = memref.alloc() {{.*}}: memref<2x2xcomplex<f64>>
// CHECK: [[destBlock:%.+]] = memref.collapse_shape [[dest]]
// CHECK-SAME: memref<2x2xcomplex<f64>> into memref<4xcomplex<f64>>
// The bufferized pop writes its destination in place, so it has no result.
// CHECK: catalyst.list_pop_block [[list]], [[destBlock]] : <complex<f64>>, memref<4xcomplex<f64>>{{$}}
// CHECK: catalyst.list_dealloc [[list]]
// CHECK: bufferization.to_tensor [[dest]]
func.func @push_pop_complex_matrix(%matrix: tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>> {
  %list = catalyst.list_init : !catalyst.arraylist<complex<f64>>
  catalyst.list_push_block %matrix, %list : tensor<2x2xcomplex<f64>>, !catalyst.arraylist<complex<f64>>
  %empty = tensor.empty() : tensor<2x2xcomplex<f64>>
  %restored = catalyst.list_pop_block %list, %empty
    : !catalyst.arraylist<complex<f64>>, tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
  catalyst.list_dealloc %list : !catalyst.arraylist<complex<f64>>
  return %restored : tensor<2x2xcomplex<f64>>
}

// -----

// A rank-1 tensor is already contiguous, so there is nothing to collapse.

// CHECK-LABEL: func.func @push_vector
// CHECK-NOT: memref.collapse_shape
// CHECK: catalyst.list_push_block %{{.*}}, %{{.*}} : memref<4xf64>, <f64>
func.func @push_vector(%angles: tensor<4xf64>) {
  %list = catalyst.list_init : !catalyst.arraylist<f64>
  catalyst.list_push_block %angles, %list : tensor<4xf64>, !catalyst.arraylist<f64>
  catalyst.list_dealloc %list : !catalyst.arraylist<f64>
  return
}

// -----

// The elements pushed are only read, so an operand whose buffer is a strided view can be copied
// into a fresh contiguous buffer first. This is the same fallback `catalyst.launch_kernel` uses.

// CHECK-LABEL: func.func @push_strided
// CHECK: [[slice:%.+]] = memref.subview
// CHECK-SAME: memref<8xf64> to memref<4xf64, strided<[1], offset: 2>>
// CHECK: [[contiguous:%.+]] = memref.alloc() : memref<4xf64>
// CHECK: memref.copy [[slice]], [[contiguous]]
// CHECK: catalyst.list_push_block [[contiguous]], %{{.*}} : memref<4xf64>, <f64>
func.func @push_strided(%angles: tensor<8xf64>) {
  %list = catalyst.list_init : !catalyst.arraylist<f64>
  %slice = tensor.extract_slice %angles[2] [4] [1] : tensor<8xf64> to tensor<4xf64>
  catalyst.list_push_block %slice, %list : tensor<4xf64>, !catalyst.arraylist<f64>
  catalyst.list_dealloc %list : !catalyst.arraylist<f64>
  return
}

// -----

// Dynamic dimensions need no special handling: the collapsed block simply has a dynamic size.

// CHECK-LABEL: func.func @pop_dynamic
// CHECK: [[dest:%.+]] = memref.alloc(%{{.*}}, %{{.*}}) {{.*}}: memref<?x?xf64>
// CHECK: [[block:%.+]] = memref.collapse_shape [[dest]]
// CHECK-SAME: memref<?x?xf64> into memref<?xf64>
// CHECK: catalyst.list_pop_block %{{.*}}, [[block]] : <f64>, memref<?xf64>
// CHECK: bufferization.to_tensor [[dest]]
func.func @pop_dynamic(%rows: index, %columns: index) -> tensor<?x?xf64> {
  %list = catalyst.list_init : !catalyst.arraylist<f64>
  %empty = tensor.empty(%rows, %columns) : tensor<?x?xf64>
  %restored = catalyst.list_pop_block %list, %empty
    : !catalyst.arraylist<f64>, tensor<?x?xf64> -> tensor<?x?xf64>
  catalyst.list_dealloc %list : !catalyst.arraylist<f64>
  return %restored : tensor<?x?xf64>
}
