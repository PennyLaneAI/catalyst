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

// RUN: quantum-opt %s --convert-arraylist-to-memref --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @list_init()
func.func @list_init() {
    %0 = catalyst.list_init : !catalyst.arraylist<f64>
    // CHECK: [[initCapacity:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK-NEXT: [[zero:%.+]] = arith.constant 0 : index
    // CHECK-NEXT: [[data:%.+]] = memref.alloc([[initCapacity]]) : memref<?xf64>
    // CHECK-NEXT: [[dataField:%.+]] = memref.alloc() : memref<memref<?xf64>>
    // CHECK-NEXT: [[sizeField:%.+]] = memref.alloc() : memref<index>
    // CHECK-NEXT: [[capacityField:%.+]] = memref.alloc() : memref<index>
    // CHECK-NEXT: memref.store [[data]], [[dataField]]
    // CHECK-NEXT: memref.store [[zero]], [[sizeField]]
    // CHECK-NEXT: memref.store [[initCapacity]], [[capacityField]]
    // CHECK-NEXT: {{%.+}} = builtin.unrealized_conversion_cast [[dataField]], [[sizeField]], [[capacityField]] : memref<memref<?xf64>>, memref<index>, memref<index> to !catalyst.arraylist<f64>
    return
}

// CHECK: func.func @list_push([[list:%.+]]: !catalyst.arraylist<f64>, [[val:%.+]]: f64)
func.func @list_push(%arg0: !catalyst.arraylist<f64>, %arg1: f64) {
    catalyst.list_push %arg1, %arg0 : !catalyst.arraylist<f64>
    // CHECK: [[unpacked:%.+]]:3 = builtin.unrealized_conversion_cast [[list]]
    // CHECK: call @__catalyst_arraylist_pushf64([[unpacked]]#0, [[unpacked]]#1, [[unpacked]]#2, [[val]])
    return
}

// CHECK: func.func @list_load_data([[list:%.+]]: !catalyst.arraylist<f64>)
func.func @list_load_data(%arg0: !catalyst.arraylist<f64>) -> memref<?xf64> {
    %data = catalyst.list_load_data %arg0 : !catalyst.arraylist<f64> -> memref<?xf64>
    // CHECK: [[unpacked:%.+]]:3 = builtin.unrealized_conversion_cast [[list]]
    // CHECK: [[data:%.+]] = memref.load [[unpacked]]#0
    // CHECK: [[size:%.+]] = memref.load [[unpacked]]#1
    // CHECK: [[view:%.+]] = memref.subview [[data]][0] [[[size]]] [1] : memref<?xf64> to memref<?xf64>
    return %data : memref<?xf64>
    // CHECK: return [[view]]
}

// -----

// The block helpers copy between the caller's buffer and the list's own storage, one element at a
// time. A `memref.copy` between subviews would be lowered to a call to the `memrefCopy` runtime
// symbol, which compiled programs do not link.

// CHECK: func.func private @__catalyst_arraylist_push_blockf64(
// CHECK-SAME: [[data:%[^:]+]]: memref<memref<?xf64>>, [[size:%[^:]+]]: memref<index>,
// CHECK-SAME: [[capacity:%[^:]+]]: memref<index>, [[block:%[^:]+]]: memref<?xf64>
// CHECK-DAG: [[c0:%.+]] = arith.constant 0 : index
// CHECK: [[sizeVal:%.+]] = memref.load [[size]]
// CHECK: [[capacityVal:%.+]] = memref.load [[capacity]]
// CHECK: [[count:%.+]] = memref.dim [[block]], [[c0]]
// CHECK: [[newSize:%.+]] = arith.addi [[sizeVal]], [[count]]
// The list grows geometrically, but never by less than the block needs.
// CHECK: [[grow:%.+]] = arith.cmpi ugt, [[newSize]], [[capacityVal]]
// CHECK: scf.if [[grow]]
// CHECK:   [[doubled:%.+]] = arith.muli [[capacityVal]]
// CHECK:   [[newCapacity:%.+]] = arith.maxui [[doubled]], [[newSize]]
// CHECK:   [[oldData:%.+]] = memref.load [[data]]
// CHECK:   [[newData:%.+]] = memref.realloc [[oldData]]([[newCapacity]])
// CHECK:   memref.store [[newData]], [[data]]
// CHECK:   memref.store [[newCapacity]], [[capacity]]
// CHECK: [[elements:%.+]] = memref.load [[data]]
// CHECK: scf.for [[i:%.+]] = {{%.+}} to [[count]] step {{%.+}} {
// CHECK:   [[to:%.+]] = arith.addi [[sizeVal]], [[i]]
// CHECK:   [[element:%.+]] = memref.load [[block]]
// CHECK:   memref.store [[element]], [[elements]]{{\[}}[[to]]{{\]}}
// CHECK: memref.store [[newSize]], [[size]]

// CHECK: func.func @list_push_block([[list:%.+]]: !catalyst.arraylist<f64>, [[values:%.+]]: memref<4xf64>)
func.func @list_push_block(%arg0: !catalyst.arraylist<f64>, %arg1: memref<4xf64>) {
    catalyst.list_push_block %arg1, %arg0 : memref<4xf64>, !catalyst.arraylist<f64>
    // CHECK: [[unpacked:%.+]]:3 = builtin.unrealized_conversion_cast [[list]]
    // CHECK: [[cast:%.+]] = memref.cast [[values]] : memref<4xf64> to memref<?xf64>
    // CHECK: call @__catalyst_arraylist_push_blockf64([[unpacked]]#0, [[unpacked]]#1, [[unpacked]]#2, [[cast]])
    return
}

// -----

// CHECK: func.func private @__catalyst_arraylist_pop_blockf64(
// CHECK-SAME: [[data:%[^:]+]]: memref<memref<?xf64>>, [[size:%[^:]+]]: memref<index>,
// CHECK-SAME: [[capacity:%[^:]+]]: memref<index>, [[dest:%[^:]+]]: memref<?xf64>
// CHECK-DAG: [[c0:%.+]] = arith.constant 0 : index
// CHECK: [[sizeVal:%.+]] = memref.load [[size]]
// CHECK: [[count:%.+]] = memref.dim [[dest]], [[c0]]
// CHECK: [[newSize:%.+]] = arith.subi [[sizeVal]], [[count]]
// CHECK-NOT: memref.realloc
// CHECK: [[elements:%.+]] = memref.load [[data]]
// CHECK: scf.for [[i:%.+]] = {{%.+}} to [[count]] step {{%.+}} {
// CHECK:   [[from:%.+]] = arith.addi [[newSize]], [[i]]
// CHECK:   [[element:%.+]] = memref.load [[elements]]{{\[}}[[from]]{{\]}}
// CHECK:   memref.store [[element]], [[dest]]
// CHECK: memref.store [[newSize]], [[size]]

// CHECK: func.func @list_pop_block([[list:%.+]]: !catalyst.arraylist<f64>, [[dest:%.+]]: memref<4xf64>)
func.func @list_pop_block(%arg0: !catalyst.arraylist<f64>, %arg1: memref<4xf64>) {
    catalyst.list_pop_block %arg0, %arg1 : !catalyst.arraylist<f64>, memref<4xf64>
    // CHECK: [[unpacked:%.+]]:3 = builtin.unrealized_conversion_cast [[list]]
    // CHECK: [[cast:%.+]] = memref.cast [[dest]] : memref<4xf64> to memref<?xf64>
    // CHECK: call @__catalyst_arraylist_pop_blockf64([[unpacked]]#0, [[unpacked]]#1, [[unpacked]]#2, [[cast]])
    return
}

// -----

// A dynamically shaped block needs no cast: it already has the type the helper takes.
// CHECK: func.func @list_push_block_dynamic([[list:%.+]]: !catalyst.arraylist<complex<f64>>, [[values:%.+]]: memref<?xcomplex<f64>>)
func.func @list_push_block_dynamic(%arg0: !catalyst.arraylist<complex<f64>>,
                                   %arg1: memref<?xcomplex<f64>>) {
    catalyst.list_push_block %arg1, %arg0 : memref<?xcomplex<f64>>, !catalyst.arraylist<complex<f64>>
    // CHECK-NOT: memref.cast
    // CHECK: call @"__catalyst_arraylist_push_blockcomplex<f64>"({{%.+}}#0, {{%.+}}#1, {{%.+}}#2, [[values]])
    return
}

// -----

// Only a bufferized, contiguous block can be lowered: the elements have to live in memory for the
// list to copy them out.
func.func @list_push_block_unbufferized(%arg0: !catalyst.arraylist<f64>, %arg1: tensor<4xf64>) {
    // expected-error @below {{expects 'elements' to be a contiguous rank-1 memref of f64 here}}
    // expected-error @below {{failed to legalize operation 'catalyst.list_push_block'}}
    catalyst.list_push_block %arg1, %arg0 : tensor<4xf64>, !catalyst.arraylist<f64>
    return
}
