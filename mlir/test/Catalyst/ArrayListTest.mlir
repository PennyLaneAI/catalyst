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
