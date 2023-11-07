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

// RUN: quantum-opt --convert-catalyst-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

// CHECK: llvm.func @__quantum__rt__print_tensor(!llvm.ptr<struct<(i64, ptr, i8)>>, i1)

// CHECK-LABEL: @dbprint_val
func.func @dbprint_val(%arg0 : memref<1xi64>) {
    // CHECK: [[memref:%.+]] = builtin.unrealized_conversion_cast %arg0

    // CHECK: [[struct:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[rank:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[struct0:%.+]] = llvm.insertvalue [[rank]], [[struct]][0]

    // CHECK: [[memref_ptr:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.store [[memref]], [[memref_ptr]]
    // CHECK: [[memref_ptr_cast:%.+]] = llvm.bitcast [[memref_ptr]] : {{.*}} to !llvm.ptr
    // CHECK: [[struct1:%.+]] = llvm.insertvalue [[memref_ptr_cast]], [[struct0]][1]

    // CHECK: [[typeEnc:%.+]] = llvm.mlir.constant(5 : i8)
    // CHECK: [[struct2:%.+]] = llvm.insertvalue [[typeEnc]], [[struct1]][2]

    // CHECK: [[struct_ptr:%.+]] = llvm.alloca {{.*}} -> !llvm.ptr<struct<(i64, ptr, i8)>>
    // CHECK: llvm.store [[struct2]], [[struct_ptr]]
    // CHECK: [[memref_flag:%.+]] = llvm.mlir.constant(false)
    // CHECK: llvm.call @__quantum__rt__print_tensor([[struct_ptr]], [[memref_flag]])
    "catalyst.print"(%arg0) : (memref<1xi64>) -> ()

    // CHECK: [[memref_flag2:%.+]] = llvm.mlir.constant(true)
    // CHECK: llvm.call @__quantum__rt__print_tensor({{%.+}}, [[memref_flag2]])
    "catalyst.print"(%arg0) {print_descriptor} : (memref<1xi64>) -> ()

    return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @[[hash:["0-9]+]]("Hello, Catalyst")
// CHECK-DAG: llvm.func @__quantum__rt__print_string(!llvm.ptr<i8>)

// CHECK-LABEL: @dbprint_str
func.func @dbprint_str() {

    // CHECK: [[C0:%.+]] = llvm.mlir.constant(0 : index)
    // CHECK: [[array_ptr:%.+]] = llvm.mlir.addressof @[[hash]] : !llvm.ptr<array<15 x i8>>
    // CHECK: [[char_ptr:%.+]] = llvm.getelementptr [[array_ptr]][[[C0]], [[C0]]] : {{.*}} -> !llvm.ptr<i8>
    // CHECK: llvm.call @__quantum__rt__print_string([[char_ptr]])
    "catalyst.print"() {const_val = "Hello, Catalyst"} : () -> ()

    return
}
