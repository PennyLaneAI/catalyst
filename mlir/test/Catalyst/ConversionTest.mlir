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

// RUN: quantum-opt --finalize-memref-to-llvm --catalyst-bufferize --convert-catalyst-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

// CHECK: llvm.func @_catalyst_memref_print(!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>)

// CHECK-LABEL: @dbprint_val
func.func @dbprint_val(%arg0 : tensor<1xi64>) {

    // CHECK: llvm.call @_catalyst_memref_print({{.*}}) : (!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    "catalyst.print"(%arg0) : (tensor<1xi64>) -> ()

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
