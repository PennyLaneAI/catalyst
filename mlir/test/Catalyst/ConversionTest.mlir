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

//////////////////////////
// Catalyst AssertionOp //
//////////////////////////

// CHECK-DAG: llvm.mlir.global internal constant @[[hash:["0-9]+]]("Test Message")
// CHECK-DAG: llvm.func @__catalyst__rt__assert_bool(i1, !llvm.ptr)

func.func @assert_constant(%arg0: i1, %arg1: !llvm.ptr) {
    // CHECK: [[array_ptr:%.+]] = llvm.mlir.addressof @[[hash]] : !llvm.ptr
    // CHECK: [[char_ptr:%.+]] = llvm.getelementptr inbounds [[array_ptr]][0, 0] : {{.*}} -> !llvm.ptr, !llvm.array<12 x i8>
    // CHECK: llvm.call @__catalyst__rt__assert_bool(%arg0, [[char_ptr]])
    "catalyst.assert"(%arg0) <{error = "Test Message"}> : (i1) -> ()
    
    return
}

// -----

//////////////////////
// Catalyst PrintOp //
//////////////////////

// CHECK-DAG: llvm.func @__catalyst__rt__print_tensor(!llvm.ptr, i1)

// CHECK-LABEL: @dbprint_val
func.func @dbprint_val(%arg0 : memref<1xi64>) {
    // CHECK: [[memref:%.+]] = builtin.unrealized_conversion_cast %arg0

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca2:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca3:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>


    // CHECK: [[typeEnc:%.+]] = llvm.mlir.constant(5 : i8)
    // CHECK: [[rank:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[struct:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[struct0:%.+]] = llvm.insertvalue [[rank]], [[struct]][0]

    // CHECK: llvm.store [[memref]], [[alloca3]]

    // CHECK: [[struct1:%.+]] = llvm.insertvalue [[alloca3]], [[struct0]][1]
    // CHECK: [[struct2:%.+]] = llvm.insertvalue [[typeEnc]], [[struct1]][2]
    // CHECK: llvm.store [[struct2]], [[alloca2]]

    // CHECK: [[memref_flag:%.+]] = llvm.mlir.constant(false)
    // CHECK: llvm.call @__catalyst__rt__print_tensor([[alloca2]], [[memref_flag]])
    "catalyst.print"(%arg0) : (memref<1xi64>) -> ()

    // CHECK: [[memref_flag2:%.+]] = llvm.mlir.constant(true)
    // CHECK: llvm.call @__catalyst__rt__print_tensor({{%.+}}, [[memref_flag2]])
    "catalyst.print"(%arg0) {print_descriptor} : (memref<1xi64>) -> ()

    return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @[[hash:["0-9]+]]("Hello, Catalyst")
// CHECK-DAG: llvm.func @__catalyst__rt__print_string(!llvm.ptr)

// CHECK-LABEL: @dbprint_str
func.func @dbprint_str() {

    // CHECK: [[array_ptr:%.+]] = llvm.mlir.addressof @[[hash]] : !llvm.ptr
    // CHECK: [[char_ptr:%.+]] = llvm.getelementptr inbounds [[array_ptr]][0, 0] : {{.*}} -> !llvm.ptr, !llvm.array<15 x i8>
    // CHECK: llvm.call @__catalyst__rt__print_string([[char_ptr]])
    "catalyst.print"() {const_val = "Hello, Catalyst"} : () -> ()

    return
}

// -----

// CHECK-LABEL: @custom_call
func.func @custom_call(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) -> () {

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x !llvm.array<1 x ptr>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca2:%.+]] = llvm.alloca [[c1]] x !llvm.array<1 x ptr>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca3:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)>

    // CHECK: llvm.call @lapack_dgesdd([[alloca2]], [[alloca0]])
    catalyst.custom_call fn("lapack_dgesdd") (%arg0, %arg1) {number_original_arg = array<i32: 1>} : (memref<3x3xf64>, memref<3x3xf64>) -> ()
    return 
}

// -----

// CHECK-LABEL: @test0
module @test0 {

  // Make sure that arguments are !llvm.ptrs.
  // CHECK-LABEL: func.func private @callback_4(
  // CHECK-SAME: [[arg0:%.+]]: !llvm.ptr,
  // CHECK-SAME: [[arg1:%.+]]: !llvm.ptr)
  // CHECK-SAME: noinline

  // Make sure that we pass the constants that we need.
  // CHECK-DAG: [[id:%.+]] = llvm.mlir.constant(4
  // CHECK-DAG: [[argc:%.+]] = llvm.mlir.constant(2
  // CHECK-DAG: [[resc:%.+]] = llvm.mlir.constant(3

  // CHECK: llvm.call @__catalyst_inactive_callback([[id]], [[argc]], [[resc]]
  catalyst.callback @callback_4(memref<f64>, memref<f64>) attributes {argc = 2 : i64, id = 4 : i64, resc = 3 : i64}
}

// -----

// CHECK-LABEL: @test1
module @test1 {
  catalyst.callback @callback_1(memref<f64>, memref<f64>) attributes {argc = 1 : i64, id = 1 : i64, resc = 1 : i64}
  // CHECK: __catalyst_inactive_callback(i64, i64, i64, ...) attributes {passthrough = ["nofree"]}
  // CHECK-LABEL: func.func private @foo(
  // CHECK-SAME: [[arg0:%.+]]: tensor<f64>
  // CHECK-SAME:)
  func.func private @foo(%arg0: tensor<f64>) -> tensor<f64> {
    // CHECK: [[memref0:%.+]] = bufferization.to_memref [[arg0]]
    // CHECK: [[ptr0:%.+]] = llvm.alloca {{.*}}
    // CHECK: [[ptr1:%.+]] = llvm.alloca {{.*}}
    // CHECK: [[struct0:%.+]] = builtin.unrealized_conversion_cast [[memref0]]

    // CHECK: [[tensor1:%.+]] = bufferization.alloc_tensor()
    // CHECK: [[memref1:%.+]] = bufferization.to_memref [[tensor1]]
    // CHECK: [[struct1:%.+]] = builtin.unrealized_conversion_cast [[memref1]]

    // CHECK: llvm.store [[struct0]], [[ptr1]]
    // CHECK: llvm.store [[struct1]], [[ptr0]]

    // call @callback_1([[ptr0]], [[ptr1]])

    %0 = bufferization.to_memref %arg0 : memref<f64>
    %1 = bufferization.alloc_tensor() {memory_space = 0 : i64} : tensor<f64>
    %2 = bufferization.to_memref %1 : memref<f64>


    catalyst.callback_call @callback_1(%0, %2) : (memref<f64>, memref<f64>) -> ()
    %3 = bufferization.to_tensor %2 : memref<f64>
    return %3 : tensor<f64>
  }
}

// -----

// CHECK-LABEL: @test2
module @test2 {

  func.func private @fwd.fwd(memref<f64>, memref<f64>, memref<f64>, memref<f64>) -> memref<f64>
  func.func private @bwd.rev(memref<f64>, memref<f64>, memref<f64>, memref<f64>)
  func.func private @foo(memref<f64>, memref<f64>)

  // CHECK-LABEL: llvm.mlir.global external @__enzyme_register_gradient_foo
  // CHECK-DAG: [[foo:%.+]] = func.constant @foo
  // CHECK-DAG: [[fwd:%.+]] = func.constant @fwd.fwd
  // CHECK-DAG: [[rev:%.+]] = func.constant @bwd.rev

  gradient.custom_grad @foo @fwd.fwd @bwd.rev
}
