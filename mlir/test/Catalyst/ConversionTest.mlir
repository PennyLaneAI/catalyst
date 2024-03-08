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

// CHECK-DAG: llvm.func @__catalyst__rt__print_tensor(!llvm.ptr, i1)

// CHECK-LABEL: @dbprint_val
func.func @dbprint_val(%arg0 : memref<1xi64>) {
    // CHECK: [[memref:%.+]] = builtin.unrealized_conversion_cast %arg0
    // CHECK: [[typeEnc:%.+]] = llvm.mlir.constant(5 : i8)
    // CHECK: [[rank:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[struct:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr, i8)>

    // CHECK: [[struct0:%.+]] = llvm.insertvalue [[rank]], [[struct]][0]

    // CHECK: [[memref_ptr:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.store [[memref]], [[memref_ptr]]
    // CHECK: [[struct1:%.+]] = llvm.insertvalue [[memref_ptr]], [[struct0]][1]

    // CHECK: [[struct2:%.+]] = llvm.insertvalue [[typeEnc]], [[struct1]][2]

    // CHECK: [[struct_ptr:%.+]] = llvm.alloca {{.*}} : (i64) -> !llvm.ptr
    // CHECK: llvm.store [[struct2]], [[struct_ptr]]
    // CHECK: [[memref_flag:%.+]] = llvm.mlir.constant(false)
    // CHECK: llvm.call @__catalyst__rt__print_tensor([[struct_ptr]], [[memref_flag]])
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

func.func @custom_call(%arg0: memref<3x3xf64>) -> memref<3x3xf64> {
    // CHECK: [[convertedArg:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<3x3xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: [[alloc:%.+]] = memref.alloc() : memref<3x3xf64>
    // CHECK-NEXT: [[allocConverted:%.+]] = builtin.unrealized_conversion_cast [[alloc]] : memref<3x3xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: [[c1:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: [[numericTypeArg:%.+]] = llvm.mlir.constant(7 : i8) : i8
    // CHECK-NEXT: [[c2:%.+]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT: [[undef:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr, i8)>
    // CHECK-NEXT: [[encodeRank:%.+]] = llvm.insertvalue [[c2]], [[undef]][0] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[c0:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: [[argStruct:%.+]] = llvm.extractvalue [[convertedArg]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK-NEXT: [[getPtr:%.+]] = llvm.getelementptr inbounds [[argStruct]][[[c0]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    // CHECK-NEXT: [[encodeData:%.+]] = llvm.insertvalue [[getPtr]], [[encodeRank]][1] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[encodeType:%.+]] = llvm.insertvalue [[numericTypeArg]], [[encodeData]][2] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[alloca:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)> : (i64) -> !llvm.ptr
    // CHECK-NEXT: llvm.store [[encodeType]], [[alloca]] : !llvm.struct<(i64, ptr, i8)>, !llvm.ptr
    // CHECK-NEXT: [[undefPtr:%.+]] = llvm.mlir.undef : !llvm.array<1 x ptr>
    // CHECK-NEXT: [[allocaInserted:%.+]] = llvm.insertvalue [[alloca]], [[undefPtr]][0] : !llvm.array<1 x ptr> 
    // CHECK-NEXT: [[allocaArray:%.+]] = llvm.alloca [[c1]] x !llvm.array<1 x ptr> : (i64) -> !llvm.ptr
    // CHECK-NEXT: llvm.store [[allocaInserted]], [[allocaArray]] : !llvm.array<1 x ptr>, !llvm.ptr
    // CHECK-NEXT: [[numericTypeRes:%.+]] = llvm.mlir.constant(7 : i8) : i8
    // CHECK-NEXT: [[c2_1:%.+]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT: [[undef1_1:%.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr, i8)>
    // CHECK-NEXT: [[encodedRes:%.+]] = llvm.insertvalue [[c2_1]], [[undef1_1]][0] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[c0:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: [[resStruct:%.+]] = llvm.extractvalue [[allocConverted]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK-NEXT: [[getPtrRes:%.+]] = llvm.getelementptr inbounds [[resStruct]][[[c0]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    // CHECK-NEXT: [[encodedResData:%.+]] = llvm.insertvalue [[getPtrRes]], [[encodedRes]][1] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[encodedResType:%.+]] = llvm.insertvalue [[numericTypeRes]], [[encodedResData]][2] : !llvm.struct<(i64, ptr, i8)> 
    // CHECK-NEXT: [[allocaRes:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)> : (i64) -> !llvm.ptr
    // CHECK: llvm.store [[encodedResType]], [[allocaRes]] : !llvm.struct<(i64, ptr, i8)>, !llvm.ptr
    // CHECK: [[arrayRes:%.+]] = llvm.mlir.undef : !llvm.array<1 x ptr>
    // CHECK: [[allocaInsertedRes:%.+]] = llvm.insertvalue [[allocaRes]], [[arrayRes]][0] : !llvm.array<1 x ptr> 
    // CHECK: [[allocaArrayRes:%.+]] = llvm.alloca [[c1]] x !llvm.array<1 x ptr> : (i64) -> !llvm.ptr
    // CHECK: llvm.store [[allocaInsertedRes]], [[allocaArrayRes]] : !llvm.array<1 x ptr>, !llvm.ptr
    // CHECK: llvm.call @lapack_dgesdd([[allocaArray]], [[allocaArrayRes]]) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: return [[alloc]] : memref<3x3xf64>

    %alloc = memref.alloc() : memref<3x3xf64>
    catalyst.custom_call fn("lapack_dgesdd") (%arg0, %alloc) {number_original_arg = array<i32: 1>} : (memref<3x3xf64>, memref<3x3xf64>) -> ()
    return %alloc: memref<3x3xf64>
}

// -----

// A python without parameters and without returns.

func.func @python_call () {
    // CHECK: [[identifier:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: [[argcount:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @pyregistry([[identifier]], [[argcount]])
    catalyst.pycallback() { identifier = 0 } : () -> ()
    return
}
