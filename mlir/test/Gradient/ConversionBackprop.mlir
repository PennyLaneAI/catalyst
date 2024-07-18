// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --convert-gradient-to-llvm --canonicalize -cse --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @argmap(%arg0: memref<f64>) -> memref<?xf64>

// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_dupnoneed
// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_const
// CHECK-DAG:  llvm.func @__enzyme_autodiff0(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<f64>, memref<?xf64>)

// CHECK-LABEL: func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) -> memref<f64> {
func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) -> memref<f64> {
    // Constants and quantum gradient casting
    // CHECK-DAG:   [[memsetVal:%.+]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK-DAG:   [[c8:%.+]] = llvm.mlir.constant(8 : index) : i64
    // CHECK-DAG:   [[argmapPtr:%.+]] = constant @argmap
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : index) : i64

    // Cast the arg shadow
    // CHECK-DAG:   [[shadowCasted:%.+]] = builtin.unrealized_conversion_cast %arg1

    // More casting
    // CHECK-DAG: [[argmapCasted:%.+]] = builtin.unrealized_conversion_cast [[argmapPtr]] : (memref<f64>, memref<?xf64>) -> () to !llvm.ptr
    // CHECK-DAG: [[enzymeConst:%.+]] = llvm.mlir.addressof @enzyme_const
    // CHECK-DAG: [[enzymeDupNoNeed:%.+]] = llvm.mlir.addressof @enzyme_dupnoneed

    // Unpack the MemRef and zero out the shadow
    // CHECK:     [[arg0Casted:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK:     [[arg0Allocated:%.+]] = llvm.extractvalue [[arg0Casted]][0]
    // CHECK:     [[arg0Aligned:%.+]] = llvm.extractvalue [[arg0Casted]][1]
    // CHECK:     [[shadowAligned:%.+]] = llvm.extractvalue [[shadowCasted]][1]
    // CHECK:     [[memsetSize:%.+]] = llvm.mul [[c8]], [[c1]]
    // CHECK-DAG: "llvm.intr.memset"([[shadowAligned]], [[memsetVal]], [[memsetSize]]) <{isVolatile = false}>
    // CHECK-DAG: [[arg0Offset:%.+]] = llvm.extractvalue [[arg0Casted]][2]

    // Cast the primal result
    // CHECK: [[outputCasted:%.+]] = builtin.unrealized_conversion_cast %arg2
    // CHECK: [[outputAllocated:%.+]] = llvm.extractvalue [[outputCasted]][0]
    // CHECK: [[outputAligned:%.+]] = llvm.extractvalue [[outputCasted]][1]

    // Call Enzyme
    // CHECK:   [[qJacobianCasted:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned:%.+]] = llvm.extractvalue [[qJacobianCasted]][1]
    // CHECK-DAG: [[outputOffset:%.+]] = llvm.extractvalue [[outputCasted]][2]
    // CHECK-DAG: [[outputSize:%.+]] = llvm.extractvalue [[outputCasted]][3, 0]
    // CHECK-DAG: [[outputStride:%.+]] = llvm.extractvalue [[outputCasted]][4, 0]
    // CHECK: llvm.call @__enzyme_autodiff0([[argmapCasted]], [[enzymeConst]], [[arg0Allocated]], [[arg0Aligned]], [[shadowAligned]], [[arg0Offset]], [[enzymeConst]], [[outputAllocated]], [[enzymeDupNoNeed]], [[outputAligned]], [[qJacobianAligned]], [[outputOffset]], [[outputSize]], [[outputStride]])

    gradient.backprop @argmap(%arg0) grad_out(%arg1 : memref<f64>) callee_out(%arg2 : memref<?xf64>) cotangents(%arg3 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<f64>) -> ()

    return %arg1 : memref<f64>
}


// -----

func.func private @argmap(%arg0: memref<f64>) -> memref<?xf64>

// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_dupnoneed
// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_const
// CHECK-DAG:  llvm.func @__enzyme_autodiff0(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<f64>, memref<?xf64>)

// CHECK-LABEL: func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) -> memref<f64> {
func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) -> memref<f64> {
    // Constants and quantum gradient casting
    // CHECK-DAG:   [[memsetVal:%.+]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK-DAG:   [[c8:%.+]] = llvm.mlir.constant(8 : index) : i64
    // CHECK-DAG:   [[argmapPtr:%.+]] = constant @argmap
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : index) : i64

    // Cast the arg shadow
    // CHECK-DAG:   [[shadowCasted:%.+]] = builtin.unrealized_conversion_cast %arg1

    // More casting
    // CHECK-DAG: [[argmapCasted:%.+]] = builtin.unrealized_conversion_cast [[argmapPtr]] : (memref<f64>, memref<?xf64>) -> () to !llvm.ptr
    // CHECK-DAG: [[enzymeConst:%.+]] = llvm.mlir.addressof @enzyme_const
    // CHECK-DAG: [[enzymeDupNoNeed:%.+]] = llvm.mlir.addressof @enzyme_dupnoneed

    // Unpack the MemRef and zero out the shadow
    // CHECK:     [[arg0Casted:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK:     [[arg0Allocated:%.+]] = llvm.extractvalue [[arg0Casted]][0]
    // CHECK:     [[arg0Aligned:%.+]] = llvm.extractvalue [[arg0Casted]][1]
    // CHECK:     [[shadowAligned:%.+]] = llvm.extractvalue [[shadowCasted]][1]
    // CHECK:     [[memsetSize:%.+]] = llvm.mul [[c8]], [[c1]]
    // CHECK-DAG: "llvm.intr.memset"([[shadowAligned]], [[memsetVal]], [[memsetSize]]) <{isVolatile = false}>
    // CHECK-DAG: [[arg0Offset:%.+]] = llvm.extractvalue [[arg0Casted]][2]

    // Cast the primal result
    // CHECK: [[outputCasted:%.+]] = builtin.unrealized_conversion_cast %arg2
    // CHECK: [[outputAllocated:%.+]] = llvm.extractvalue [[outputCasted]][0]
    // CHECK: [[outputAligned:%.+]] = llvm.extractvalue [[outputCasted]][1]

    // Call Enzyme
    // CHECK:   [[qJacobianCasted:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned:%.+]] = llvm.extractvalue [[qJacobianCasted]][1]
    // CHECK-DAG: [[outputOffset:%.+]] = llvm.extractvalue [[outputCasted]][2]
    // CHECK-DAG: [[outputSize:%.+]] = llvm.extractvalue [[outputCasted]][3, 0]
    // CHECK-DAG: [[outputStride:%.+]] = llvm.extractvalue [[outputCasted]][4, 0]
    // CHECK: llvm.call @__enzyme_autodiff0([[argmapCasted]], [[enzymeConst]], [[arg0Allocated]], [[arg0Aligned]], [[shadowAligned]], [[arg0Offset]], [[enzymeConst]], [[outputAllocated]], [[enzymeDupNoNeed]], [[outputAligned]], [[qJacobianAligned]], [[outputOffset]], [[outputSize]], [[outputStride]])

    gradient.backprop @argmap(%arg0) grad_out(%arg1 : memref<f64>) callee_out(%arg2 : memref<?xf64>) cotangents(%arg3 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>, keepValueResults = false} : (memref<f64>) -> ()

    return %arg1 : memref<f64>
}


// -----

func.func private @argmap2(%arg0: memref<f64>, %arg1: f64) -> (memref<?xf64>, memref<?xf64>)

// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_dupnoneed
// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_const
// CHECK-DAG:  llvm.func @__enzyme_autodiff1(...) -> f64
// CHECK-DAG:  llvm.mlir.global external @__enzyme_function_like_free()
// CHECK-DAG:  llvm.mlir.global external @__enzyme_allocation_like()
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap2(memref<f64>, f64, memref<?xf64>, memref<?xf64>)

// CHECK-LABEL:  func.func @backpropArgmap2(%arg0: memref<f64>, %arg1: f64, %arg2: memref<f64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) -> (memref<f64>, f64)
func.func @backpropArgmap2(%arg0: memref<f64>, %arg1: f64, %arg2: memref<f64>, %arg3: memref<?xf64>, %arg4 : memref<?xf64>, %arg5 : memref<?xf64>, %arg6: memref<?xf64>) -> (memref<f64>, f64) {
    // Constants and quantum gradient casting
    // CHECK-DAG:   [[memsetVal:%.+]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK-DAG:   [[c8:%.+]] = llvm.mlir.constant(8 : index) : i64
    // CHECK-DAG:   [[argmapPtr:%.+]] = constant @argmap
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : index) : i64

    // Cast the arg shadow
    // CHECK-DAG:   [[shadowCasted:%.+]] = builtin.unrealized_conversion_cast %arg2

    // More casting
    // CHECK-DAG: [[argmapCasted:%.+]] = builtin.unrealized_conversion_cast [[argmapPtr]] : (memref<f64>, f64, memref<?xf64>, memref<?xf64>) -> () to !llvm.ptr
    // CHECK-DAG: [[enzymeConst:%.+]] = llvm.mlir.addressof @enzyme_const
    // CHECK-DAG: [[enzymeDupNoNeed:%.+]] = llvm.mlir.addressof @enzyme_dupnoneed

    // Unpack the MemRef and zero out the shadow
    // CHECK:     [[arg0Casted:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK:     [[arg0Allocated:%.+]] = llvm.extractvalue [[arg0Casted]][0]
    // CHECK:     [[arg0Aligned:%.+]] = llvm.extractvalue [[arg0Casted]][1]
    // CHECK:     [[shadowAligned:%.+]] = llvm.extractvalue [[shadowCasted]][1]
    // CHECK:     [[memsetSize:%.+]] = llvm.mul [[c8]], [[c1]]
    // CHECK-DAG: "llvm.intr.memset"([[shadowAligned]], [[memsetVal]], [[memsetSize]]) <{isVolatile = false}>
    // CHECK-DAG: [[arg0Offset:%.+]] = llvm.extractvalue [[arg0Casted]][2]

    // Cast the primal result

    // CHECK: [[outputCasted:%.+]] = builtin.unrealized_conversion_cast %arg3
    // CHECK: [[outputAllocated:%.+]] = llvm.extractvalue [[outputCasted]][0]
    // CHECK: [[outputAligned:%.+]] = llvm.extractvalue [[outputCasted]][1]

    // Cotangents
    // CHECK:   [[qJacobianCasted:%.+]] = builtin.unrealized_conversion_cast %arg5 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned:%.+]] = llvm.extractvalue [[qJacobianCasted]][1]
    // CHECK-DAG: [[outputOffset:%.+]] = llvm.extractvalue [[outputCasted]][2]
    // CHECK-DAG: [[outputSize:%.+]] = llvm.extractvalue [[outputCasted]][3, 0]
    // CHECK-DAG: [[outputStride:%.+]] = llvm.extractvalue [[outputCasted]][4, 0]

    // Cast the primal results
    // CHECK-DAG: [[outputCasted1:%.+]] = builtin.unrealized_conversion_cast %arg4
    // CHECK: [[outputAllocated1:%.+]] = llvm.extractvalue [[outputCasted1]][0]
    // CHECK: [[outputAligned1:%.+]] = llvm.extractvalue [[outputCasted1]][1]

    // Cotangents
    // CHECK:   [[qJacobianCasted1:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned1:%.+]] = llvm.extractvalue [[qJacobianCasted1]][1]
    // CHECK-DAG: [[outputOffset1:%.+]] = llvm.extractvalue [[outputCasted1]][2]
    // CHECK-DAG: [[outputSize1:%.+]] = llvm.extractvalue [[outputCasted1]][3, 0]
    // CHECK-DAG: [[outputStride1:%.+]] = llvm.extractvalue [[outputCasted1]][4, 0]

    // CHECK: [[res:%.+]] = llvm.call @__enzyme_autodiff1([[argmapCasted]], [[enzymeConst]], [[arg0Allocated]], [[arg0Aligned]], [[shadowAligned]], [[arg0Offset]], %arg1, [[enzymeConst]], [[outputAllocated]], [[enzymeDupNoNeed]], [[outputAligned]], [[qJacobianAligned]], [[outputOffset]], [[outputSize]], [[outputStride]], [[enzymeConst]], [[outputAllocated1]], [[enzymeDupNoNeed]], [[outputAligned1]], [[qJacobianAligned1]], [[outputOffset1]], [[outputSize1]], [[outputStride1]])
    %res = gradient.backprop @argmap2(%arg0, %arg1) grad_out(%arg2: memref<f64>) callee_out(%arg3, %arg4 : memref<?xf64>, memref<?xf64>) cotangents(%arg5, %arg6 : memref<?xf64>, memref<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, keepValueResults = false, resultSegmentSizes = array<i32: 0, 1>} : (memref<f64>, f64) -> f64
    // CHECK: return %arg2, [[res]]
    return %arg2, %res: memref<f64>, f64
}


// -----

func.func private @argmap3(%arg0: memref<f64>, %arg1: f64) -> (memref<?xf64>, memref<?xf64>)

// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_dupnoneed
// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_const
// CHECK-DAG:  llvm.func @__enzyme_autodiff1(...) -> f64
// CHECK-DAG:  llvm.mlir.global external @__enzyme_function_like_free()
// CHECK-DAG:  llvm.mlir.global external @__enzyme_allocation_like()
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap3(memref<f64>, f64, memref<?xf64>, memref<?xf64>)

// CHECK-LABEL:  func.func @backpropArgmap3(%arg0: memref<f64>, %arg1: f64, %arg2: memref<f64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) -> (memref<f64>, f64)
func.func @backpropArgmap3(%arg0: memref<f64>, %arg1: f64, %arg2: memref<f64>, %arg3: memref<?xf64>, %arg4 : memref<?xf64>, %arg5 : memref<?xf64>, %arg6: memref<?xf64>) -> (memref<f64>, f64) {
    // Constants and quantum gradient casting
    // CHECK-DAG:   [[memsetVal:%.+]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK-DAG:   [[c8:%.+]] = llvm.mlir.constant(8 : index) : i64
    // CHECK-DAG:   [[argmapPtr:%.+]] = constant @argmap
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : index) : i64

    // Cast the arg shadow
    // CHECK-DAG:   [[shadowCasted:%.+]] = builtin.unrealized_conversion_cast %arg2

    // More casting
    // CHECK-DAG: [[argmapCasted:%.+]] = builtin.unrealized_conversion_cast [[argmapPtr]] : (memref<f64>, f64, memref<?xf64>, memref<?xf64>) -> () to !llvm.ptr
    // CHECK-DAG: [[enzymeConst:%.+]] = llvm.mlir.addressof @enzyme_const

    // Unpack the MemRef and zero out the shadow
    // CHECK: [[arg0Casted:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[arg0Allocated:%.+]] = llvm.extractvalue [[arg0Casted]][0]
    // CHECK: [[arg0Aligned:%.+]] = llvm.extractvalue [[arg0Casted]][1]
    // CHECK: [[shadowAligned:%.+]] = llvm.extractvalue [[shadowCasted]][1]
    // CHECK: [[memsetSize:%.+]] = llvm.mul [[c8]], [[c1]]
    // CHECK-DAG: "llvm.intr.memset"([[shadowAligned]], [[memsetVal]], [[memsetSize]]) <{isVolatile = false}>
    // CHECK-DAG: [[arg0Offset:%.+]] = llvm.extractvalue [[arg0Casted]][2]

    // Cast the primal result
    // CHECK-DAG: [[outputCasted:%.+]] = builtin.unrealized_conversion_cast %arg3
    // CHECK: [[outputAllocated:%.+]] = llvm.extractvalue [[outputCasted]][0]
    // CHECK: [[outputAligned:%.+]] = llvm.extractvalue [[outputCasted]][1]

    // Cotangents
    // CHECK:   [[qJacobianCasted:%.+]] = builtin.unrealized_conversion_cast %arg5 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned:%.+]] = llvm.extractvalue [[qJacobianCasted]][1]
    // CHECK-DAG: [[outputOffset:%.+]] = llvm.extractvalue [[outputCasted]][2]
    // CHECK-DAG: [[outputSize:%.+]] = llvm.extractvalue [[outputCasted]][3, 0]
    // CHECK-DAG: [[outputStride:%.+]] = llvm.extractvalue [[outputCasted]][4, 0]

    // Cast the primal results
    // CHECK-DAG: [[outputCasted1:%.+]] = builtin.unrealized_conversion_cast %arg4
    // CHECK: [[outputAllocated1:%.+]] = llvm.extractvalue [[outputCasted1]][0]
    // CHECK: [[outputAligned1:%.+]] = llvm.extractvalue [[outputCasted1]][1]

    // Cotangents
    // CHECK:   [[qJacobianCasted1:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-DAG: [[qJacobianAligned1:%.+]] = llvm.extractvalue [[qJacobianCasted1]][1]
    // CHECK-DAG: [[outputOffset1:%.+]] = llvm.extractvalue [[outputCasted1]][2]
    // CHECK-DAG: [[outputSize1:%.+]] = llvm.extractvalue [[outputCasted1]][3, 0]
    // CHECK-DAG: [[outputStride1:%.+]] = llvm.extractvalue [[outputCasted1]][4, 0]

    // CHECK: [[res:%.+]] = llvm.call @__enzyme_autodiff1([[argmapCasted]], [[enzymeConst]], [[arg0Allocated]], [[arg0Aligned]], [[shadowAligned]], [[arg0Offset]], %arg1, [[enzymeConst]], [[outputAllocated]], [[outputAligned]], [[qJacobianAligned]], [[outputOffset]], [[outputSize]], [[outputStride]], [[enzymeConst]], [[outputAllocated1]], [[outputAligned1]], [[qJacobianAligned1]], [[outputOffset1]], [[outputSize1]], [[outputStride1]])
    %res = gradient.backprop @argmap3(%arg0, %arg1) grad_out(%arg2: memref<f64>) callee_out(%arg3, %arg4 : memref<?xf64>, memref<?xf64>) cotangents(%arg5, %arg6 : memref<?xf64>, memref<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, keepValueResults = true, resultSegmentSizes = array<i32: 0, 1>} : (memref<f64>, f64) -> f64
    // CHECK: return %arg2, [[res]]
    return %arg2, %res: memref<f64>, f64
}
