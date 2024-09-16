// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --memref-to-llvm-tbaa --split-input-file %s | FileCheck %s

// Test that no tbaa metadata is inserted if we are not taking the gradient of a function

// CHECK-NOT: = #llvm.tbaa_root
module @my_model {
    func.func @func_noenzyme_notbaa(%arg0: memref<i32>, %arg1: memref<4xi32>) -> (memref<i32>, memref<4xi32>) {
        %0 = memref.load %arg0[] : memref<i32>
        %idx0 = index.constant 0
        memref.store %0, %arg1[%idx0] : memref<4xi32>
        return %arg0, %arg1: memref<i32>, memref<4xi32>
    }
}

// -----

// CHECK: [[root:#.+]] = #llvm.tbaa_root<id = "Catalyst TBAA">
// CHECK: [[typedesc:#.+]] = #llvm.tbaa_type_desc<id = "int", members = {<[[root]], 0>}>
// CHECK: [[tag:#.+]] = #llvm.tbaa_tag<base_type = [[typedesc]], access_type = [[typedesc]], offset = 0>
module @my_model {
    llvm.func @__enzyme_autodiff0(...)
    func.func @func_i32(%arg0: memref<i32>, %arg1: memref<4xi32>) -> (memref<i32>, memref<4xi32>) {
        // CHECK: [[castArg0:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<i32> to !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[castArg1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<4xi32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[extract0:%.+]] = llvm.extractvalue [[castArg0]][1] : !llvm.struct<(ptr, ptr, i64)> 
        // CHECK: [[load:%.+]] = llvm.load [[extract0]] {tbaa = [[[tag]]]} : !llvm.ptr -> i32
        // CHECK: [[idx:%.+]] = index.constant 0
        // CHECK: [[idxCast:%.+]] = builtin.unrealized_conversion_cast [[idx]] : index to i64
        // CHECK: [[extract1:%.+]] = llvm.extractvalue [[castArg1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        // CHECK: [[getPtr:%.+]] = llvm.getelementptr [[extract1]][[[idxCast]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        // CHECK: llvm.store [[load]], [[getPtr]] {tbaa = [[[tag]]]} : i32, !llvm.ptr
        %0 = memref.load %arg0[] : memref<i32>
        %idx0 = index.constant 0
        memref.store %0, %arg1[%idx0] : memref<4xi32>
        return %arg0, %arg1: memref<i32>, memref<4xi32>
    }
}

// -----

// CHECK: [[root:#.+]] = #llvm.tbaa_root<id = "Catalyst TBAA">
// CHECK: [[typedesc:#.+]] = #llvm.tbaa_type_desc<id = "float", members = {<[[root]], 0>}>
// CHECK: [[tag:#.+]] = #llvm.tbaa_tag<base_type = [[typedesc]], access_type = [[typedesc]], offset = 0>
module @my_model {
    llvm.func @__enzyme_autodiff1(...)
    func.func @func_f32(%arg0: memref<f32>, %arg1: memref<4xf32>) -> (memref<f32>, memref<4xf32>) {
        // CHECK: [[castArg0:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f32> to !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[castArg1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<4xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[extract0:%.+]] = llvm.extractvalue [[castArg0]][1] : !llvm.struct<(ptr, ptr, i64)> 
        // CHECK: [[load:%.+]] = llvm.load [[extract0]] {tbaa = [[[tag]]]} : !llvm.ptr -> f32
        // CHECK: [[idx:%.+]] = index.constant 0
        // CHECK: [[idxCast:%.+]] = builtin.unrealized_conversion_cast [[idx]] : index to i64
        // CHECK: [[extract1:%.+]] = llvm.extractvalue [[castArg1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        // CHECK: [[getPtr:%.+]] = llvm.getelementptr [[extract1]][[[idxCast]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        // CHECK: llvm.store [[load]], [[getPtr]] {tbaa = [[[tag]]]} : f32, !llvm.ptr
        %0 = memref.load %arg0[] : memref<f32>
        %idx0 = index.constant 0
        memref.store %0, %arg1[%idx0] : memref<4xf32>
        return %arg0, %arg1: memref<f32>, memref<4xf32>
    }
}

// -----

// CHECK: [[root:#.+]] = #llvm.tbaa_root<id = "Catalyst TBAA">
// CHECK: [[typedesc:#.+]] = #llvm.tbaa_type_desc<id = "double", members = {<[[root]], 0>}>
// CHECK: [[tag:#.+]] = #llvm.tbaa_tag<base_type = [[typedesc]], access_type = [[typedesc]], offset = 0>
module @my_model {
    llvm.func @__enzyme_autodiff1(...)
    func.func @func_f64(%arg0: memref<f64>, %arg1: memref<4xf64>) -> (memref<f64>, memref<4xf64>) {
        // CHECK: [[castArg0:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[castArg1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[extract0:%.+]] = llvm.extractvalue [[castArg0]][1] : !llvm.struct<(ptr, ptr, i64)> 
        // CHECK: [[load:%.+]] = llvm.load [[extract0]] {tbaa = [[[tag]]]} : !llvm.ptr -> f64
        // CHECK: [[idx:%.+]] = index.constant 0
        // CHECK: [[idxCast:%.+]] = builtin.unrealized_conversion_cast [[idx]] : index to i64
        // CHECK: [[extract1:%.+]] = llvm.extractvalue [[castArg1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        // CHECK: [[getPtr:%.+]] = llvm.getelementptr [[extract1]][[[idxCast]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        // CHECK: llvm.store [[load]], [[getPtr]] {tbaa = [[[tag]]]} : f64, !llvm.ptr
        %0 = memref.load %arg0[] : memref<f64>
        %idx0 = index.constant 0
        memref.store %0, %arg1[%idx0] : memref<4xf64>
        return %arg0, %arg1: memref<f64>, memref<4xf64>
    }
}

// -----

// CHECK: [[root:#.+]] = #llvm.tbaa_root<id = "Catalyst TBAA">
// CHECK: [[typedescdouble:#.+]] = #llvm.tbaa_type_desc<id = "double", members = {<[[root]], 0>}>
// CHECK: [[typedescint:#.+]] = #llvm.tbaa_type_desc<id = "int", members = {<[[root]], 0>}>
// CHECK: [[tagdouble:#.+]] = #llvm.tbaa_tag<base_type = [[typedescdouble]], access_type = [[typedescdouble]], offset = 0>
// CHECK: [[tagint:#.+]] = #llvm.tbaa_tag<base_type = [[typedescint]], access_type = [[typedescint]], offset = 0>
module @my_model {
    llvm.func @__enzyme_autodiff2(...)
    func.func @func_mix_f64_index(%arg0: memref<f64>, %arg1: memref<4xf64>, %arg2: memref<index>, %arg3: memref<3xindex>) -> (memref<4xf64>, memref<3xindex>) {
        // CHECK: [[castArg0:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[castArg1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[castArg2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<index> to !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[castArg3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<3xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[extract0:%.+]] = llvm.extractvalue [[castArg0]][1] : !llvm.struct<(ptr, ptr, i64)> 
        // CHECK: [[load0:%.+]] = llvm.load [[extract0]] {tbaa = [[[tagdouble]]]} : !llvm.ptr -> f64
        // CHECK: [[idx:%.+]] = index.constant 0
        // CHECK: [[idxCast:%.+]] = builtin.unrealized_conversion_cast [[idx]] : index to i64
        // CHECK: [[extract1:%.+]] = llvm.extractvalue [[castArg1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[getPtr:%.+]] = llvm.getelementptr [[extract1]][[[idxCast]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        // CHECK: llvm.store [[load0]], [[getPtr]] {tbaa = [[[tagdouble]]]} : f64, !llvm.ptr
        // CHECK: [[extract2:%.+]] = llvm.extractvalue [[castArg2]][1] : !llvm.struct<(ptr, ptr, i64)>
        // CHECK: [[load1:%.+]] = llvm.load [[extract2]] {tbaa = [[[tagint]]]} : !llvm.ptr -> i64
        // CHECK: [[idx1:%.+]] = index.constant 1
        // CHECK: [[idxCast1:%.+]] = builtin.unrealized_conversion_cast [[idx1]] : index to i64
        // CHECK: [[extract2:%.+]] = llvm.extractvalue [[castArg3]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[getPtr1:%.+]] = llvm.getelementptr [[extract2]][[[idxCast1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        // CHECK: llvm.store [[load1]], [[getPtr1]] {tbaa = [[[tagint]]]} : i64, !llvm.ptr
        %0 = memref.load %arg0[] : memref<f64>
        %idx0 = index.constant 0
        memref.store %0, %arg1[%idx0] : memref<4xf64>
        %1 = memref.load %arg2[] : memref<index>
        %idx1 = index.constant 1
        memref.store %1, %arg3[%idx1] : memref<3xindex>
        return %arg1, %arg3: memref<4xf64>, memref<3xindex>
    }
}
