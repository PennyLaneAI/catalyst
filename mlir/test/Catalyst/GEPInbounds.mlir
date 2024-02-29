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

// RUN: quantum-opt --gep-inbounds --split-input-file %s | FileCheck %s

llvm.mlir.global private constant @__constant_2xi32(dense<2> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>

func.func @func(%arg0: !llvm.ptr, %arg1: i64) -> (!llvm.ptr, !llvm.ptr, !llvm.ptr) {
    %0 = llvm.mlir.addressof @__constant_2xi32 : !llvm.ptr
    // CHECK:    llvm.getelementptr inbounds
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
    %2 = llvm.mlir.zero : !llvm.ptr
    // CHECK-NOT:    llvm.getelementptr inbounds
    %3 = llvm.getelementptr %2[2] : (!llvm.ptr) -> !llvm.ptr, i32
    // CHECK:    llvm.getelementptr inbounds
    %4 = llvm.getelementptr inbounds %arg0[%arg1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    return %1, %3, %4: !llvm.ptr, !llvm.ptr, !llvm.ptr
}