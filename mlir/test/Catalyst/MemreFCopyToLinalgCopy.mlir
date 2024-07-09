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

// RUN: quantum-opt --memrefcpy-to-linalgcpy --split-input-file %s | FileCheck %s


func.func private @runtime_memref(%arg0: memref<5x4xf64>) -> memref<5xf64>{
    %subview = memref.subview %arg0[0, 0] [5, 1] [1, 1] : memref<5x4xf64> to memref<5x1xf64, strided<[4, 1]>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<5x1xf64, strided<[4, 1]>> into memref<5xf64, strided<[4]>>
    %alloc = memref.alloc() : memref<5xf64>
    memref.copy %collapse_shape, %alloc : memref<5xf64, strided<[4]>> to memref<5xf64>
    return %alloc : memref<5xf64>
}