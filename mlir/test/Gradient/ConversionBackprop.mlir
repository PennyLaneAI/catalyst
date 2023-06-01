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

// RUN: quantum-opt --convert-gradient-to-llvm --expand-strided-metadata --convert-memref-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

// CHECK-DAG: llvm.func @__enzyme_autodiff(...)
func.func private @argmap(%arg0: memref<f64>) -> (memref<?xf64>)


func.func @backpropArgmap(%arg0: memref<f64>, %arg1 : index, %arg2: memref<?xf64>) -> memref<?xf64> {

    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    gradient.backprop @argmap(%arg0) size(%arg1) qjacobians(%arg2: memref<?xf64>) in(%alloc0 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<f64>) -> ()

    return %alloc0: memref<?xf64>
}

// -----

func.func private @argmap(%arg0: memref<1xf64>) -> (memref<?xf64>)


func.func @backpropArgmap2(%arg0: memref<1xf64>, %arg1 : index, %arg2: memref<?xf64>) -> memref<?xf64> {
    
    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    gradient.backprop @argmap(%arg0) size(%arg1) qjacobians(%arg2: memref<?xf64>) in(%alloc0 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<1xf64>) -> ()

    return %alloc0: memref<?xf64>
}