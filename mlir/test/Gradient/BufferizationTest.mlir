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

// RUN: quantum-opt --gradient-bufferize --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @circuit(%arg0: f64)

func.func @adjoint(%arg0: f64, %arg1: index) {

    // CHECK:        [[alloc:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   gradient.adjoint @circuit(%arg0) size(%arg1) in([[alloc]] : memref<?xf64>) : (f64) -> ()
    %grad = gradient.adjoint @circuit(%arg0) size(%arg1) : (f64) -> tensor<?xf64>
    return
}
