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

// RUN: quantum-opt --quantum-bufferize --split-input-file %s | FileCheck %s

//////////////////
// Measurements //
//////////////////

func.func @counts(%q0: !quantum.bit, %q1: !quantum.bit, %dyn_shots: i64) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs
    %shots = arith.constant 2 : i64

    // CHECK: [[shots:%.+]] = arith.constant 2 : i64
    // CHECK: [[eigval_alloc:%.+]] = memref.alloc() : memref<4xf64>
    // CHECK: [[counts_alloc:%.+]] = memref.alloc() : memref<4xi64>
    // CHECK: quantum.counts {{.*}} in([[eigval_alloc]] : memref<4xf64>, [[counts_alloc]] : memref<4xi64>) [[shots]]
    %samples:2 = quantum.counts %obs %shots : tensor<4xf64>, tensor<4xi64>

    // CHECK: [[dyn_eigval_alloc:%.+]] = memref.alloc() : memref<4xf64>
    // CHECK: [[dyn_counts_alloc:%.+]] = memref.alloc() : memref<4xi64>
    // CHECK: quantum.counts {{.*}} in([[dyn_eigval_alloc]] : memref<4xf64>, [[dyn_counts_alloc]] : memref<4xi64>) {{.*}}
    %dyn_samples:2 = quantum.counts %obs %dyn_shots : tensor<4xf64>, tensor<4xi64>

    func.return
}

// -----

func.func @sample(%q0: !quantum.bit, %q1: !quantum.bit, %dyn_shots: i64) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs
    %shots = arith.constant 1000 : i64

    // CHECK: [[shots:%.+]] = arith.constant 1000 : i64
    // CHECK: [[alloc:%.+]] = memref.alloc() : memref<1000x2xf64>
    // CHECK: quantum.sample {{.*}} in([[alloc]] : memref<1000x2xf64>) [[shots]]
    %samples = quantum.sample %obs %shots : tensor<1000x2xf64>

    // CHECK: [[idx:%.+]] = index.casts {{.*}} : i64 to index
    // CHECK: [[dyn_alloc:%.+]] = memref.alloc([[idx]]) : memref<?x2xf64>
    // CHECK: quantum.sample {{.*}} in([[dyn_alloc]] : memref<?x2xf64>) {{.*}}
    %dyn_samples = quantum.sample %obs %dyn_shots : tensor<?x2xf64>

    func.return
}

// -----

func.func @probs(%q0: !quantum.bit, %q1: !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs
    // CHECK: [[alloc:%.+]] = memref.alloc() : memref<4xf64>
    // CHECK: quantum.probs {{.*}} in([[alloc]] : memref<4xf64>)
    %probs = quantum.probs %obs : tensor<4xf64>
    func.return
}

// -----

func.func @state(%q0: !quantum.bit, %q1: !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs
    // CHECK: [[alloc:%.+]] = memref.alloc() : memref<4xcomplex<f64>>
    // CHECK: quantum.state {{.*}} in([[alloc]] : memref<4xcomplex<f64>>)
    %state = quantum.state %obs : tensor<4xcomplex<f64>>
    func.return
}

// -----

// CHECK-LABEL: @set_state
module @set_state {
  func.func @foo(%arg0: tensor<2xcomplex<f64>>, %q0 : !quantum.bit) {
    // CHECK: quantum.set_state(%{{.*}}) %{{.*}} : (memref<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    %0 = quantum.set_state(%arg0) %q0 : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @set_basis_state
module @set_basis_state {
  func.func @foo(%arg0: tensor<2xi1>, %q0 : !quantum.bit) {
    // CHECK: quantum.set_basis_state(%{{.*}}) %{{.*}} : (memref<2xi1>, !quantum.bit) -> !quantum.bit
    %0 = quantum.set_basis_state(%arg0) %q0 : (tensor<2xi1>, !quantum.bit) -> !quantum.bit
    return
  }
}

