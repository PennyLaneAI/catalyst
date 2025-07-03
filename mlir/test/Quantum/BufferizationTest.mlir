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

// RUN: quantum-opt --one-shot-bufferize --split-input-file %s | FileCheck %s

func.func @qubit_unitary(%q0: !quantum.bit, %matrix: tensor<2x2xcomplex<f64>>) {
    // CHECK: [[memref:%.+]] = bufferization.to_memref %arg1 : tensor<2x2xcomplex<f64>> to memref<2x2xcomplex<f64>>
    // CHECK: {{%.+}} = quantum.unitary([[memref]] : memref<2x2xcomplex<f64>>) %arg0 : !quantum.bit
    %out_qubits = quantum.unitary(%matrix : tensor<2x2xcomplex<f64>>) %q0 : !quantum.bit

    func.return
}

// -----

func.func @hermitian(%q0: !quantum.bit, %matrix: tensor<2x2xcomplex<f64>>) {
    // CHECK: [[memref:%.+]] = bufferization.to_memref %arg1 : tensor<2x2xcomplex<f64>> to memref<2x2xcomplex<f64>>
    // CHECK: {{%.+}} = quantum.hermitian([[memref]] : memref<2x2xcomplex<f64>>) %arg0 : !quantum.obs
    %obs = quantum.hermitian(%matrix : tensor<2x2xcomplex<f64>>) %q0 : !quantum.obs

    func.return
}

// -----

func.func @hamiltonian(%obs: !quantum.obs, %coeffs: tensor<1xf64>){
    // CHECK: [[memref:%.+]] = bufferization.to_memref %arg1 : tensor<1xf64> to memref<1xf64>
    // CHECK: {{%.+}} = quantum.hamiltonian([[memref]] : memref<1xf64>) %arg0 : !quantum.obs
    %hamil = quantum.hamiltonian(%coeffs: tensor<1xf64>) %obs : !quantum.obs

    func.return
}

// -----

//////////////////
// Measurements //
//////////////////

func.func @counts(%q0: !quantum.bit, %q1: !quantum.bit, %c : i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // CHECK: [[eigval_alloc:%.+]] = memref.alloc() : memref<4xf64>
    // CHECK: [[counts_alloc:%.+]] = memref.alloc() : memref<4xi64>
    // CHECK: quantum.counts {{.*}} in([[eigval_alloc]] : memref<4xf64>, [[counts_alloc]] : memref<4xi64>)
    %static_counts:2 = quantum.counts %obs : tensor<4xf64>, tensor<4xi64>

    // CHECK: [[idx1:%.+]] = index.casts %arg2 : i64 to index
    // CHECK: [[alloc1:%.+]] = memref.alloc([[idx1]]) : memref<?xf64>
    // CHECK: [[idx2:%.+]] = index.casts %arg2 : i64 to index
    // CHECK: [[alloc2:%.+]] = memref.alloc([[idx2]]) : memref<?xi64>
    // CHECK: quantum.counts {{.*}} in([[alloc1]] : memref<?xf64>, [[alloc2]] : memref<?xi64>)
    %dyn_counts:2 = quantum.counts %obs shape %c : tensor<?xf64>, tensor<?xi64>

    func.return
}

// -----

func.func @sample(%c1 : i64, %c2 : i64, %q0: !quantum.bit, %q1: !quantum.bit, %dyn_shots: i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // CHECK: [[idx1:%.+]] = index.casts %arg0 : i64 to index
    // CHECK: [[alloc1:%.+]] = memref.alloc([[idx1]]) : memref<?x2xf64>
    // CHECK: quantum.sample {{.*}} in([[alloc1]] : memref<?x2xf64>)
    %samples_dyn1 = quantum.sample %obs shape %c1: tensor<?x2xf64>

    // CHECK: [[idx2:%.+]] = index.casts %arg1 : i64 to index
    // CHECK: [[alloc2:%.+]] = memref.alloc([[idx2]]) : memref<42x?xf64>
    // CHECK: quantum.sample {{.*}} in([[alloc2]] : memref<42x?xf64>)
    %samples_dyn2 = quantum.sample %obs shape %c2 : tensor<42x?xf64>

    // CHECK: [[alloc_static:%.+]] = memref.alloc() : memref<10x2xf64>
    // CHECK: quantum.sample {{.*}} in([[alloc_static]] : memref<10x2xf64>)
    %samples_static = quantum.sample %obs : tensor<10x2xf64>

    // CHECK: [[idx3:%.+]] = index.casts %arg0 : i64 to index
    // CHECK: [[idx4:%.+]] = index.casts %arg1 : i64 to index
    // CHECK: [[alloc3:%.+]] = memref.alloc([[idx3]], [[idx4]]) : memref<?x?xf64>
    // CHECK: quantum.sample {{.*}} in([[alloc3]] : memref<?x?xf64>)
    %samples_dynAll = quantum.sample %obs shape %c1, %c2 : tensor<?x?xf64>

    func.return
}

// -----

func.func @probs_static(%q0: !quantum.bit, %q1: !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs
    // CHECK: [[alloc:%.+]] = memref.alloc() : memref<4xf64>
    // CHECK: quantum.probs {{.*}} in([[alloc]] : memref<4xf64>)
    %probs = quantum.probs %obs : tensor<4xf64>
    func.return
}

// -----

func.func @probs_dynamic(%q0: !quantum.bit, %q1: !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs
    %c4 = arith.constant 4 : i64
    // CHECK: [[four:%.+]] = arith.constant 4
    // CHECK: [[index:%.+]] = index.casts [[four]] : i64 to index
    // CHECK: [[alloc:%.+]] = memref.alloc([[index]]) : memref<?xf64>
    // CHECK: quantum.probs {{.*}} in([[alloc]] : memref<?xf64>)
    %probs = quantum.probs %obs shape %c4 : tensor<?xf64>
    func.return
}

// -----

func.func @state_static(%q0: !quantum.bit, %q1: !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs
    // CHECK: [[alloc:%.+]] = memref.alloc() : memref<4xcomplex<f64>>
    // CHECK: quantum.state {{.*}} in([[alloc]] : memref<4xcomplex<f64>>)
    %state = quantum.state %obs : tensor<4xcomplex<f64>>
    func.return
}


// -----

func.func @state_dynamic(%r : !quantum.reg) {
    %obs = quantum.compbasis qreg %r : !quantum.obs
    %c4 = arith.constant 4 : i64
    // CHECK: [[four:%.+]] = arith.constant 4
    // CHECK: [[index:%.+]] = index.casts [[four]] : i64 to index
    // CHECK: [[alloc:%.+]] = memref.alloc([[index]]) : memref<?xcomplex<f64>>
    // CHECK: quantum.state {{.*}} in([[alloc]] : memref<?xcomplex<f64>>)
    %state = quantum.state %obs shape %c4 : tensor<?xcomplex<f64>>
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
