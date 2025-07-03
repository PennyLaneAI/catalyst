// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --finalize-memref-to-llvm --convert-index-to-llvm --convert-quantum-to-llvm --split-input-file %s | FileCheck %s

////////////////////////
// Runtime Management //
////////////////////////

// CHECK: llvm.func @__catalyst__rt__initialize(!llvm.ptr)

// CHECK-LABEL: @init
func.func @init() {

    // CHECK: llvm.call @__catalyst__rt__initialize({{%.+}}) : (!llvm.ptr) -> ()
    quantum.init

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__finalize()

// CHECK-LABEL: @finalize
func.func @finalize() {

    // CHECK: llvm.call @__catalyst__rt__finalize()
    quantum.finalize

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)

// CHECK-LABEL: @device
func.func @device() {
    // CHECKL llvm.mlir.global internal constant @lightning.qubit("lightning.qubit\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @rtd_lightning.so("rtd_lightning.so\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @"{my_attr: my_attr_value}"("{my_attr: my_attr_value}\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @lightning.kokkos("lightning.kokkos\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @"{my_other_attr: my_other_attr_value}"("{my_other_attr: my_other_attr_value}\00") {addr_space = 0 : i32}

    // CHECK: [[shots:%.+]] = llvm.mlir.constant(1000 : i64) : i64
    // CHECK: [[d0:%.+]] = llvm.mlir.addressof @rtd_lightning.so : !llvm.ptr
    // CHECK: [[d1:%.+]] = llvm.getelementptr inbounds [[d0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    // CHECK: [[bo:%.+]] = llvm.mlir.addressof @lightning.qubit : !llvm.ptr
    // CHECK: [[b1:%.+]] = llvm.getelementptr inbounds [[bo]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i8>
    // CHECK: [[d3:%.+]] = llvm.mlir.addressof @"{my_attr: my_attr_value}" : !llvm.ptr
    // CHECK: [[d4:%.+]] = llvm.getelementptr inbounds [[d3]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
    // CHECK: [[false:%.+]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.call @__catalyst__rt__device_init([[d1]], [[b1]], [[d4]], [[shots]], [[false]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %shots = llvm.mlir.constant(1000 : i64) : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "lightning.qubit", "{my_attr: my_attr_value}"]

    // CHECK: [[e0:%.+]] = llvm.mlir.addressof @rtd_lightning.so : !llvm.ptr
    // CHECK: [[e1:%.+]] = llvm.getelementptr inbounds [[e0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    // CHECK: [[e2:%.+]] = llvm.mlir.addressof @lightning.kokkos : !llvm.ptr
    // CHECK: [[e3:%.+]] = llvm.getelementptr inbounds [[e2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    // CHECK: [[e4:%.+]] = llvm.mlir.addressof @"{my_other_attr: my_other_attr_value}" : !llvm.ptr
    // CHECK: [[e5:%.+]] = llvm.getelementptr inbounds [[e4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
    // CHECK: [[false:%.+]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.call @__catalyst__rt__device_init([[e1]], [[e3]], [[e5]], [[shots]], [[false]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()

    quantum.device shots(%shots) ["rtd_lightning.so", "lightning.kokkos", "{my_other_attr: my_other_attr_value}"]

    // CHECK: [[d0:%.+]] = llvm.mlir.addressof @rtd_lightning.so : !llvm.ptr
    // CHECK: [[d1:%.+]] = llvm.getelementptr inbounds [[d0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    // CHECK: [[bo:%.+]] = llvm.mlir.addressof @lightning.qubit : !llvm.ptr
    // CHECK: [[b1:%.+]] = llvm.getelementptr inbounds [[bo]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i8>
    // CHECK: [[d3:%.+]] = llvm.mlir.addressof @"{my_noshots_attr: my_noshots_attr_value}" : !llvm.ptr
    // CHECK: [[d4:%.+]] = llvm.getelementptr inbounds [[d3]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
    // CHECK: [[zero_shots:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[false:%.+]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.call @__catalyst__rt__device_init([[d1]], [[b1]], [[d4]], [[zero_shots]], [[false]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    quantum.device ["rtd_lightning.so", "lightning.qubit", "{my_noshots_attr: my_noshots_attr_value}"]

    // CHECK: [[true:%.+]] = llvm.mlir.constant(true) : i1
    // CHECK: llvm.call @__catalyst__rt__device_init({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, [[true]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    quantum.device ["blah.so", "blah.qubit", ""] {auto_qubit_management}

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__num_qubits()

// CHECK-LABEL: @num_qubits
func.func @num_qubits() {
    // CHECK: {{%.+}} = llvm.call @__catalyst__rt__num_qubits() : () -> i64
    %0 = quantum.num_qubits : i64
    return
}

// -----

///////////////////////
// Memory Management //
///////////////////////

// CHECK: llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr

// CHECK-LABEL: @alloc
func.func @alloc(%c : i64) {

    // CHECK: llvm.call @__catalyst__rt__qubit_allocate_array(%arg0)
    quantum.alloc(%c) : !quantum.reg

    // CHECK: [[c5:%.+]] = llvm.mlir.constant(5 : i64)
    // CHECK: llvm.call @__catalyst__rt__qubit_allocate_array([[c5]])
    quantum.alloc(5) : !quantum.reg

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)

// CHECK-LABEL: @dealloc
func.func @dealloc(%r : !quantum.reg) {

    // CHECK: llvm.call @__catalyst__rt__qubit_release_array(%arg0)
    quantum.dealloc %r : !quantum.reg

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr

// CHECK-LABEL: @extract
func.func @extract(%r : !quantum.reg, %c : i64) {

    // CHECK: [[qb_ptr:%.+]] = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%arg0, %arg1) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK: llvm.load [[qb_ptr]] : !llvm.ptr -> !llvm.ptr
    quantum.extract %r[%c] : !quantum.reg -> !quantum.bit

    // CHECK: [[c5:%.+]] = llvm.mlir.constant(5 : i64)
    // CHECK: [[qb_ptr:%.+]] = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%arg0, [[c5]]) : (!llvm.ptr, i64) -> !llvm.ptr
    // CHECK: llvm.load [[qb_ptr]] : !llvm.ptr -> !llvm.ptr

    quantum.extract %r[5] : !quantum.reg -> !quantum.bit

    return
}

// -----

// CHECK-LABEL: @insert
func.func @insert(%r : !quantum.reg, %q : !quantum.bit) -> !quantum.reg {

    %new_r = quantum.insert %r[5], %q : !quantum.reg, !quantum.bit

    // CHECK-NEXT: return %arg0
    return %new_r : !quantum.reg
}

// -----

///////////////////
// Quantum Gates //
///////////////////

// CHECK-LABEL: @custom_gate
module @custom_gate {
  // CHECK: llvm.func @__catalyst__qis__Identity(!llvm.ptr, !llvm.ptr)
  // CHECK-LABEL: @test
  func.func @test(%q0: !quantum.bit, %p: f64) -> () {
    // CHECK: [[nullptr:%.+]] = llvm.mlir.zero
    // CHECK: llvm.call @__catalyst__qis__Identity(%arg0, [[nullptr]])
    %q1 = quantum.custom "Identity"() %q0 : !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @custom_gate
module @custom_gate {
  // CHECK: llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  // CHECK-LABEL: @test
  func.func @test(%q0: !quantum.bit, %p: f64) -> () {
    // CHECK: [[nullptr:%.+]] = llvm.mlir.zero
    // CHECK: llvm.call @__catalyst__qis__RX(%arg1, %arg0, [[nullptr]])
    %q1 = quantum.custom "RX"(%p) %q0 : !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @custom_gate
module @custom_gate {
  // CHECK: llvm.func @__catalyst__qis__SWAP(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  // CHECK-LABEL: @test
  func.func @test(%q0: !quantum.bit, %p: f64) -> () {
    // CHECK: [[nullptr:%.+]] = llvm.mlir.zero
    // CHECK: llvm.call @__catalyst__qis__SWAP(%arg0, %arg0, [[nullptr]])
    %q1:2 = quantum.custom "SWAP"() %q0, %q0 : !quantum.bit, !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @custom_gate
module @custom_gate {
  // CHECK-DAG: llvm.func @__catalyst__qis__CRot(f64, f64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  // CHECK-LABEL: @test
  func.func @test(%q0: !quantum.bit, %p: f64) -> () {
    // CHECK: [[nullptr:%.+]] = llvm.mlir.zero
    // CHECK: llvm.call @__catalyst__qis__CRot(%arg1, %arg1, %arg1, %arg0, %arg0, [[nullptr]])
    %q1:2 = quantum.custom "CRot"(%p, %p, %p) %q0, %q0 : !quantum.bit, !quantum.bit
    return
  }
}

// -----


// CHECK-LABEL: @custom_gate
module @custom_gate {
  // CHECK: llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  // CHECK-LABEL: @test
  func.func @test(%q0: !quantum.bit, %p: f64) -> () {
    // CHECK: [[nullptr:%.+]] = llvm.mlir.zero
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i1, i64, ptr, ptr)>
    // CHECK: [[true:%.+]] = llvm.mlir.constant(true)
    // CHECK: [[off0:%.+]] = llvm.getelementptr inbounds [[alloca]][0, 0]
    // CHECK: [[off1:%.+]] = llvm.getelementptr inbounds [[alloca]][0, 1]
    // CHECK: [[off2:%.+]] = llvm.getelementptr inbounds [[alloca]][0, 2]
    // CHECK: [[off3:%.+]] = llvm.getelementptr inbounds [[alloca]][0, 3]
    // CHECK: llvm.store [[true]], [[off0]]
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.store [[c0]], [[off1]]
    // CHECK: llvm.store [[nullptr]], [[off2]]
    // CHECK: llvm.store [[nullptr]], [[off3]]
    // CHECK: llvm.call @__catalyst__qis__RX(%arg1, %arg0, [[alloca]])
    %q1 = quantum.custom "RX"(%p) %q0 { adjoint } : !quantum.bit
    return
  }
}

// -----

// CHECK: llvm.func @__catalyst__qis__MultiRZ(f64, !llvm.ptr, i64, ...)

// CHECK-LABEL: @multirz
func.func @multirz(%q0 : !quantum.bit, %p : f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {

    // CHECK: [[p:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__MultiRZ(%arg1, [[p]], [[c1]], %arg0)
    %q1 = quantum.multirz(%p) %q0 : !quantum.bit

    // CHECK: [[p:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__MultiRZ(%arg1, [[p]], [[c2]], %arg0, %arg0)
    %q2:2 = quantum.multirz(%p) %q1, %q1 : !quantum.bit, !quantum.bit

    // CHECK: [[p:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: [[c3:%.+]] = llvm.mlir.constant(3 : i64)
    // CHECK: llvm.call @__catalyst__qis__MultiRZ(%arg1, [[p]], [[c3]], %arg0, %arg0, %arg0)
    %q3:3 = quantum.multirz(%p) %q2#0, %q2#1, %q2#1 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: [[st1:%.+]] = llvm.insertvalue %arg0
    // CHECK: [[st2:%.+]] = llvm.insertvalue %arg0, [[st1]]
    // CHECK: [[st3:%.+]] = llvm.insertvalue %arg0, [[st2]]
    // CHECK: return [[st3]]
    return %q3#0, %q3#1, %q3#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__QubitUnitary(!llvm.ptr, !llvm.ptr, i64, ...)

// CHECK-LABEL: @qubit_unitary
func.func @qubit_unitary(%q0 : !quantum.bit, %p1 : memref<2x2xcomplex<f64>>,  %p2 : memref<4x4xcomplex<f64>>) -> (!quantum.bit, !quantum.bit) {



    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[alloca:%.+]] = llvm.alloca [[one]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[alloca:%.+]] = llvm.alloca [[one]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    %q1 = quantum.unitary(%p1 : memref<2x2xcomplex<f64>>) %q0 : !quantum.bit

    %q2:2 = quantum.unitary(%p2 : memref<4x4xcomplex<f64>>) %q1, %q1 : !quantum.bit, !quantum.bit

    // CHECK: llvm.call @__catalyst__qis__QubitUnitary(
    // CHECK: llvm.call @__catalyst__qis__QubitUnitary(

    return %q2#0, %q2#1 : !quantum.bit, !quantum.bit
}

// -----

/////////////////
// Observables //
/////////////////

// CHECK-LABEL: @compbasis
func.func @compbasis(%q : !quantum.bit, %r : !quantum.reg) {

    // CHECK: builtin.unrealized_conversion_cast %arg0
    quantum.compbasis qubits %q : !quantum.obs
    // CHECK: builtin.unrealized_conversion_cast %arg0, %arg0
    quantum.compbasis qubits %q, %q : !quantum.obs

    // CHECK: builtin.unrealized_conversion_cast{{[[:space:]]}}
    quantum.compbasis qreg %r : !quantum.obs

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64

// CHECK-LABEL: @namedobs
func.func @namedobs(%q : !quantum.bit) {

    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @__catalyst__qis__NamedObs([[c0]], %arg0)
    quantum.namedobs %q[Identity] : !quantum.obs
    // CHECK: [[c4:%.+]] = llvm.mlir.constant(4 : i64)
    // CHECK: llvm.call @__catalyst__qis__NamedObs([[c4]], %arg0)
    quantum.namedobs %q[Hadamard] : !quantum.obs

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__HermitianObs(!llvm.ptr, i64, ...) -> i64

// CHECK-LABEL: @hermitian
func.func @hermitian(%q : !quantum.bit, %p1 : memref<2x2xcomplex<f64>>, %p2 : memref<4x4xcomplex<f64>>) {
    // Only check the last members of the deconstructed memref struct being inserted.

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    quantum.hermitian(%p1 : memref<2x2xcomplex<f64>>) %q : !quantum.obs
    quantum.hermitian(%p2 : memref<4x4xcomplex<f64>>) %q, %q : !quantum.obs

    // CHECK: llvm.call @__catalyst__qis__HermitianObs
    // CHECK: llvm.call @__catalyst__qis__HermitianObs

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__TensorObs(i64, ...) -> i64

// CHECK-LABEL: @tensor
func.func @tensor(%obs : !quantum.obs) {

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__TensorObs([[c1]], %arg0)
    quantum.tensor %obs : !quantum.obs
    // CHECK: [[c3:%.+]] = llvm.mlir.constant(3 : i64)
    // CHECK: llvm.call @__catalyst__qis__TensorObs([[c3]], %arg0, %arg0, %arg0)
    quantum.tensor %obs, %obs, %obs : !quantum.obs

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__HamiltonianObs(!llvm.ptr, i64, ...) -> i64
// CHECK-LABEL: @hamiltonian
func.func @hamiltonian(%obs : !quantum.obs, %p1 : memref<1xf64>, %p2 : memref<3xf64>) {
    // CHECK: [[memrefvar:%.+]] = llvm.mlir.poison
    // CHECK: [[memrefvar0:%.+]] = llvm.insertvalue %arg1, [[memrefvar]][0]
    // CHECK: [[memrefvar1:%.+]] = llvm.insertvalue %arg2, [[memrefvar0]][1]
    // CHECK: [[memrefvar2:%.+]] = llvm.insertvalue %arg3, [[memrefvar1]][2]
    // CHECK: [[memrefvar3:%.+]] = llvm.insertvalue %arg4, [[memrefvar2]][3, 0]
    // CHECK: [[memrefvar4:%.+]] = llvm.insertvalue %arg5, [[memrefvar3]][4, 0]
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.store [[memrefvar4]], [[alloca]]
    // CHECK: llvm.call @__catalyst__qis__HamiltonianObs([[alloca]], [[c1]], %arg0)
    quantum.hamiltonian(%p1 : memref<1xf64>) %obs : !quantum.obs
    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__HamiltonianObs(!llvm.ptr, i64, ...) -> i64

// CHECK-LABEL: @hamiltonian
func.func @hamiltonian(%obs : !quantum.obs, %p1 : memref<1xf64>, %p2 : memref<3xf64>) {
    // CHECK: [[memrefvar:%.+]] = llvm.mlir.poison
    // CHECK: [[memrefvar0:%.+]] = llvm.insertvalue %arg6, [[memrefvar]][0]
    // CHECK: [[memrefvar1:%.+]] = llvm.insertvalue %arg7, [[memrefvar0]][1]
    // CHECK: [[memrefvar2:%.+]] = llvm.insertvalue %arg8, [[memrefvar1]][2]
    // CHECK: [[memrefvar3:%.+]] = llvm.insertvalue %arg9, [[memrefvar2]][3, 0]
    // CHECK: [[memrefvar4:%.+]] = llvm.insertvalue %arg10, [[memrefvar3]][4, 0]
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c3:%.+]] = llvm.mlir.constant(3 : i64)
    // CHECK: llvm.store [[memrefvar4]], [[alloca]]
    // CHECK: llvm.call @__catalyst__qis__HamiltonianObs([[alloca]], [[c3]], %arg0, %arg0, %arg0)
    quantum.hamiltonian(%p2 : memref<3xf64>) %obs, %obs, %obs : !quantum.obs
    return
}

// -----

//////////////////
// Measurements //
//////////////////

// CHECK: llvm.func @__catalyst__qis__Measure(!llvm.ptr, i32) -> !llvm.ptr

// CHECK-LABEL: @measure
func.func @measure(%q : !quantum.bit) -> !quantum.bit {

    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(-1 : i32) : i32

    // CHECK: llvm.call @__catalyst__qis__Measure(%arg0, [[postselect]])
    %res, %new_q = quantum.measure %q : i1, !quantum.bit

    // CHECK: return %arg0
    return %new_q : !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__Measure(!llvm.ptr, i32) -> !llvm.ptr

// CHECK-LABEL: @measure
func.func @measure(%q : !quantum.bit) -> !quantum.bit {

    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.call @__catalyst__qis__Measure(%arg0, [[postselect]])
    %res, %new_q = quantum.measure %q postselect 0 : i1, !quantum.bit

    // CHECK: return %arg0
    return %new_q : !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__Sample(!llvm.ptr, i64, ...)

// CHECK-LABEL: @sample
func.func @sample(%q : !quantum.bit, %dyn_shots: i64) {

    %o1 = quantum.compbasis qubits %q : !quantum.obs

    %idx1 = index.casts %dyn_shots : i64 to index
    %dyn_alloc1 = memref.alloc(%idx1) : memref<?x1xf64>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Sample([[ptr]], [[c1]], %arg0)
    quantum.sample %o1 in(%dyn_alloc1 : memref<?x1xf64>)

    return
}

// -----

// CHECK-LABEL: @sample
func.func @sample(%q : !quantum.bit, %dyn_shots: i64) {
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__Sample([[ptr]], [[c2]], %arg0, %arg0)
    %o2 = quantum.compbasis qubits %q, %q : !quantum.obs
    %idx2 = index.casts %dyn_shots : i64 to index
    %dyn_alloc2 = memref.alloc(%idx2) : memref<?x2xf64>
    quantum.sample %o2 in(%dyn_alloc2 : memref<?x2xf64>)
    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__Counts(!llvm.ptr, i64, ...)

// CHECK-LABEL: @counts
func.func @counts(%q : !quantum.bit) {

    %o1 = quantum.compbasis qubits %q : !quantum.obs
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Counts([[ptr]], [[c1]], %arg0)
    %in_eigvals1 = memref.alloc() : memref<2xf64>
    %in_counts1 = memref.alloc() : memref<2xi64>
    quantum.counts %o1 in(%in_eigvals1 : memref<2xf64>, %in_counts1 : memref<2xi64>)


    return
}

// -----

// CHECK-LABEL: @counts
func.func @counts(%q : !quantum.bit) {
    %o2 = quantum.compbasis qubits %q, %q : !quantum.obs
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__Counts([[ptr]], [[c2]], %arg0, %arg0)
    %in_eigvals2 = memref.alloc() : memref<4xf64>
    %in_counts2 = memref.alloc() : memref<4xi64>
    quantum.counts %o2 in(%in_eigvals2 : memref<4xf64>, %in_counts2 : memref<4xi64>)
    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__Expval(i64)

// CHECK-LABEL: @expval
func.func @expval(%obs : !quantum.obs) {

    // CHECK: llvm.call @__catalyst__qis__Expval(%arg0)
    quantum.expval %obs : f64

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__Variance(i64)

// CHECK-LABEL: @var
func.func @var(%obs : !quantum.obs) {

    // CHECK: llvm.call @__catalyst__qis__Variance(%arg0)
    quantum.var %obs : f64

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__Probs(!llvm.ptr, i64, ...)

// CHECK-LABEL: @probs
func.func @probs(%q : !quantum.bit) {

    %o1 = quantum.compbasis qubits %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Probs([[ptr]], [[c1]], %arg0)
    %alloc1 = memref.alloc() : memref<2xf64>
    quantum.probs %o1 in(%alloc1 : memref<2xf64>)

    return
}

// -----
// CHECK-LABEL: @probs
func.func @probs(%q : !quantum.bit) {
    %o2 = quantum.compbasis qubits %q, %q, %q, %q : !quantum.obs
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c4:%.+]] = llvm.mlir.constant(4 : i64)
    // CHECK: llvm.call @__catalyst__qis__Probs([[ptr]], [[c4]], %arg0, %arg0, %arg0, %arg0)
    %alloc2 = memref.alloc() : memref<16xf64>
    quantum.probs %o2 in(%alloc2 : memref<16xf64>)
    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__State(!llvm.ptr, i64, ...)

// CHECK-LABEL: @state
func.func @state(%q : !quantum.bit) {
    %o1 = quantum.compbasis qubits %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @__catalyst__qis__State([[ptr]], [[c0]])
    %alloc1 = memref.alloc() : memref<2xcomplex<f64>>
    quantum.state %o1 in(%alloc1 : memref<2xcomplex<f64>>)

    return
}

// -----

// CHECK-LABEL: @state
func.func @state(%q : !quantum.bit) {
    %o2 = quantum.compbasis qubits %q, %q, %q, %q : !quantum.obs
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @__catalyst__qis__State([[ptr]], [[c0]])
    %alloc2 = memref.alloc() : memref<16xcomplex<f64>>
    quantum.state %o2 in(%alloc2: memref<16xcomplex<f64>>)
    return
}

// -----

// CHECK-LABEL: @controlled_circuit
func.func @controlled_circuit(%1 : !quantum.bit, %2 : !quantum.bit, %3 : !quantum.bit) {

    %true = llvm.mlir.constant (1 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: [[true:%.+]] = llvm.mlir.constant(true)
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x i1
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x !llvm.ptr
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[mod:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i1, i64, ptr, ptr)>


    // CHECK-DAG: [[cst6:%.+]] = llvm.mlir.constant(6.0
    // CHECK-DAG: [[cst9:%.+]] = llvm.mlir.constant(9.0
    // CHECK-DAG: [[cst3:%.+]] = llvm.mlir.constant(3.0
    // CHECK-DAG: [[null:%.+]] = llvm.mlir.zero
    // CHECK-DAG: [[false:%.+]] = llvm.mlir.constant(false)

    // CHECK: [[offset0:%.+]] = llvm.getelementptr inbounds [[mod]][0, 0]
    // CHECK: [[offset1:%.+]] = llvm.getelementptr inbounds [[mod]][0, 1]
    // CHECK: [[offset2:%.+]] = llvm.getelementptr inbounds [[mod]][0, 2]
    // CHECK: [[offset3:%.+]] = llvm.getelementptr inbounds [[mod]][0, 3]
    // CHECK: [[offset:%.+]] = llvm.getelementptr inbounds [[alloca1]][0]

    // CHECK: llvm.store %arg2, [[offset]]

    // CHECK: [[ptr_to_bool:%.+]] = llvm.getelementptr inbounds [[alloca0]][0]
    // CHECK: llvm.store [[true]], [[ptr_to_bool]]
    // CHECK: llvm.store {{.*}}, [[offset2]]
    // CHECK: llvm.store {{.*}}, [[offset3]]
    // CHECK: llvm.call @__catalyst__qis__Rot
    // CHECK-SAME: [[mod]]
    %out_qubits, %out_ctrl_qubits = quantum.custom "Rot"(%cst, %cst_1, %cst_0) %2 ctrls (%3) ctrlvals (%true) : !quantum.bit ctrls !quantum.bit

    return
}
// -----

// CHECK-LABEL: @controlled_circuit
func.func @controlled_circuit(%1 : !quantum.bit, %2 : !quantum.bit, %3 : !quantum.bit) {
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %true = llvm.mlir.constant (1 : i1) :i1

    // CHECK: [[cst6:%.+]] = llvm.mlir.constant(6.0
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x i1
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x !llvm.ptr
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[mod:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i1, i64, ptr, ptr)>
    // CHECK: [[true:%.+]] = llvm.mlir.constant(true)

    // CHECK: [[offset0:%.+]] = llvm.getelementptr inbounds [[mod]][0, 0]
    // CHECK: [[offset1:%.+]] = llvm.getelementptr inbounds [[mod]][0, 1]
    // CHECK: [[offset2:%.+]] = llvm.getelementptr inbounds [[mod]][0, 2]
    // CHECK: [[offset3:%.+]] = llvm.getelementptr inbounds [[mod]][0, 3]
    // CHECK: [[offset:%.+]] = llvm.getelementptr inbounds [[alloca1]][0]

    // CHECK: llvm.store %arg2, [[offset]]
    // CHECK: [[ptr_to_bool:%.+]] = llvm.getelementptr inbounds [[alloca0]][0]
    // CHECK: llvm.store [[true]], [[ptr_to_bool]]
    // CHECK: llvm.store {{.*}}, [[offset2]]
    // CHECK: llvm.store {{.*}}, [[offset3]]
    // CHECK: llvm.call @__catalyst__qis__MultiRZ
    // CHECK-SAME: [[mod]]

    %out_qubits_2:2, %out_ctrl_qubits_3 = quantum.multirz(%cst) %2, %1 ctrls (%3) ctrlvals (%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    return
}

// -----

// CHECK-LABEL: @controlled_circuit
func.func @controlled_circuit(%1 : !quantum.bit, %2 : !quantum.bit, %3 : !quantum.bit) {
    %arg0 = memref.alloc() : memref<2x2xcomplex<f64>>
    %true = llvm.mlir.constant (1 : i1) :i1

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca0:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca1:%.+]] = llvm.alloca [[c1]] x i1

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[alloca2:%.+]] = llvm.alloca [[c1]] x !llvm.ptr

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[mod:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(i1, i64, ptr, ptr)>

    // CHECK: [[offset0:%.+]] = llvm.getelementptr inbounds [[mod]][0, 0]
    // CHECK: [[offset1:%.+]] = llvm.getelementptr inbounds [[mod]][0, 1]
    // CHECK: [[offset2:%.+]] = llvm.getelementptr inbounds [[mod]][0, 2]
    // CHECK: [[offset3:%.+]] = llvm.getelementptr inbounds [[mod]][0, 3]

    // CHECK: llvm.store {{.*}}, [[offset2]]
    // CHECK: llvm.store {{.*}}, [[offset3]]

    // CHECK: llvm.call @__catalyst__qis__QubitUnitary
    // CHECK-SAME: [[mod]]

    %out_qubits_4, %out_ctrl_qubits_5 = quantum.unitary(%arg0 : memref<2x2xcomplex<f64>>) %2 ctrls (%3) ctrlvals (%true) : !quantum.bit ctrls !quantum.bit
    return
}

// -----

// CHECK-LABEL: @test_set_state
module @test_set_state {
    func.func @set_state(%1 : !quantum.bit, %arg0 : memref<2xcomplex<f64>>) {
        // CHECK: llvm.call @__catalyst__qis__SetState
        %2 = quantum.set_state(%arg0) %1 : (memref<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
        return
    }
}

// -----

// CHECK-LABEL: @test_set_basis_state
module @test_set_basis_state {
    func.func @set_basis_state(%1 : !quantum.bit, %arg0 : memref<1xi1>) {
        // CHECK: llvm.call @__catalyst__qis__SetBasisState
        %2 = quantum.set_basis_state(%arg0) %1 : (memref<1xi1>, !quantum.bit) -> !quantum.bit
        return
    }
}
