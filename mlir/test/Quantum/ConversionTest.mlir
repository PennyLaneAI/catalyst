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

// RUN: quantum-opt --finalize-memref-to-llvm --convert-quantum-to-llvm --split-input-file %s | FileCheck %s

////////////////////////
// Runtime Management //
////////////////////////

// CHECK: llvm.func @__catalyst__rt__initialize()

// CHECK-LABEL: @init
func.func @init() {

    // CHECK: llvm.call @__catalyst__rt__initialize()
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

// CHECK: llvm.func @__catalyst__rt__device_init(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>)

// CHECK-LABEL: @device
func.func @device() {
    // CHECKL llvm.mlir.global internal constant @lightning.qubit("lightning.qubit\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @rtd_lightning.so("rtd_lightning.so\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @"{shots: 0}"("{shots: 0}\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @lightning.kokkos("lightning.kokkos\00") {addr_space = 0 : i32}
    // CHECKL llvm.mlir.global internal constant @"{shots: 1000}"("{shots: 1000}\00") {addr_space = 0 : i32}

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[d0:%.+]] = llvm.mlir.addressof @rtd_lightning.so : !llvm.ptr<array<17 x i8>>
    // CHECK: [[d1:%.+]] = llvm.getelementptr [[d0]][[[c1]], [[c1]]] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[bo:%.+]] = llvm.mlir.addressof @lightning.qubit : !llvm.ptr<array<16 x i8>>
    // CHECK: [[b1:%.+]] = llvm.getelementptr [[bo]][[[c0]], [[c0]]] : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[d3:%.+]] = llvm.mlir.addressof @"{shots: 0}" : !llvm.ptr<array<11 x i8>>
    // CHECK: [[d4:%.+]] = llvm.getelementptr [[d3]][[[c2]], [[c2]]] : (!llvm.ptr<array<11 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @__catalyst__rt__device_init([[d1]], [[b1]], [[d4]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    quantum.device ["rtd_lightning.so", "lightning.qubit", "{shots: 0}"]

    // CHECK: [[c3:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[e0:%.+]] = llvm.mlir.addressof @rtd_lightning.so : !llvm.ptr<array<17 x i8>>
    // CHECK: [[e1:%.+]] = llvm.getelementptr [[e0]][[[c3]], [[c3]]] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: [[c4:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[e2:%.+]] = llvm.mlir.addressof @lightning.kokkos : !llvm.ptr<array<17 x i8>>
    // CHECK: [[e3:%.+]] = llvm.getelementptr [[e2]][[[c4]], [[c4]]] : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: [[c5:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: [[e4:%.+]] = llvm.mlir.addressof @"{shots: 1000}" : !llvm.ptr<array<14 x i8>>
    // CHECK: [[e5:%.+]] = llvm.getelementptr [[e4]][[[c5]], [[c5]]] : (!llvm.ptr<array<14 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @__catalyst__rt__device_init([[e1]], [[e3]], [[e5]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    quantum.device ["rtd_lightning.so", "lightning.kokkos", "{shots: 1000}"]

    return
}

// -----

///////////////////////
// Memory Management //
///////////////////////

// CHECK: llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr<struct<"Array", opaque>>

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

// CHECK: llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr<struct<"Array", opaque>>)

// CHECK-LABEL: @dealloc
func.func @dealloc(%r : !quantum.reg) {

    // CHECK: llvm.call @__catalyst__rt__qubit_release_array(%arg0)
    quantum.dealloc %r : !quantum.reg

    return
}

// -----

// CHECK: llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr<struct<"Array", opaque>>, i64) -> !llvm.ptr<i8>

// CHECK-LABEL: @extract
func.func @extract(%r : !quantum.reg, %c : i64) {

    // CHECK: [[elem_ptr:%.+]] = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%arg0, %arg1)
    // CHECK: [[qb_ptr:%.+]] = llvm.bitcast [[elem_ptr]]
    // CHECK: llvm.load [[qb_ptr]] : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
    quantum.extract %r[%c] : !quantum.reg -> !quantum.bit

    // CHECK: [[c5:%.+]] = llvm.mlir.constant(5 : i64)
    // CHECK: [[elem_ptr:%.+]] = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%arg0, [[c5]])
    // CHECK: [[qb_ptr:%.+]] = llvm.bitcast [[elem_ptr]]
    // CHECK: llvm.load [[qb_ptr]] : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
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

// CHECK-DAG: llvm.func @__catalyst__qis__Identity(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>)
// CHECK-DAG: llvm.func @__catalyst__qis__RX(f64, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>)
// CHECK-DAG: llvm.func @__catalyst__qis__SWAP(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>)
// CHECK-DAG: llvm.func @__catalyst__qis__CRot(f64, f64, f64, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>)
// CHECK-DAG: llvm.func @__catalyst__qis__Toffoli(!llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<"Qubit", opaque>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>)

// CHECK-LABEL: @custom_gate
func.func @custom_gate(%q0 : !quantum.bit, %p : f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__Identity(%arg0, [[p]])
    %q1 = quantum.custom "Identity"() %q0 { result_segment_sizes = array<i32: 1, 0> } : !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__RX(%arg1, %arg0, [[p]])
    %q2 = quantum.custom "RX"(%p) %q1 { result_segment_sizes = array<i32: 1, 0> } : !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__SWAP(%arg0, %arg0, [[p]])
    %q3:2 = quantum.custom "SWAP"() %q2, %q2 { result_segment_sizes = array<i32: 2, 0> } : !quantum.bit, !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__CRot(%arg1, %arg1, %arg1, %arg0, %arg0, [[p]])
    %q4:2 = quantum.custom "CRot"(%p, %p, %p) %q3#0, %q3#1 { result_segment_sizes = array<i32: 2, 0> } : !quantum.bit, !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__Toffoli(%arg0, %arg0, %arg0, [[p]])
    %q5:3 = quantum.custom "Toffoli"() %q4#0, %q4#1, %q4#1 { result_segment_sizes = array<i32: 3, 0> } : !quantum.bit, !quantum.bit, !quantum.bit

    // FIXME!
    // CHECK: [[o:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[a:%.+]] = llvm.alloca [[o]] x !llvm.struct
    // CHECK: llvm.call @__catalyst__qis__RX(%arg1, %arg0, [[a]])
    %q6 = quantum.custom "RX"(%p) %q5#0 { adjoint, result_segment_sizes = array<i32: 1, 0> } : !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: llvm.call @__catalyst__qis__RX(%arg1, %arg0, [[p]])
    %q7 = quantum.custom "RX"(%p) %q6#0 { result_segment_sizes = array<i32: 1, 0> } : !quantum.bit

    // CHECK: [[st1:%.+]] = llvm.insertvalue %arg0
    // CHECK: [[st2:%.+]] = llvm.insertvalue %arg0, [[st1]]
    // CHECK: [[st3:%.+]] = llvm.insertvalue %arg0, [[st2]]
    // CHECK: return [[st3]]
    return %q6, %q5#1, %q5#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__MultiRZ(f64, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>, i64, ...)

// CHECK-LABEL: @multirz
func.func @multirz(%q0 : !quantum.bit, %p : f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__MultiRZ(%arg1, [[p]], [[c1]], %arg0)
    %q1 = quantum.multirz(%p) %q0 : !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__MultiRZ(%arg1, [[p]], [[c2]], %arg0, %arg0)
    %q2:2 = quantum.multirz(%p) %q1, %q1 : !quantum.bit, !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[p:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
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

// CHECK: llvm.func @__catalyst__qis__QubitUnitary(!llvm.ptr<struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>>, !llvm.ptr<struct<(i1, i64, ptr<ptr<struct<"Qubit", opaque>>>, ptr<i1>)>>, i64, ...)

// CHECK-LABEL: @qubit_unitary
func.func @qubit_unitary(%q0 : !quantum.bit, %p1 : memref<2x2xcomplex<f64>>,  %p2 : memref<4x4xcomplex<f64>>) -> (!quantum.bit, !quantum.bit) {
    // Only check the last members of the deconstructed memref struct being inserted.
    // CHECK: [[m1:%.+]] = llvm.insertvalue %arg7
    // CHECK: [[m2:%.+]] = llvm.insertvalue %arg14

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[a:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: [[c1_1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[c1_2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf1:%.+]] = llvm.alloca [[c1_2]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: llvm.store [[m1]], [[buf1]]
    // CHECK: llvm.call @__catalyst__qis__QubitUnitary([[buf1]], [[a]], [[c1_1]], %arg0)
    %q1 = quantum.unitary(%p1 : memref<2x2xcomplex<f64>>) %q0 : !quantum.bit

    // CHECK: [[z:%.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: [[a:%.+]] = llvm.inttoptr [[z]] : i64 to !llvm.ptr<struct
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: [[c1_2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf2:%.+]] = llvm.alloca [[c1_2]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: llvm.store [[m2]], [[buf2]]
    // CHECK: llvm.call @__catalyst__qis__QubitUnitary([[buf2]], [[a]], [[c2]], %arg0, %arg0)
    %q2:2 = quantum.unitary(%p2 : memref<4x4xcomplex<f64>>) %q1, %q1 : !quantum.bit, !quantum.bit

    // CHECK: [[st1:%.+]] = llvm.insertvalue %arg0
    // CHECK: [[st2:%.+]] = llvm.insertvalue %arg0, [[st1]]
    // CHECK: return [[st2]]
    return %q2#0, %q2#1 : !quantum.bit, !quantum.bit
}

// -----

/////////////////
// Observables //
/////////////////

// CHECK-LABEL: @compbasis
func.func @compbasis(%q : !quantum.bit) {

    // CHECK: builtin.unrealized_conversion_cast %arg0
    quantum.compbasis %q : !quantum.obs
    // CHECK: builtin.unrealized_conversion_cast %arg0, %arg0
    quantum.compbasis %q, %q : !quantum.obs

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr<struct<"Qubit", opaque>>) -> i64

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

// CHECK: llvm.func @__catalyst__qis__HermitianObs(!llvm.ptr<struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>>, i64, ...) -> i64

// CHECK-LABEL: @hermitian
func.func @hermitian(%q : !quantum.bit, %p1 : memref<2x2xcomplex<f64>>, %p2 : memref<4x4xcomplex<f64>>) {
    // Only check the last members of the deconstructed memref struct being inserted.
    // CHECK: [[m1:%.+]] = llvm.insertvalue %arg7
    // CHECK: [[m2:%.+]] = llvm.insertvalue %arg14

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[c1_1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf1:%.+]] = llvm.alloca [[c1_1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: llvm.store [[m1]], [[buf1]]
    // CHECK: llvm.call @__catalyst__qis__HermitianObs([[buf1]], [[c1]], %arg0)
    quantum.hermitian(%p1 : memref<2x2xcomplex<f64>>) %q : !quantum.obs
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: [[c1_2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf2:%.+]] = llvm.alloca [[c1_2]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: llvm.store [[m2]], [[buf2]]
    // CHECK: llvm.call @__catalyst__qis__HermitianObs([[buf2]], [[c2]], %arg0, %arg0)
    quantum.hermitian(%p2 : memref<4x4xcomplex<f64>>) %q, %q : !quantum.obs

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

// CHECK: llvm.func @__catalyst__qis__HamiltonianObs(!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64, ...) -> i64

// CHECK-LABEL: @hamiltonian
func.func @hamiltonian(%obs : !quantum.obs, %p1 : memref<1xf64>, %p2 : memref<3xf64>) {
    // Only check the last members of the deconstructed memref struct being inserted.
    // CHECK: [[v1:%.+]] = llvm.insertvalue %arg5
    // CHECK: [[v2:%.+]] = llvm.insertvalue %arg10

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[c1_1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf1:%.+]] = llvm.alloca [[c1_1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.store [[v1]], [[buf1]]
    // CHECK: llvm.call @__catalyst__qis__HamiltonianObs([[buf1]], [[c1]], %arg0)
    quantum.hamiltonian(%p1 : memref<1xf64>) %obs : !quantum.obs
    // CHECK: [[c3:%.+]] = llvm.mlir.constant(3 : i64)
    // CHECK: [[c1_2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[buf2:%.+]] = llvm.alloca [[c1_2]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.store [[v2]], [[buf2]]
    // CHECK: llvm.call @__catalyst__qis__HamiltonianObs([[buf2]], [[c3]], %arg0, %arg0, %arg0)
    quantum.hamiltonian(%p2 : memref<3xf64>) %obs, %obs, %obs : !quantum.obs

    return
}

// -----

//////////////////
// Measurements //
//////////////////

// CHECK: llvm.func @__catalyst__qis__Measure(!llvm.ptr<struct<"Qubit", opaque>>, i32) -> !llvm.ptr<struct<"Result", opaque>>

// CHECK-LABEL: @measure
func.func @measure(%q : !quantum.bit) -> !quantum.bit {

    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(-1 : i32) : i32

    // CHECK: llvm.call @__catalyst__qis__Measure(%arg0, [[postselect]])
    %res, %new_q = quantum.measure %q : i1, !quantum.bit

    // CHECK: return %arg0
    return %new_q : !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__Measure(!llvm.ptr<struct<"Qubit", opaque>>, i32) -> !llvm.ptr<struct<"Result", opaque>>

// CHECK-LABEL: @measure
func.func @measure(%q : !quantum.bit) -> !quantum.bit {

    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.call @__catalyst__qis__Measure(%arg0, [[postselect]])
    %res, %new_q = quantum.measure %q {postselect = 0 : i32} : i1, !quantum.bit

    // CHECK: return %arg0
    return %new_q : !quantum.bit
}

// -----

// CHECK: llvm.func @__catalyst__qis__Sample(!llvm.ptr<struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>>, i64, i64, ...)

// CHECK-LABEL: @sample
func.func @sample(%q : !quantum.bit) {

    %o1 = quantum.compbasis %q : !quantum.obs
    %o2 = quantum.compbasis %q, %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[c1000:%.+]] = llvm.mlir.constant(1000 : i64)
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Sample([[ptr]], [[c1000]], [[c1]], %arg0)
    %alloc1 = memref.alloc() : memref<1000x1xf64>
    quantum.sample %o1 in(%alloc1 : memref<1000x1xf64>) {shots = 1000 : i64}
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: [[c2000:%.+]] = llvm.mlir.constant(2000 : i64)
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__Sample([[ptr]], [[c2000]], [[c2]], %arg0, %arg0)
    %alloc2 = memref.alloc() : memref<2000x2xf64>
    quantum.sample %o2 in(%alloc2 : memref<2000x2xf64>) {shots = 2000 : i64}

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__Counts(!llvm.ptr<struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>>, i64, i64, ...)

// CHECK-LABEL: @counts
func.func @counts(%q : !quantum.bit) {

    %o1 = quantum.compbasis %q : !quantum.obs
    %o2 = quantum.compbasis %q, %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c1000:%.+]] = llvm.mlir.constant(1000 : i64)
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Counts([[ptr]], [[c1000]], [[c1]], %arg0)
    %in_eigvals1 = memref.alloc() : memref<2xf64>
    %in_counts1 = memref.alloc() : memref<2xi64>
    quantum.counts %o1 in(%in_eigvals1 : memref<2xf64>, %in_counts1 : memref<2xi64>) {shots = 1000 : i64}
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c2000:%.+]] = llvm.mlir.constant(2000 : i64)
    // CHECK: [[c2:%.+]] = llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @__catalyst__qis__Counts([[ptr]], [[c2000]], [[c2]], %arg0, %arg0)
    %in_eigvals2 = memref.alloc() : memref<4xf64>
    %in_counts2 = memref.alloc() : memref<4xi64>
    quantum.counts %o2 in(%in_eigvals2 : memref<4xf64>, %in_counts2 : memref<4xi64>) {shots = 2000 : i64}

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

// CHECK: llvm.func @__catalyst__qis__Probs(!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64, ...)

// CHECK-LABEL: @probs
func.func @probs(%q : !quantum.bit) {

    %o1 = quantum.compbasis %q : !quantum.obs
    %o2 = quantum.compbasis %q, %q, %q, %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @__catalyst__qis__Probs([[ptr]], [[c1]], %arg0)
    %alloc1 = memref.alloc() : memref<2xf64>
    quantum.probs %o1 in(%alloc1 : memref<2xf64>)
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c4:%.+]] = llvm.mlir.constant(4 : i64)
    // CHECK: llvm.call @__catalyst__qis__Probs([[ptr]], [[c4]], %arg0, %arg0, %arg0, %arg0)
    %alloc2 = memref.alloc() : memref<16xf64>
    quantum.probs %o2 in(%alloc2 : memref<16xf64>)

    return
}

// -----

// CHECK: llvm.func @__catalyst__qis__State(!llvm.ptr<struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>>, i64, ...)

// CHECK-LABEL: @state
func.func @state(%q : !quantum.bit) {
    // CHECK: [[qb:%.+]] = builtin.unrealized_conversion_cast %arg0

    %o1 = quantum.compbasis %q : !quantum.obs
    %o2 = quantum.compbasis %q, %q, %q, %q : !quantum.obs

    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @__catalyst__qis__State([[ptr]], [[c0]])
    %alloc1 = memref.alloc() : memref<2xcomplex<f64>>
    quantum.state %o1 in(%alloc1 : memref<2xcomplex<f64>>)
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[c0:%.+]] = llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @__catalyst__qis__State([[ptr]], [[c0]])
    %alloc2 = memref.alloc() : memref<16xcomplex<f64>>
    quantum.state %o2 in(%alloc2: memref<16xcomplex<f64>>)

    return
}
