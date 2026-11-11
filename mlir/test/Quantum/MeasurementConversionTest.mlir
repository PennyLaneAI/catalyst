// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --convert-quantum-to-llvm %s | FileCheck %s

// CHECK-DAG: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i1, !llvm.ptr, i1, i1, i64, ...) -> !llvm.ptr
// CHECK-DAG: llvm.mlir.global internal constant @pauli_word_IXYZ("IXYZ\00")
// CHECK-DAG: llvm.mlir.global internal constant @pauli_word_Z("Z\00")

// CHECK-LABEL: llvm.func @pauli_measurement
func.func @pauli_measurement(%q0: !quantum.bit, %q1: !quantum.bit,
                            %q2: !quantum.bit, %q3: !quantum.bit)
    -> (i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  // CHECK: llvm.mlir.addressof @pauli_word_IXYZ
  // CHECK: [[WORD:%.+]] = llvm.getelementptr inbounds
  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1
  // CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1
  // CHECK: [[COUNT:%.+]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: [[PTR:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[WORD]], [[FALSE]], [[WORD]], [[FALSE]], [[TRUE]], [[COUNT]], %arg0, %arg1, %arg2, %arg3)
  // CHECK: [[M:%.+]] = llvm.load [[PTR]] : !llvm.ptr -> i1
  %m, %out:4 = quantum.measurement "PauliMeasure"() qubits(%q0, %q1, %q2, %q3)
      static_data = {pauli_word = "IXYZ", postselect = none}
      : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
  // CHECK: llvm.insertvalue [[M]]
  // CHECK: llvm.insertvalue %arg0
  // CHECK: llvm.insertvalue %arg1
  // CHECK: llvm.insertvalue %arg2
  // CHECK: llvm.insertvalue %arg3
  return %m, %out#0, %out#1, %out#2, %out#3
      : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: llvm.func @default_postselection
func.func @default_postselection(%q: !quantum.bit) -> (i1, !quantum.bit) {
  // CHECK: llvm.call @__catalyst__qis__PauliMeasure
  // CHECK: llvm.load {{%.+}} : !llvm.ptr -> i1
  %m, %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "Z", postselect = -1 : si64}
      : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// CHECK-LABEL: llvm.func @discarded_measurement_result
func.func @discarded_measurement_result(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: llvm.call @__catalyst__qis__PauliMeasure
  // CHECK-NOT: llvm.load
  %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "Z", meas_uid = "direct"} : !quantum.bit
  // CHECK: llvm.return %arg0 : !llvm.ptr
  return %out : !quantum.bit
}
