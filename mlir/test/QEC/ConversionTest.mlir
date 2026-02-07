// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --convert-qec-to-llvm --convert-quantum-to-llvm --split-input-file -verify-diagnostics %s | FileCheck %s

//////////////////////////////////////////
// Pauli-based Computational Operations //
//////////////////////////////////////////

// -----

// CHECK-LABEL: @test_pauli_rot
module @test_pauli_rot {
    // CHECK: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i64, ...)
    // CHECK: llvm.mlir.global internal constant @pauli_word_X("X\00")
    func.func @pauli_rot(%q0 : !quantum.bit, %angle : f64) -> (!quantum.bit) {
        // CHECK: llvm.mlir.addressof @pauli_word_X : !llvm.ptr
        // CHECK: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
        // CHECK: [[numQubits:%.+]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: llvm.call @__catalyst__qis__PauliRot([[pauliPtr]], {{%.+}}, {{%.+}}, [[numQubits]], %arg0)
        %out = quantum.paulirot ["X"](%angle) %q0 : !quantum.bit
        return %out : !quantum.bit
    }
}

// -----


// CHECK-LABEL: @test_ppr
module @test_ppr {
    // CHECK: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i64, ...)
    // CHECK: llvm.mlir.global internal constant @pauli_word_XIZ("XIZ\00")
    func.func @ppr(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        // CHECK: llvm.mlir.addressof @pauli_word_XIZ : !llvm.ptr
        // CHECK: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
        // CHECK: [[theta:%.+]] = llvm.mlir.constant(1.5
        // CHECK: [[numQubits:%.+]] = llvm.mlir.constant(3 : i64) : i64
        // CHECK: llvm.call @__catalyst__qis__PauliRot([[pauliPtr]], [[theta]], {{%.+}}, [[numQubits]], %arg0, %arg1, %arg2)
        %out:3 = qec.ppr ["X", "I", "Z"](4) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
        return %out#0, %out#1, %out#2 : !quantum.bit, !quantum.bit, !quantum.bit
    }
}

// -----


// CHECK-LABEL: @test_ppr_arbitrary
module @test_ppr_arbitrary {
    // CHECK: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i64, ...)
    // CHECK: llvm.mlir.global internal constant @pauli_word_XZ("XZ\00")
    func.func @ppr_arbitrary(%q0 : !quantum.bit, %q1 : !quantum.bit, %theta : f64) -> (!quantum.bit, !quantum.bit) {
        // CHECK: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(2.000000e+00 : f64) : f64
        // CHECK: [[MUL:%.+]] = llvm.fmul %arg2, [[CONST2]] : f64
        // CHECK: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: llvm.call @__catalyst__qis__PauliRot([[pauliPtr]], [[MUL]], {{%.+}}, [[numQubits]], %arg0, %arg1)
        %out:2 = qec.ppr.arbitrary ["X", "Z"](%theta) %q0, %q1 : !quantum.bit, !quantum.bit
        return %out#0, %out#1 : !quantum.bit, !quantum.bit
    }
}

// -----

// CHECK-LABEL: @test_ppm
module @test_ppm {
    // CHECK: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i64, ...) -> !llvm.ptr
    // CHECK: llvm.mlir.global internal constant @pauli_word_XY("XY\00")
    func.func @ppm(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
        // CHECK: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: [[resultPtr:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[pauliPtr]], [[numQubits]], %arg0, %arg1)
        // CHECK: [[mres:%.+]] = llvm.load [[resultPtr]] : !llvm.ptr -> i1
        %mres, %out:2 = qec.ppm ["X", "Y"] %q0, %q1 : i1, !quantum.bit, !quantum.bit
        return %mres, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
    }
}

// -----

// CHECK-LABEL: @test_ppm_negative_basis
module @test_ppm_negative_basis {
    // CHECK: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i64, ...) -> !llvm.ptr
    // CHECK: llvm.mlir.global internal constant @pauli_word_XY("XY\00")
    func.func @ppm_negative_basis(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
        // CHECK: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: [[resultPtr:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[pauliPtr]], [[numQubits]], %arg0, %arg1)
        // CHECK: [[mres:%.+]] = llvm.load [[resultPtr]] : !llvm.ptr -> i1
        // CHECK: [[true:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK: [[mres_negated:%.+]] = llvm.xor [[mres]], [[true]] : i1
        %mres, %out:2 = qec.ppm ["X", "Y"](-1) %q0, %q1 : i1, !quantum.bit, !quantum.bit
        return %mres, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
    }
}

// -----
