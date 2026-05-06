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

// RUN: quantum-opt --convert-pbc-to-llvm --convert-quantum-to-llvm --split-input-file -verify-diagnostics %s | FileCheck %s

//////////////////////////////////////////
// Pauli-based Computational Operations //
//////////////////////////////////////////

// -----

// CHECK-LABEL: @test_ppr
module @test_ppr {
    // CHECK: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i1, i64, ...)
    // CHECK: llvm.mlir.global internal constant @pauli_word_XIZ("XIZ\00")
    func.func @ppr(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit, %pred : i1) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        // CHECK-DAG: [[ctrue:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: [[theta:%.+]] = llvm.mlir.constant(1.5
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XIZ : !llvm.ptr
        // CHECK-DAG: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
        // CHECK-DAG: [[zero:%.+]] = llvm.mlir.zero : !llvm.ptr
        // CHECK-DAG: [[numQubits:%.+]] = llvm.mlir.constant(3 : i64) : i64
        // CHECK: llvm.call @__catalyst__qis__PauliRot([[pauliPtr]], [[theta]], [[zero]], [[ctrue]], [[numQubits]], %arg0, %arg1, %arg2)
        %qs:3 = pbc.ppr ["X", "I", "Z"](4) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
        // CHECK: llvm.call @__catalyst__qis__PauliRot({{%.+}}, {{%.+}}, {{%.+}}, %arg3, {{%.+}}, %arg0, %arg1, %arg2)
        %out:3 = pbc.ppr ["X", "I", "Z"](4) %qs#0, %qs#1, %qs#2 cond(%pred) : !quantum.bit, !quantum.bit, !quantum.bit
        return %out#0, %out#1, %out#2 : !quantum.bit, !quantum.bit, !quantum.bit
    }
}

// -----


// CHECK-LABEL: @test_ppr_arbitrary
module @test_ppr_arbitrary {
    // CHECK: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i1, i64, ...)
    // CHECK: llvm.mlir.global internal constant @pauli_word_XZ("XZ\00")
    func.func @ppr_arbitrary(%q0 : !quantum.bit, %q1 : !quantum.bit, %theta : f64, %pred : i1) -> (!quantum.bit, !quantum.bit) {
        // CHECK-DAG: [[ctrue:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: [[CONST2:%.+]] = llvm.mlir.constant(2.000000e+00 : f64) : f64
        // CHECK-DAG: [[MUL:%.+]] = llvm.fmul %arg2, [[CONST2]] : f64
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XZ : !llvm.ptr
        // CHECK-DAG: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK-DAG: [[zero:%.+]] = llvm.mlir.zero : !llvm.ptr
        // CHECK-DAG: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: llvm.call @__catalyst__qis__PauliRot([[pauliPtr]], [[MUL]], [[zero]], [[ctrue]], [[numQubits]], %arg0, %arg1)
        %qs:2 = pbc.ppr.arbitrary ["X", "Z"](%theta) %q0, %q1 : !quantum.bit, !quantum.bit
        // CHECK: llvm.call @__catalyst__qis__PauliRot({{%.+}}, {{%.+}}, {{%.+}}, %arg3, {{%.+}}, %arg0, %arg1)
        %out:2 = pbc.ppr.arbitrary ["X", "Z"](%theta) %qs#0, %qs#1 cond(%pred) : !quantum.bit, !quantum.bit
        return %out#0, %out#1 : !quantum.bit, !quantum.bit
    }
}

// -----

// CHECK-LABEL: @test_ppm
module @test_ppm {
    // CHECK: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i1, !llvm.ptr, i1, i1, i64, ...) -> !llvm.ptr
    // CHECK: llvm.mlir.global internal constant @pauli_word_XY("XY\00")
    func.func @ppm(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
        // CHECK-DAG: [[ctrue:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: [[cfalse:%.+]] = llvm.mlir.constant(false) : i1
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XY : !llvm.ptr
        // CHECK-DAG: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK-DAG: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: [[resultPtr:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[pauliPtr]], [[cfalse]], [[pauliPtr]], [[cfalse]], [[ctrue]], [[numQubits]], %arg0, %arg1)
        // CHECK: [[mres:%.+]] = llvm.load [[resultPtr]] : !llvm.ptr -> i1
        %mres, %out:2 = pbc.ppm ["X", "Y"] %q0, %q1 : i1, !quantum.bit, !quantum.bit
        return %mres, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
    }
}

// -----

// CHECK-LABEL: @test_ppm_negative_basis
module @test_ppm_negative_basis {
    // CHECK: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i1, !llvm.ptr, i1, i1, i64, ...) -> !llvm.ptr
    // CHECK: llvm.mlir.global internal constant @pauli_word_XY("XY\00")
    func.func @ppm_negative_basis(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
        // CHECK-DAG: [[neg:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: [[switch:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XY : !llvm.ptr
        // CHECK-DAG: [[pauliPtr:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK-DAG: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: [[resultPtr:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[pauliPtr]], [[neg]], [[pauliPtr]], [[neg]], [[switch]], [[numQubits]], %arg0, %arg1)
        // CHECK: [[mres:%.+]] = llvm.load [[resultPtr]] : !llvm.ptr -> i1
        %mres, %out:2 = pbc.ppm ["X", "Y"](-) %q0, %q1 : i1, !quantum.bit, !quantum.bit
        return %mres, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
    }
}

// -----

// CHECK-LABEL: @test_select_ppm
module @test_select_ppm {
    // CHECK: llvm.func @__catalyst__qis__PauliMeasure(!llvm.ptr, i1, !llvm.ptr, i1, i1, i64, ...) -> !llvm.ptr
    // CHECK: llvm.mlir.global internal constant @pauli_word_XY("XY\00")
    func.func @select_ppm(%q0 : !quantum.bit, %q1 : !quantum.bit, %sel : i1) -> (i1, !quantum.bit, !quantum.bit) {
        // CHECK-DAG: [[cfalse:%.+]] = llvm.mlir.constant(false) : i1
        // CHECK-DAG: [[ctrue:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XY : !llvm.ptr
        // CHECK-DAG: [[pauliPtr0:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK-DAG: llvm.mlir.addressof @pauli_word_XZ : !llvm.ptr
        // CHECK-DAG: [[pauliPtr1:%.+]] = llvm.getelementptr inbounds {{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
        // CHECK-DAG: [[numQubits:%.+]] = llvm.mlir.constant(2 : i64) : i64
        // CHECK: [[resultPtr:%.+]] = llvm.call @__catalyst__qis__PauliMeasure([[pauliPtr0]], [[cfalse]], [[pauliPtr1]], [[ctrue]], %arg2, [[numQubits]], %arg0, %arg1)
        // CHECK: [[mres:%.+]] = llvm.load [[resultPtr]] : !llvm.ptr -> i1
        %mres, %out:2 = pbc.select.ppm (%sel ? ["X", "Y"] : ["X", "Z"](-)) %q0, %q1 : i1, !quantum.bit, !quantum.bit
        return %mres, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
    }
}
