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

// RUN: quantum-opt %s --lower-mitigation --split-input-file --verify-diagnostics | FileCheck %s

// Random local folding reproduces Mitiq's `fold_gates_at_random`. The fold count
// `(scale_factor - 1) / 2` arrives as an f64. Every gate is folded `base = floor(count)`
// times unconditionally, and then exactly `k = round((count - base) * n)` of the `n` gates
// are folded once more, chosen uniformly without replacement. The subset is drawn at run
// time with selection sampling: gate `i` is folded when
// `__catalyst__rt__random_double() < (k - chosen) / (n - i)`, threading `chosen` between
// gates. For odd-integer scale factors `k == 0`, so this matches `local-all`.

// CHECK: func.func private @__catalyst__rt__random_double() -> f64

// CHECK-LABEL:   func.func private @circuit.folded(%arg0: f64) -> tensor<f64> {
    // CHECK-DAG:   [[idx0:%.+]] = index.constant 0
    // CHECK-DAG:   [[idx1:%.+]] = index.constant 1
    // CHECK-DAG:   [[c0i64:%.+]] = arith.constant 0 : i64
    // CHECK-DAG:   [[c1i64:%.+]] = arith.constant 1 : i64
    // CHECK-DAG:   [[half:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK-DAG:   [[ngates:%.+]] = arith.constant 2.000000e+00 : f64
    // Split the f64 fold count into base + remainder and compute k = round(delta * n).
    // CHECK:   [[baseI:%.+]] = arith.fptosi %arg0 : f64 to i64
    // CHECK:   [[baseIdx:%.+]] = index.casts [[baseI]] : i64 to index
    // CHECK:   [[baseF:%.+]] = arith.sitofp [[baseI]] : i64 to f64
    // CHECK:   [[delta:%.+]] = arith.subf %arg0, [[baseF]] : f64
    // CHECK:   [[deltaN:%.+]] = arith.mulf [[delta]], [[ngates]] : f64
    // CHECK:   [[rounded:%.+]] = arith.addf [[deltaN]], [[half]] : f64
    // CHECK:   [[k:%.+]] = arith.fptosi [[rounded]] : f64 to i64

    // CHECK:   [[qReg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK:   [[q0:%.+]] = quantum.extract [[qReg]][ 0] : !quantum.reg -> !quantum.bit
    // Gate 0: base folds.
    // CHECK:   [[q0base:%.+]] = scf.for %arg1 = [[idx0]] to [[baseIdx]] step [[idx1]] iter_args([[a0:%.+]] = [[q0]]) -> (!quantum.bit) {
    // CHECK:     [[h0:%.+]] = quantum.custom "Hadamard"() [[a0]] : !quantum.bit
    // CHECK:     [[h0adj:%.+]] = quantum.custom "Hadamard"() [[h0]] adj : !quantum.bit
    // CHECK:     scf.yield [[h0adj]] : !quantum.bit
    // CHECK:   }
    // Gate 0: selection probability k / (n - 0), one extra fold if selected.
    // CHECK:   [[kf0:%.+]] = arith.sitofp [[k]] : i64 to f64
    // CHECK:   [[prob0:%.+]] = arith.divf [[kf0]], [[ngates]] : f64
    // CHECK:   [[rnd0:%.+]] = call @__catalyst__rt__random_double() : () -> f64
    // CHECK:   [[coin0:%.+]] = arith.cmpf olt, [[rnd0]], [[prob0]] : f64
    // CHECK:   [[if0:%.+]]:2 = scf.if [[coin0]] -> (!quantum.bit, i64) {
    // CHECK:     [[h0e:%.+]] = quantum.custom "Hadamard"() [[q0base]] : !quantum.bit
    // CHECK:     [[h0eadj:%.+]] = quantum.custom "Hadamard"() [[h0e]] adj : !quantum.bit
    // CHECK:     scf.yield [[h0eadj]], [[c1i64]] : !quantum.bit, i64
    // CHECK:   } else {
    // CHECK:     scf.yield [[q0base]], [[c0i64]] : !quantum.bit, i64
    // CHECK:   }
    // CHECK:   [[h0orig:%.+]] = quantum.custom "Hadamard"() [[if0]]#0 : !quantum.bit

    // CHECK:   [[q1:%.+]] = quantum.extract [[qReg]][ 1] : !quantum.reg -> !quantum.bit
    // Gate 1: base folds.
    // CHECK:   [[c01base:%.+]]:2 = scf.for %arg1 = [[idx0]] to [[baseIdx]] step [[idx1]] iter_args([[b0:%.+]] = [[h0orig]], [[b1:%.+]] = [[q1]]) -> (!quantum.bit, !quantum.bit) {
    // CHECK:     [[cn:%.+]]:2 = quantum.custom "CNOT"() [[b0]], [[b1]] : !quantum.bit, !quantum.bit
    // CHECK:     [[cnadj:%.+]]:2 = quantum.custom "CNOT"() [[cn]]#0, [[cn]]#1 adj : !quantum.bit, !quantum.bit
    // CHECK:     scf.yield [[cnadj]]#0, [[cnadj]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   }
    // Gate 1: selection probability (k - chosen) / (n - 1); n - 1 == 1, so the divide folds away.
    // CHECK:   [[need1:%.+]] = arith.subi [[k]], [[if0]]#1 : i64
    // CHECK:   [[need1f:%.+]] = arith.sitofp [[need1]] : i64 to f64
    // CHECK:   [[rnd1:%.+]] = call @__catalyst__rt__random_double() : () -> f64
    // CHECK:   [[coin1:%.+]] = arith.cmpf olt, [[rnd1]], [[need1f]] : f64
    // CHECK:   [[if1:%.+]]:3 = scf.if [[coin1]] -> (!quantum.bit, !quantum.bit, i64) {
    // CHECK:     [[cne:%.+]]:2 = quantum.custom "CNOT"() [[c01base]]#0, [[c01base]]#1 : !quantum.bit, !quantum.bit
    // CHECK:     [[cneadj:%.+]]:2 = quantum.custom "CNOT"() [[cne]]#0, [[cne]]#1 adj : !quantum.bit, !quantum.bit
    // CHECK:     [[inc:%.+]] = arith.addi [[if0]]#1, [[c1i64]] : i64
    // CHECK:     scf.yield [[cneadj]]#0, [[cneadj]]#1, [[inc]] : !quantum.bit, !quantum.bit, i64
    // CHECK:   } else {
    // CHECK:     scf.yield [[c01base]]#0, [[c01base]]#1, [[if0]]#1 : !quantum.bit, !quantum.bit, i64
    // CHECK:   }
    // CHECK:   [[cnorig:%.+]]:2 = quantum.custom "CNOT"() [[if1]]#0, [[if1]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   [[obs:%.+]] = quantum.namedobs [[cnorig]]#0[ PauliY] : !quantum.obs
    // CHECK:   [[result:%.+]] = quantum.expval [[obs]] : f64

//CHECK-LABEL: func.func @circuit() -> tensor<f64> attributes {qnode} {
func.func @circuit() -> tensor<f64> attributes {qnode} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %2 : !quantum.bit, !quantum.bit
    %3 = quantum.namedobs %out_qubits_0#0[ PauliY] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %from_elements = tensor.from_elements %4 : tensor<f64>
    %5 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
}

func.func @mitigated_circuit() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    %0 = mitigation.zne @circuit() folding (random) numFolds (%numFolds : tensor<3xf64>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}
