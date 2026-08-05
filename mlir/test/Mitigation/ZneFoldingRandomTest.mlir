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
// gates.
//
// Each test case below checks one aspect of the lowering on the same two-gate circuit
// (a Hadamard and a CNOT).

// Test the runtime PRNG is declared and the f64 fold count is split in the entry
// block into the integer base count and the number of extra folds `k = round(delta * n)`.

// CHECK: func.func private @__catalyst__rt__random_double() -> f64

// CHECK-LABEL: func.func private @counts.folded(%arg0: f64)
    // CHECK-DAG:   [[half:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK-DAG:   [[ngates:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK:   [[baseI:%.+]] = arith.fptosi %arg0 : f64 to i64
    // CHECK:   [[baseF:%.+]] = arith.sitofp [[baseI]] : i64 to f64
    // CHECK:   [[delta:%.+]] = arith.subf %arg0, [[baseF]] : f64
    // CHECK:   [[deltaN:%.+]] = arith.mulf [[delta]], [[ngates]] : f64
    // CHECK:   [[rounded:%.+]] = arith.addf [[deltaN]], [[half]] : f64
    // CHECK:   {{%.+}} = arith.fptosi [[rounded]] : f64 to i64

func.func @counts() -> tensor<f64> attributes {qnode} {
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

func.func @counts_mitigated() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    %0 = mitigation.zne @counts() folding (random) numFolds (%numFolds : tensor<3xf64>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// Test per-gate folding structure. Each gate gets a loop of `base` unconditional
// folding pairs, followed by a coin flip `random_double() < k / n` guarding one extra
// folding pair; the original gate then consumes the result.

// CHECK-LABEL: func.func private @gate.folded(%arg0: f64)
    // CHECK-DAG:   [[c0i64:%.+]] = arith.constant 0 : i64
    // CHECK-DAG:   [[c1i64:%.+]] = arith.constant 1 : i64
    // CHECK-DAG:   [[ngates:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK:   [[baseI:%.+]] = arith.fptosi %arg0 : f64 to i64
    // CHECK:   [[baseIdx:%.+]] = index.casts [[baseI]] : i64 to index
    // CHECK:   [[k:%.+]] = arith.fptosi {{%.+}} : f64 to i64

    // Base folds: `base` unconditional folding pairs.
    // CHECK:   [[q0:%.+]] = quantum.extract {{%.+}}[ 0]
    // CHECK:   [[q0base:%.+]] = scf.for {{%.+}} = {{%.+}} to [[baseIdx]] step {{%.+}} iter_args([[a0:%.+]] = [[q0]]) -> (!quantum.bit) {
    // CHECK:     [[h0:%.+]] = quantum.custom "Hadamard"() [[a0]] : !quantum.bit
    // CHECK:     [[h0adj:%.+]] = quantum.custom "Hadamard"() [[h0]] adj : !quantum.bit
    // CHECK:     scf.yield [[h0adj]] : !quantum.bit
    // CHECK:   }

    // Extra fold with probability k / n, incrementing the `chosen` counter when taken.
    // CHECK:   [[kf:%.+]] = arith.sitofp [[k]] : i64 to f64
    // CHECK:   [[prob:%.+]] = arith.divf [[kf]], [[ngates]] : f64
    // CHECK:   [[rnd:%.+]] = call @__catalyst__rt__random_double() : () -> f64
    // CHECK:   [[coin:%.+]] = arith.cmpf olt, [[rnd]], [[prob]] : f64
    // CHECK:   [[if0:%.+]]:2 = scf.if [[coin]] -> (!quantum.bit, i64) {
    // CHECK:     [[h0e:%.+]] = quantum.custom "Hadamard"() [[q0base]] : !quantum.bit
    // CHECK:     [[h0eadj:%.+]] = quantum.custom "Hadamard"() [[h0e]] adj : !quantum.bit
    // CHECK:     scf.yield [[h0eadj]], [[c1i64]] : !quantum.bit, i64
    // CHECK:   } else {
    // CHECK:     scf.yield [[q0base]], [[c0i64]] : !quantum.bit, i64
    // CHECK:   }

    // The original gate consumes the (possibly folded) qubit.
    // CHECK:   {{%.+}} = quantum.custom "Hadamard"() [[if0]]#0 : !quantum.bit

func.func @gate() -> tensor<f64> attributes {qnode} {
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

func.func @gate_mitigated() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    %0 = mitigation.zne @gate() folding (random) numFolds (%numFolds : tensor<3xf64>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// Test selection sampling without replacement. The `chosen` counter yielded by the
// first gate's scf.if feeds the second gate's selection probability
// `(k - chosen) / (n - 1)` and is incremented inside its then-branch, guaranteeing
// exactly `k` extra folds overall.

// CHECK-LABEL: func.func private @thread.folded(%arg0: f64)
    // CHECK-DAG:   [[c1i64:%.+]] = arith.constant 1 : i64
    // CHECK:   [[rounded:%.+]] = arith.addf {{%.+}}, {{%.+}} : f64
    // CHECK:   [[k:%.+]] = arith.fptosi [[rounded]] : f64 to i64

    // First gate: yields the updated `chosen` count as its last result.
    // CHECK:   [[if0:%.+]]:2 = scf.if {{%.+}} -> (!quantum.bit, i64)

    // Second gate: needs `k - chosen` more picks out of the one remaining gate
    // (the division by `n - 1 == 1` is folded away).
    // CHECK:   [[need1:%.+]] = arith.subi [[k]], [[if0]]#1 : i64
    // CHECK:   [[need1f:%.+]] = arith.sitofp [[need1]] : i64 to f64
    // CHECK:   [[rnd1:%.+]] = call @__catalyst__rt__random_double() : () -> f64
    // CHECK:   [[coin1:%.+]] = arith.cmpf olt, [[rnd1]], [[need1f]] : f64
    // CHECK:   [[if1:%.+]]:3 = scf.if [[coin1]] -> (!quantum.bit, !quantum.bit, i64) {
    // CHECK:     [[inc:%.+]] = arith.addi [[if0]]#1, [[c1i64]] : i64
    // CHECK:     scf.yield {{%.+}}, {{%.+}}, [[inc]] : !quantum.bit, !quantum.bit, i64
    // CHECK:   } else {
    // CHECK:     scf.yield {{%.+}}, {{%.+}}, [[if0]]#1 : !quantum.bit, !quantum.bit, i64
    // CHECK:   }
    // CHECK:   {{%.+}}:2 = quantum.custom "CNOT"() [[if1]]#0, [[if1]]#1 : !quantum.bit, !quantum.bit

func.func @thread() -> tensor<f64> attributes {qnode} {
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

func.func @thread_mitigated() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    %0 = mitigation.zne @thread() folding (random) numFolds (%numFolds : tensor<3xf64>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// Test the mitigated caller extracts each (possibly fractional) f64 fold count and
// passes it straight to the folded circuit, unlike the integer folding methods which
// cast it to an index.

// CHECK-LABEL: func.func @caller_mitigated()
    // CHECK-DAG:   [[folds:%.+]] = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    // CHECK:   scf.for
    // CHECK:   [[count:%.+]] = tensor.extract [[folds]][{{%.+}}] : tensor<3xf64>
    // CHECK:   {{%.+}} = func.call @caller.folded([[count]]) : (f64) -> tensor<f64>

func.func @caller() -> tensor<f64> attributes {qnode} {
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

func.func @caller_mitigated() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[0.000000e+00, 5.000000e-01, 1.000000e+00]> : tensor<3xf64>
    %0 = mitigation.zne @caller() folding (random) numFolds (%numFolds : tensor<3xf64>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}
