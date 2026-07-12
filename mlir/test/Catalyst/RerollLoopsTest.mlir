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

// RUN: quantum-opt --reroll-loops="min-period=2 min-savings=4" --split-input-file %s | FileCheck %s

// A scalar chain of alternating ops rerolls into an scf.for threading one
// value. The first addf seeds the loop (its operand pattern differs), the
// trailing mulf stays as an epilogue.

// CHECK-LABEL: @scalar_chain
// CHECK:         %[[SEED:.+]] = arith.addf
// CHECK:         %[[FOR:.+]] = scf.for {{.*}} iter_args(%[[IT:.+]] = %[[SEED]]) -> (f64)
// CHECK:           %[[M:.+]] = arith.mulf %[[IT]],
// CHECK:           %[[A:.+]] = arith.addf %[[M]],
// CHECK:           scf.yield %[[A]] : f64
// CHECK:         %[[EPI:.+]] = arith.mulf %[[FOR]],
// CHECK:         return %[[EPI]]
func.func @scalar_chain(%arg0: f64, %c: f64) -> f64 {
  %0 = arith.addf %arg0, %c : f64
  %1 = arith.mulf %0, %c : f64
  %2 = arith.addf %1, %c : f64
  %3 = arith.mulf %2, %c : f64
  %4 = arith.addf %3, %c : f64
  %5 = arith.mulf %4, %c : f64
  %6 = arith.addf %5, %c : f64
  %7 = arith.mulf %6, %c : f64
  %8 = arith.addf %7, %c : f64
  %9 = arith.mulf %8, %c : f64
  %10 = arith.addf %9, %c : f64
  %11 = arith.mulf %10, %c : f64
  %12 = arith.addf %11, %c : f64
  %13 = arith.mulf %12, %c : f64
  %14 = arith.addf %13, %c : f64
  %15 = arith.mulf %14, %c : f64
  return %15 : f64
}

// -----

// A repeated gate sequence threading two qubits rerolls with both qubit
// values as iter_args; the rotation angle is loop-invariant. The pass may pick
// any rotation of the repeated window (here the run starts at the first CNOT),
// leaving a prologue/epilogue outside the loop.

// CHECK-LABEL: @gate_sequence
// CHECK:         quantum.alloc
// CHECK:         %[[FOR:.+]]:2 = scf.for {{.*}} iter_args(%[[Q0:.+]] = %{{.+}}, %[[Q1:.+]] = %{{.+}}) -> (!quantum.bit, !quantum.bit)
// CHECK:           %[[CNOT:.+]]:2 = quantum.custom "CNOT"() %[[Q0]], %[[Q1]]
// CHECK:           %[[H:.+]] = quantum.custom "Hadamard"() %[[CNOT]]#0
// CHECK:           %[[RZ:.+]] = quantum.custom "RZ"(%{{.+}}) %[[CNOT]]#1
// CHECK:           scf.yield %[[H]], %[[RZ]]
// CHECK:         %[[LAST:.+]]:2 = quantum.custom "CNOT"() %[[FOR]]#0, %[[FOR]]#1
// CHECK:         quantum.insert %{{.+}}[ 0], %[[LAST]]#0
func.func @gate_sequence(%theta: f64) -> !quantum.reg {
  %r0 = quantum.alloc( 2) : !quantum.reg
  %q0 = quantum.extract %r0[ 0] : !quantum.reg -> !quantum.bit
  %q1 = quantum.extract %r0[ 1] : !quantum.reg -> !quantum.bit

  %h0 = quantum.custom "Hadamard"() %q0 : !quantum.bit
  %z0 = quantum.custom "RZ"(%theta) %q1 : !quantum.bit
  %c0:2 = quantum.custom "CNOT"() %h0, %z0 : !quantum.bit, !quantum.bit

  %h1 = quantum.custom "Hadamard"() %c0#0 : !quantum.bit
  %z1 = quantum.custom "RZ"(%theta) %c0#1 : !quantum.bit
  %c1:2 = quantum.custom "CNOT"() %h1, %z1 : !quantum.bit, !quantum.bit

  %h2 = quantum.custom "Hadamard"() %c1#0 : !quantum.bit
  %z2 = quantum.custom "RZ"(%theta) %c1#1 : !quantum.bit
  %c2:2 = quantum.custom "CNOT"() %h2, %z2 : !quantum.bit, !quantum.bit

  %h3 = quantum.custom "Hadamard"() %c2#0 : !quantum.bit
  %z3 = quantum.custom "RZ"(%theta) %c2#1 : !quantum.bit
  %c3:2 = quantum.custom "CNOT"() %h3, %z3 : !quantum.bit, !quantum.bit

  %r1 = quantum.insert %r0[ 0], %c3#0 : !quantum.reg, !quantum.bit
  %r2 = quantum.insert %r1[ 1], %c3#1 : !quantum.reg, !quantum.bit
  return %r2 : !quantum.reg
}

// -----

// Iterations with *different* angle values must not be rerolled: the varying
// operand fails loop-invariance verification.

// CHECK-LABEL: @varying_angles
// CHECK-NOT:     scf.for
func.func @varying_angles(%t0: f64, %t1: f64, %t2: f64, %t3: f64) -> !quantum.reg {
  %r0 = quantum.alloc( 1) : !quantum.reg
  %q0 = quantum.extract %r0[ 0] : !quantum.reg -> !quantum.bit
  %a = quantum.custom "RZ"(%t0) %q0 : !quantum.bit
  %b = quantum.custom "RX"(%t0) %a : !quantum.bit
  %c = quantum.custom "RZ"(%t1) %b : !quantum.bit
  %d = quantum.custom "RX"(%t1) %c : !quantum.bit
  %e = quantum.custom "RZ"(%t2) %d : !quantum.bit
  %f = quantum.custom "RX"(%t2) %e : !quantum.bit
  %g = quantum.custom "RZ"(%t3) %f : !quantum.bit
  %h = quantum.custom "RX"(%t3) %g : !quantum.bit
  %r1 = quantum.insert %r0[ 0], %h : !quantum.reg, !quantum.bit
  return %r1 : !quantum.reg
}

// -----

// Results used outside the repeat (other than the final threaded values) block
// rerolling: here every intermediate feeds the final sum.

// CHECK-LABEL: @external_uses
// CHECK-NOT:     scf.for
func.func @external_uses(%arg0: f64, %c: f64) -> f64 {
  %0 = arith.addf %arg0, %c : f64
  %1 = arith.mulf %0, %c : f64
  %2 = arith.addf %1, %c : f64
  %3 = arith.mulf %2, %c : f64
  %4 = arith.addf %3, %c : f64
  %5 = arith.mulf %4, %c : f64
  %6 = arith.addf %5, %c : f64
  %7 = arith.mulf %6, %c : f64
  %s0 = arith.addf %1, %3 : f64
  %s1 = arith.addf %s0, %5 : f64
  %s2 = arith.addf %s1, %7 : f64
  return %s2 : f64
}
