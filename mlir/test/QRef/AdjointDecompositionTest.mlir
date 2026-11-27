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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,CALL
// RUN: quantum-opt --decompose-lowering=inline-rule-body --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,INLINE


/// Self-adjoint basis gate: Adjoint(H) -> H (the modifier is dropped).
///
// ALL-LABEL: func.func @self_adjoint(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit
func.func @self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CALL: %[[O:.*]] = call @adj_h_0(%[[Q]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[O:.*]] = quantum.custom "Hadamard"() %[[Q]] : !quantum.bit
  // ALL: return %[[O]]
  %out = quantum.custom "Hadamard"() %q adj : !quantum.bit
  return %out : !quantum.bit
}

func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(Hadamard){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "Hadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

/// An adjoint op must NOT fall back to a plain base-name rule.
///
// ALL-LABEL: func.func @no_base_rule_fallback(
// ALL-SAME:  %[[T:.*]]: f64, %[[Q:.*]]: !quantum.bit
func.func @no_base_rule_fallback(%q: !quantum.bit, %theta: f64) -> !quantum.bit {
  // ALL: %[[O:.*]] = quantum.custom "RX"(%[[T]]) %[[Q]] adj : !quantum.bit
  // ALL-NOT: PauliX
  // ALL: return %[[O]]
  %out = quantum.custom "RX"(%theta) %q adj : !quantum.bit
  return %out : !quantum.bit
}

func.func private @plain_rx(%theta: f64, %q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "RX", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

/// Parametric adjoint of a basis gate: Adjoint(RZ)(theta) -> RZ(-theta).
///
// ALL-LABEL: func.func @parametric_negation(
// ALL-SAME:  %[[T:.*]]: f64, %[[Q:.*]]: !quantum.bit
func.func @parametric_negation(%q: !quantum.bit, %theta: f64) -> !quantum.bit {
  // CALL: %[[O:.*]] = call @adj_rz_0(%[[T]], %[[Q]]) : (f64, !quantum.bit) -> !quantum.bit
  // INLINE: %[[NEG:.*]] = arith.negf %[[T]] : f64
  // INLINE: %[[O:.*]] = quantum.custom "RZ"(%[[NEG]]) %[[Q]] : !quantum.bit
  // ALL: return %[[O]]
  %out = quantum.custom "RZ"(%theta) %q adj : !quantum.bit
  return %out : !quantum.bit
}

func.func private @adj_rz(%theta: f64, %q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %neg = arith.negf %theta : f64
  %o = quantum.custom "RZ"(%neg) %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

/// Adjoint(Op) is a DISTINCT node from Op: with both a base rule (on the plain id) and an adjoint
/// rule (on the Adjoint(...) id) present, each op takes its own rule.
///
// ALL-LABEL: func.func @distinct_from_base(
// ALL-SAME:  %[[Q0:.*]]: !quantum.bit, %[[Q1:.*]]: !quantum.bit
func.func @distinct_from_base(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  // plain H takes the base rule -> PauliX
  // CALL: %[[A:.*]] = call @base_h_1(%[[Q0]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[A:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
  %a = quantum.custom "Hadamard"() %q0 : !quantum.bit

  // Adjoint(H) takes the adjoint rule -> PauliZ
  // CALL: %[[B:.*]] = call @adj_h2_0(%[[Q1]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[B:.*]] = quantum.custom "PauliZ"() %[[Q1]] : !quantum.bit
  %b = quantum.custom "Hadamard"() %q1 adj : !quantum.bit

  // ALL: return %[[A]], %[[B]]
  return %a, %b : !quantum.bit, !quantum.bit
}

func.func private @base_h(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Hadamard{}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @adj_h2(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(Hadamard){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliZ"() %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

/// Non-self-adjoint discrete gate: S is not its own inverse, so Adjoint(S) needs an explicit rule.
///
// ALL-LABEL: func.func @non_self_adjoint(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit
func.func @non_self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CALL: %[[O:.*]] = call @adj_s_0(%[[Q]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[C:.*]] = arith.constant -1.57{{.*}} : f64
  // INLINE: %[[O:.*]] = quantum.custom "PhaseShift"(%[[C]]) %[[Q]] : !quantum.bit
  // ALL: return %[[O]]
  %out = quantum.custom "S"() %q adj : !quantum.bit
  return %out : !quantum.bit
}

func.func private @adj_s(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(S){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %angle = arith.constant -1.5707963267948966 : f64
  %o = quantum.custom "PhaseShift"(%angle) %q : !quantum.bit
  return %o : !quantum.bit
}

// -----
/// Distribution: Adjoint of a composite decomposition reverses the sequence and adjoints each gate.
///
// ALL-LABEL: func.func @distribution(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit
func.func @distribution(%q: !quantum.bit) -> !quantum.bit {
  // CALL: %[[C:.*]] = call @adj_u_0(%[[Q]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[K:.*]] = arith.constant {{.*}} : f64
  // INLINE: %[[A:.*]] = quantum.custom "PauliX"() %[[Q]] : !quantum.bit
  // INLINE: %[[B:.*]] = quantum.custom "PhaseShift"(%[[K]]) %[[A]] : !quantum.bit
  // INLINE: %[[C:.*]] = quantum.custom "Hadamard"() %[[B]] : !quantum.bit
  // ALL: return %[[C]]
  %out = quantum.custom "U"() %q adj : !quantum.bit
  return %out : !quantum.bit
}

// CALL: func.func private @adj_u_0
// CALL:   call @adj_x_3
// CALL:   call @adj_t_2
// CALL:   call @adj_h3_1
// CALL: func.func private @adj_h3_1
// CALL:   quantum.custom "Hadamard"() %arg0
// CALL: func.func private @adj_t_2
// CALL:   %[[cst:.*]] = arith.constant -0.78539816339744828 : f64
// CALL:   quantum.custom "PhaseShift"(%[[cst]]) %arg0
// CALL: func.func private @adj_x_3(%arg0: !quantum.bit)
// CALL:   quantum.custom "PauliX"() %arg0

func.func private @adj_u(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(U){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %a = quantum.custom "PauliX"() %q adj : !quantum.bit
  %b = quantum.custom "T"() %a adj : !quantum.bit
  %c = quantum.custom "Hadamard"() %b adj : !quantum.bit
  return %c : !quantum.bit
}

func.func private @adj_h3(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(Hadamard){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "Hadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @adj_x(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(PauliX){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @adj_t(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(T){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %angle = arith.constant -0.78539816339744828 : f64
  %o = quantum.custom "PhaseShift"(%angle) %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

/// A decomposition of a NON-adjoint gate can itself emit adjoint gates.
///
// ALL-LABEL: func.func @decomp_produces_adjoint(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit
func.func @decomp_produces_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CALL: %[[C:.*]] = call @mygate_0(%[[Q]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[K:.*]] = arith.constant {{.*}} : f64
  // INLINE: %[[A:.*]] = quantum.custom "Hadamard"() %[[Q]] : !quantum.bit
  // INLINE: %[[B:.*]] = quantum.custom "PhaseShift"(%[[K]]) %[[A]] : !quantum.bit
  // INLINE: %[[C:.*]] = quantum.custom "Hadamard"() %[[B]] : !quantum.bit
  // ALL: return %[[C]]
  %out = quantum.custom "MyGate"() %q : !quantum.bit
  return %out : !quantum.bit
}

// CALL: func.func private @mygate_0
// CALL:   quantum.custom "Hadamard"()
// CALL:   call @adj_t2_1
// CALL:   quantum.custom "Hadamard"()
// CALL: func.func private @adj_t2_1
// CALL:   %[[cst:.*]] = arith.constant -0.78539816339744828 : f64
// CALL:   quantum.custom "PhaseShift"(%[[cst]])

func.func private @mygate(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "MyGate{}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %a = quantum.custom "Hadamard"() %q : !quantum.bit
  %b = quantum.custom "T"() %a adj : !quantum.bit
  %c = quantum.custom "Hadamard"() %b : !quantum.bit
  return %c : !quantum.bit
}

func.func private @adj_t2(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(T){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %angle = arith.constant -0.78539816339744828 : f64
  %o = quantum.custom "PhaseShift"(%angle) %q : !quantum.bit
  return %o : !quantum.bit
}
