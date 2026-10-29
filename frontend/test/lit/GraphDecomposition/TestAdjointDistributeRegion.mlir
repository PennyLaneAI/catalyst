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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Hadamard=1.0 alt-decomps=Adjoint(U){}{wires:1}{}=adj_u,Adjoint(Hadamard){}{wires:1}{}=adj_h})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=Hadamard=1.0 alt-decomps=Adjoint(U){}{wires:1}{}=adj_u,Adjoint(Hadamard){}{wires:1}{}=adj_h})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: func.func @distribute_region(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit
func.func @distribute_region(%q: !quantum.bit) -> !quantum.bit {
  // ALL-NOT: quantum.adjoint
  // CALL: %[[B:.*]] = call @adj_u_0(%[[Q]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[A:.*]] = quantum.custom "Hadamard"() %[[Q]] : !quantum.bit
  // INLINE: %[[B:.*]] = quantum.custom "Hadamard"() %[[A]] : !quantum.bit
  // ALL: return %[[B]]
  %out = quantum.custom "U"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// Adjoint(U) via distribution:
// ALL-LABEL: func.func private @adj_u
func.func private @adj_u(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(U){}{wires:1}{}",
    resources = {operations = {"Adjoint(Hadamard){}{wires:1}{}" = 2 : i64}} } {
  %out = quantum.adjoint(%q) : !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %a = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    %b = quantum.custom "Hadamard"() %a : !quantum.bit
    quantum.yield %b : !quantum.bit
  }
  return %out : !quantum.bit
}

// Self-adjoint:
func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(Hadamard){}{wires:1}{}",
    resources = {operations = {"Hadamard{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "Hadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

// CALL:func.func private @adj_u_0(%arg0: !quantum.bit) -> !quantum.bit {
// CALL:  %[[H1:.*]] = call @adj_h_1(%arg0) : (!quantum.bit) -> !quantum.bit
// CALL:  %[[H2:.*]] = call @adj_h_1(%[[H1]]) : (!quantum.bit) -> !quantum.bit
// CALL:  return %[[H2]] : !quantum.bit
//
// CALL:func.func private @adj_h_1(%arg0: !quantum.bit) -> !quantum.bit {
// CALL:  %[[H:.*]] = quantum.custom "Hadamard"() %arg0 : !quantum.bit
// CALL:  return %[[H]] : !quantum.bit
