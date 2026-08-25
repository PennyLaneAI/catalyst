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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Hadamard=1.0 alt-decomps=Adjoint(U{}{wires:1}{})=adj_u,Adjoint(Hadamard{}{wires:1}{})=adj_h})' %s | FileCheck %s

// CHECK-LABEL: func.func @self_adjoint(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[O:%.+]] = quantum.custom "Hadamard"() [[Q]] : !quantum.bit
  // CHECK: return [[O]]
  %out = quantum.custom "Hadamard"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// CHECK-LABEL: func.func @distribution(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @distribution(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[A:%.+]] = quantum.custom "Hadamard"() [[Q]] : !quantum.bit
  // CHECK: [[B:%.+]] = quantum.custom "Hadamard"() [[A]] : !quantum.bit
  // CHECK: return [[B]]
  %out = quantum.custom "U"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(Hadamard{}{wires:1}{})",
    resources = {operations = {"Hadamard{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "Hadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @adj_u(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(U{}{wires:1}{})",
    resources = {operations = {"Adjoint(Hadamard{}{wires:1}{})" = 2 : i64}} } {
  %a = quantum.custom "Hadamard"() %q adj : !quantum.bit
  %b = quantum.custom "Hadamard"() %a adj : !quantum.bit
  return %b : !quantum.bit
}
