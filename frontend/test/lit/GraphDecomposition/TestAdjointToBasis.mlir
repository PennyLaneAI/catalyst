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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testHadamard=1.0 alt-decomps=Adjoint(testU){}{wires:1}{}=adj_u,Adjoint(testHadamard){}{wires:1}{}=adj_h})' %s | FileCheck %s

// CHECK-LABEL: func.func @self_adjoint(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[O:%.+]] = quantum.custom "testHadamard"() [[Q]] : !quantum.bit
  // CHECK: return [[O]]
  %out = quantum.custom "testHadamard"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// CHECK-LABEL: func.func @distribution(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @distribution(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[A:%.+]] = quantum.custom "testHadamard"() [[Q]] : !quantum.bit
  // CHECK: [[B:%.+]] = quantum.custom "testHadamard"() [[A]] : !quantum.bit
  // CHECK: return [[B]]
  %out = quantum.custom "testU"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testHadamard){}{wires:1}{}",
    resources = {operations = {"testHadamard{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "testHadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @adj_u(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testU){}{wires:1}{}",
    resources = {operations = {"Adjoint(testHadamard){}{wires:1}{}" = 2 : i64}} } {
  %a = quantum.custom "testHadamard"() %q adj : !quantum.bit
  %b = quantum.custom "testHadamard"() %a adj : !quantum.bit
  return %b : !quantum.bit
}
