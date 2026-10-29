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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testHadamard=1.0 alt-decomps=Adjoint(testU){}{wires:1}{}=adj_u,Adjoint(testHadamard){}{wires:1}{}=adj_h})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testHadamard=1.0 alt-decomps=Adjoint(testU){}{wires:1}{}=adj_u,Adjoint(testHadamard){}{wires:1}{}=adj_h})' %s | FileCheck %s --check-prefixes=ALL,INLINE


// ALL-LABEL: func.func @self_adjoint(
// ALL-SAME:  [[Q:%.+]]: !quantum.bit
func.func @self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CALL: [[O:%.+]] = call @adj_h_1([[Q]])
  // INLINE: [[O:%.+]] = quantum.custom "testHadamard"() [[Q]] : !quantum.bit
  // ALL: return [[O]]
  %out = quantum.custom "testHadamard"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// ALL-LABEL: func.func @distribution(
// ALL-SAME:  [[Q:%.+]]: !quantum.bit
func.func @distribution(%q: !quantum.bit) -> !quantum.bit {
  // CALL: [[B:%.+]] = call @adj_u_0([[Q]])
  // INLINE: [[A:%.+]] = quantum.custom "testHadamard"() [[Q]] : !quantum.bit
  // INLINE: [[B:%.+]] = quantum.custom "testHadamard"() [[A]] : !quantum.bit
  // ALL: return [[B]]
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

// CALL: func.func private @adj_u_0
// CALL:   call @adj_h_1
// CALL:   call @adj_h_1
//
// CALL: func.func private @adj_h_1
// CALL:   quantum.custom "testHadamard"()
