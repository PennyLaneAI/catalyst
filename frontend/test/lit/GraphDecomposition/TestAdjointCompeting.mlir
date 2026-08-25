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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRZ=1.0 alt-decomps=Adjoint(testRot){}{wires:1}{}=[dedicated,distribute],Adjoint(testRZ){}{wires:1}{}=adj_rz})' %s | FileCheck %s

// CHECK-LABEL: func.func @competing(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @competing(%q: !quantum.bit) -> !quantum.bit {
  // CHECK-COUNT-1: quantum.custom "testRZ"()
  // CHECK-NOT: quantum.custom "testRZ"
  // CHECK: return
  %out = quantum.custom "testRot"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// pathway 1: Adjoint(testRot) -> testRZ (cost 1).
func.func private @dedicated(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testRot){}{wires:1}{}",
    resources = {operations = {"testRZ{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "testRZ"() %q : !quantum.bit
  return %o : !quantum.bit
}

// pathway 2: Adjoint(testRot) -> Adjoint(testRZ) Adjoint(testRZ) -> testRZ testRZ (cost 2).
func.func private @distribute(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testRot){}{wires:1}{}",
    resources = {operations = {"Adjoint(testRZ){}{wires:1}{}" = 2 : i64}} } {
  %a = quantum.custom "testRZ"() %q adj : !quantum.bit
  %b = quantum.custom "testRZ"() %a adj : !quantum.bit
  return %b : !quantum.bit
}

func.func private @adj_rz(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testRZ){}{wires:1}{}",
    resources = {operations = {"testRZ{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "testRZ"() %q : !quantum.bit
  return %o : !quantum.bit
}
