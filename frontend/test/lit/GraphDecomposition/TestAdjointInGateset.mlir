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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Adjoint(T)=1.0 alt-decomps=S{}{wires:1}{}=s_to_adjt})' %s | FileCheck %s

// CHECK-LABEL: func.func @adjoint_in_gateset(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @adjoint_in_gateset(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[O:%.+]] = quantum.custom "T"() [[Q]] adj : !quantum.bit
  // CHECK: return [[O]]
  %out = quantum.custom "T"() %q adj : !quantum.bit
  return %out: !quantum.bit
}

// CHECK-LABEL: func.func @decompose_to_adjoint(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit
func.func @decompose_to_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // CHECK: [[O:%.+]] = quantum.custom "T"() [[Q]] adj : !quantum.bit
  // CHECK: return [[O]]
  %out = quantum.custom "S"() %q : !quantum.bit
  return %out: !quantum.bit
}

func.func private @s_to_adjt(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "S{}{wires:1}{}",
    resources = {operations = {"Adjoint(T{}{wires:1}{})" = 1 : i64}} } {
  %o = quantum.custom "T"() %q adj : !quantum.bit
  return %o : !quantum.bit
}
