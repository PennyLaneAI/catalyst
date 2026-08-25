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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRZ=1.0 alt-decomps=Adjoint(testRZ{0:[f64]}{wires:1}{})=adj_rz})' %s | FileCheck %s

// CHECK-LABEL: func.func @parametric(
// CHECK-SAME:  [[Q:%.+]]: !quantum.bit, [[T:%.+]]: f64
func.func @parametric(%q: !quantum.bit, %theta: f64) -> !quantum.bit {
  // CHECK: [[NEG:%.+]] = arith.negf [[T]] : f64
  // CHECK: [[O:%.+]] = quantum.custom "testRZ"([[NEG]]) [[Q]] : !quantum.bit
  // CHECK: return [[O]]
  %out = quantum.custom "testRZ"(%theta) %q adj : !quantum.bit
  return %out: !quantum.bit
}

func.func private @adj_rz(%theta: f64, %q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testRZ{0:[f64]}{wires:1}{})",
    resources = {operations = {"testRZ{0:[f64]}{wires:1}{}" = 1 : i64}} } {
  %neg = arith.negf %theta : f64
  %o = quantum.custom "testRZ"(%neg) %q : !quantum.bit
  return %o : !quantum.bit
}
