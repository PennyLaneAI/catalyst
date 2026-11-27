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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRZ=1.0})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testRZ=1.0})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: func.func private @test_func.adjoint(
// ALL-SAME:  [[Q:%.+]]: !quantum.bit
// CALL: [[O:%.+]] = call @adj_rz_0([[Q]])
// INLINE: [[O:%.+]] = quantum.custom "testRZ"() [[Q]]
// ALL: return [[O]]

func.func @test_func(%q: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "testRZ"() %q : !quantum.bit
    return %out: !quantum.bit
}

// ALL-LABEL: func.func @circuit(
func.func @circuit() attributes {quantum.node} {
    %0 = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // ALL-NOT: quantum.adjoint
    // ALL: call @test_func.adjoint
    // ALL: return
    %out = quantum.adjoint(%q) : !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %a = func.call @test_func(%arg0) : (!quantum.bit) -> !quantum.bit
        quantum.yield %a : !quantum.bit
    }

    %1 = quantum.insert %0[0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %1 : !quantum.reg
    return
}

func.func private @adj_rz(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testRZ){}{wires:1}{}",
    resources = {operations = {"testRZ{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "testRZ"() %q : !quantum.bit
  return %o : !quantum.bit
}

// CALL: func.func private @adj_rz_0
