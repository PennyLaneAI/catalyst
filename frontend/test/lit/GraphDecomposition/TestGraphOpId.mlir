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

// Test that graph-decomposition succeeds when using graphOpIds

// RUN: catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=testPauliX=1.0 alt-decomps=testHadamard=my_decomp})' %s | FileCheck %s

func.func @circuit(%q: !quantum.bit) -> !quantum.bit {
  // CHECK-NOT: testHadamard
  // CHECK: testPauliX
  // CHECK: testPauliX
  %out = quantum.custom "testHadamard"() %q: !quantum.bit
  return %out: !quantum.bit
}

func.func private @my_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="testHadamard[][1]{}"} {
  %q0 = quantum.custom "testPauliX"() %q : !quantum.bit
  %q1 = quantum.custom "testPauliX"() %q0 : !quantum.bit
  return %q1 : !quantum.bit
}
