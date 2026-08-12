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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Hadamard=1.0,MultiRZ=1.0})' %s | FileCheck %s

func.func @paulirot(%angle: f64, %q1: !quantum.bit, %q2: !quantum.bit) {
  // CHECK-NOT: quantum.paulirot
  // CHECK: Hadamard
  // CHECK: quantum.multirz
  // CHECK: Hadamard
  %qout:2 = quantum.paulirot ["Z", "X"](%angle) %q1, %q2 : !quantum.bit, !quantum.bit
  return
}
