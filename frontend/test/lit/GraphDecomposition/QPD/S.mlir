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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=PhaseShift=1.0})' %s | FileCheck %s

func.func @S() -> !quantum.bit {
  %reg = quantum.alloc(1): !quantum.reg
  %0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit

  // CHECK-NOT: quantum.custom "S"
  // CHECK: quantum.custom "PhaseShift"
  %out_qubits = quantum.custom "S"() %0 : !quantum.bit
  return %out_qubits : !quantum.bit
}
