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

// RUN: not --crash catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module( graph-decomposition{gate-set=PauliX=1.0 bytecode-rules="%BYTECODE_PATH"})' %s 2>&1 | FileCheck %s

func.func @circuit(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    %pi = arith.constant 3.14 : f64
    %out:3 = quantum.paulirot ["X", "Z", "Y"](%pi) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: GraphSolverFailedError
    // CHECK: Decomposition rule not found for operator 'paulirot
    return
}
