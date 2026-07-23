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

// RUN: catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRX=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes FIRST

// RUN: catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRX=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"},graph-decomposition{gate-set=testRZ=1.0,testRY=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes SECOND

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // FIRST-NOT: testPauliX
    // FIRST: testRX

    // SECOND-NOT: testPauliX
    // SECOND-NOT: testRX
    // SECOND: testRZ
    // SECOND: testRY
    // SECOND: testRZ
    %qout = quantum.custom "testPauliX"() %q : !quantum.bit
    return %qout : !quantum.bit
}

