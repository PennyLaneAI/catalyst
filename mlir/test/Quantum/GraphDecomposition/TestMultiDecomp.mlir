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

// RUN: quantum-opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes FIRST

// RUN: quantum-opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"},graph-decomposition{gate-set=RZ=1.0,RY=1.0,gphase=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes SECOND

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // FIRST-NOT: PauliX
    // FIRST: RX

    // SECOND-NOT: PauliX
    // SECOND-NOT: RX
    // SECOND: RZ
    // SECOND: RY
    // SECOND: RZ
    %qout = quantum.custom "PauliX"() %q : !quantum.bit
    return %qout : !quantum.bit
}

