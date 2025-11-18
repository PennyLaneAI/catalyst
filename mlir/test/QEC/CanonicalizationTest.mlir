// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --split-input-file -verify-diagnostics --canonicalize %s | FileCheck %s

func.func @test_ppr_canonicalize_single_identity(%q1 : !quantum.bit, %q2 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %0 = qec.ppr ["I"](4) %q1 : !quantum.bit
    %1 = qec.ppr ["I"](4) %q2 : !quantum.bit
    return %0, %1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: qec.ppr ["I"](4)
}

// -----

func.func @test_ppr_canonicalize_multiple_identity(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %out_qubits:3 = qec.ppr ["I", "I", "I"](4) %q1, %q2, %q3: !quantum.bit, !quantum.bit, !quantum.bit
    return %out_qubits#0, %out_qubits#1, %out_qubits#2 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK-NOT: qec.ppr ["I", "I", "I"](4)
}
