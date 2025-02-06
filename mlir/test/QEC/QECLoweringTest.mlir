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

// RUN: quantum-opt --convert-to-qec --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_qec_lowering(%q1 : !quantum.bit, %q2 : !quantum.bit){
    %theta = arith.constant 0.1 : f64
    // CHECK: qec.ppr ["Z", "X", "Z"]
    // CHECK: qec.ppr ["Z"]
    // CHECK: qec.ppr ["Z"]
    // CHECK: qec.ppr ["Z", "X"]
    %q1_0 = quantum.custom "H"() %q1 : !quantum.bit
    %q1_1 = quantum.custom "S"() %q1_0 : !quantum.bit
    %q1_2 = quantum.custom "T"() %q1_1 : !quantum.bit
    %q1_3:2 = quantum.custom "CNOT"() %q1_2, %q2 : !quantum.bit, !quantum.bit
    func.return
}
