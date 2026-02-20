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

// Test conversion to value semantics quantum dialect.
//
// RUN: quantum-opt --convert-to-value-semantics --split-input-file --verify-diagnostics %s

func.func @test_expval_circuit(%arg0: f64) -> f64 {
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit
    // qref.custom "Hadamard"() %q0 : !qref.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
    qref.custom "RX"(%arg0) %q0 : !qref.bit
    // %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    // %expval = quantum.expval %obs : f64
    qref.dealloc %a : !qref.reg<2>
    //return %expval : f64
    return %arg0 : f64
}
