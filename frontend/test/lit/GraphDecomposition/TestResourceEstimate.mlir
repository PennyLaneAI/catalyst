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

// Test that resource-estimate counts each op by the basis gates of its chosen decomposition

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=testRX=1.0,testRZ=1.0 bytecode-rules="%BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s --implicit-check-not=testHadamard

// testHadamard decomposes into testRX + 2 testRZ (h_to_rz_rx), and Adjoint(testHadamard) into
// 3 testRZ (@adj_h). The adjoint region only matches the latter once it is lowered.

// Each function is estimated on its own, so calls and loop bodies stay in function_calls.
// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK:       "quantum_operations": {
// CHECK-NEXT:    "-1": {
// CHECK-NEXT:      "testRX": 1,
// CHECK-NEXT:      "testRZ": 2

// The entry's own ops are 1 testHadamard, 1 testRX, 1 adjoint region, and 1 mid-circuit
// measurement, which is not a graph operator and is counted as is.
// CHECK-LABEL: "estimate_circuit": {
// CHECK:       "function_calls": {
// CHECK-NEXT:    "dynamic": {
// CHECK-NEXT:      "dyn_for_loop_1": "{{0x[0-9a-f]+}}"
// CHECK-NEXT:    },
// CHECK-NEXT:    "static": {
// CHECK-NEXT:      "helper": 2
// CHECK:       "num_qubits"
// CHECK-NEXT:    "alloc": 1,
// CHECK-NEXT:    "arg": 0,
// CHECK-NEXT:    "total": 1
// CHECK:       "quantum_operations": {
// CHECK-NEXT:    "-1": {
// CHECK-NEXT:      "MidCircuitMeasure": 1,
// CHECK-NEXT:      "testRX": 2,
// CHECK-NEXT:      "testRZ": 5
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: "helper": {
// CHECK:       "quantum_operations": {
// CHECK-NEXT:    "-1": {
// CHECK-NEXT:      "testRX": 1,
// CHECK-NEXT:      "testRZ": 2
func.func public @estimate_circuit(%n: index) attributes {quantum.node} {
    %angle = arith.constant 0.5 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "testHadamard"() %1 : !quantum.bit
    %3 = quantum.custom "testRX"(%angle) %2 : !quantum.bit
    %4 = quantum.adjoint(%3) : !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %a = quantum.custom "testHadamard"() %arg0 : !quantum.bit
        quantum.yield %a : !quantum.bit
    }
    %5 = func.call @helper(%4) : (!quantum.bit) -> !quantum.bit
    %6 = func.call @helper(%5) : (!quantum.bit) -> !quantum.bit
    %7 = scf.for %i = %c0 to %n step %c1 iter_args(%q = %6) -> (!quantum.bit) {
        %h = quantum.custom "testHadamard"() %q : !quantum.bit
        scf.yield %h : !quantum.bit
    }
    %mres, %8 = quantum.measure %7 : i1, !quantum.bit
    %9 = quantum.insert %0[ 0], %8 : !quantum.reg, !quantum.bit
    quantum.dealloc %9 : !quantum.reg
    return
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "testHadamard"() %q : !quantum.bit
    return %out : !quantum.bit
}

func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Adjoint(testHadamard){}{wires:1}{}",
    resources = {operations = {"testRZ{0:[f64]}{wires:1}{}" = 3 : i64}} } {
    %angle = arith.constant 0.5 : f64
    %a = quantum.custom "testRZ"(%angle) %q : !quantum.bit
    %b = quantum.custom "testRZ"(%angle) %a : !quantum.bit
    %c = quantum.custom "testRZ"(%angle) %b : !quantum.bit
    return %c : !quantum.bit
}
