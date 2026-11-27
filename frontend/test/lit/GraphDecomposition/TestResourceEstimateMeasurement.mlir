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

// Test that resource-estimate can target MeasurementOp names in the gateset

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=testMCM=1.0})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "circuit"
// CHECK: "function_calls"
// CHECK:   "static"
// CHECK:       "test_op_to_test_mcm(testOp{}{wires:2}{})": 1
// CHECK: "quantum_operations"
// CHECK:       "testMCM": 1
// CHECK-NOT:   "testOp"

// CHECK-LABEL: "test_op_to_test_mcm(testOp{}{wires:2}{})": {
// CHECK: "quantum_operations"
// CHECK:   "2"
// CHECK:       "testMCM": 2
func.func public @circuit() attributes {quantum.node} {
    %q0 = quantum.alloc_qb : !quantum.bit
    %q1 = quantum.alloc_qb : !quantum.bit
    %out:2 = quantum.custom "testOp"() %q0, %q1 : !quantum.bit, !quantum.bit
    %m, %out_m:2 = quantum.measurement "testMCM"() qubits(%out#0, %out#1) : (i1, !quantum.bit, !quantum.bit)
    quantum.dealloc_qb %q0 : !quantum.bit
    quantum.dealloc_qb %q1 : !quantum.bit
    return
}

// resource-estimate only reads the target gate and resources of a rule.
func.func private @test_op_to_test_mcm(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "testOp{}{wires:2}{}",
    resources = {operations = {"testMCM{}{wires:2}{}" = 2 : i64}}
} {
    return %q0, %q1 : !quantum.bit, !quantum.bit
}
