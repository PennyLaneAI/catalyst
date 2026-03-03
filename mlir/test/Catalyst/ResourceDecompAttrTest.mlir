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

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-tracker{decomp-attr=true})" --split-input-file %s | FileCheck %s

// Basic decomposition rule

// CHECK: @basic_gates() attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, operations = {"CNOT(2)" = 1 : i64, "Hadamard(1)" = 1 : i64, "S(1)" = 1 : i64, "T(1)" = 1 : i64}}, target_gate = "basic"}
func.func @basic_gates() attributes {target_gate="basic"}  {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %4 = quantum.custom "T"() %3 : !quantum.bit
    %5 = quantum.custom "S"() %2 : !quantum.bit
    %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
    %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with PBC ops

// CHECK: func.func @pbc_operations() attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, operations = {"PPM(1)" = 2 : i64, "PPR-pi/4(1)" = 3 : i64, "PPR-pi/8(1)" = 1 : i64}}, target_gate = "pbc"}
func.func @pbc_operations() attributes {target_gate="pbc"} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %4 = pbc.ppr ["X"](4) %3 : !quantum.bit
    %5 = pbc.ppr ["Z"](4) %2 : !quantum.bit
    %6 = pbc.ppr ["Z"](8) %4 : !quantum.bit
    %mres, %out = pbc.ppm ["Z"] %5 : i1, !quantum.bit
    %mres2, %out2 = pbc.ppm ["X"] %6 : i1, !quantum.bit
    %7 = quantum.insert %0[ 0], %out2 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with measure

// CHECK: @rule_mcm() attributes {resources = {measurements = {MidCircuitMeasure = 1 : i64}, num_alloc_qubits = 1 : i64, operations = {"Hadamard(1)" = 1 : i64}}, target_gate = "gate"}
func.func @rule_mcm() attributes {target_gate="gate"} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %mres, %out = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}

// -----

// Rule depends on helper functions

// CHECK: @helper_func(%arg0: !quantum.bit) -> !quantum.bit
func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

// CHECK: rule(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, resources =
func.func @rule(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, target_gate="rule"} {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}
