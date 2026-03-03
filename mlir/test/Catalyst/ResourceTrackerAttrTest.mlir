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

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-tracker{update-attr=true})" --split-input-file %s | FileCheck %s

// Test gates, measurement and num_qubits

// CHECK: basic_gates() attributes {resources = {classical_instructions = {func.return = 1 : i64}, device_name = "", function_calls = {}, measurements = {}, num_alloc_qubits = 2 : i64, num_arg_qubits = 0 : i64, num_qubits = 2 : i64, operations = {"CNOT(2)" = 1 : i64, "Hadamard(1)" = 1 : i64, "S(1)" = 1 : i64, "T(1)" = 1 : i64}}}
func.func @basic_gates() {
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

// Test classical operations

// CHECK: @nested_static_for(%arg0: !quantum.bit) -> !quantum.bit attributes {resources = {classical_instructions = {arith.constant = 4 : i64, func.return = 1 : i64, scf.for = 4 : i64, scf.yield = 18 : i64},
func.func @nested_static_for(%arg0: !quantum.bit) -> !quantum.bit {
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %arg0) -> (!quantum.bit) {
        %q2 = scf.for %j = %c0 to %c5 step %c1 iter_args(%b = %a) -> (!quantum.bit) {
            %out = quantum.custom "PauliX"() %b : !quantum.bit
            scf.yield %out : !quantum.bit
        }
        scf.yield %q2 : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Test measure

// CHECK: @measurement_ops() attributes {resources = {classical_instructions = {func.return = 1 : i64}, device_name = "", function_calls = {}, measurements = {MidCircuitMeasure = 1 : i64},
func.func @measurement_ops() {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %mres, %out = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}

// -----

// Test function call

// CHECK: @helper_func(%arg0: !quantum.bit) -> !quantum.bit attributes {resources =
func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

// CHECK: func.func @caller_func(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, resources = {classical_instructions = {func.return = 1 : i64}, device_name = "", function_calls = {helper_func = 2 : i64},
func.func @caller_func(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode} {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}
