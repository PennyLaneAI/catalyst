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

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-tracker{output-json=true})" --split-input-file %s | FileCheck %s


// Basic gate counting

// CHECK-LABEL: "basic_gates"
// CHECK:   "num_qubits": 2
// CHECK:   "operations"
// CHECK-DAG: "Hadamard(1)": 1
// CHECK-DAG: "CNOT(2)": 1
// CHECK-DAG: "T(1)": 1
// CHECK-DAG: "S(1)": 1
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

// PBC operations (PPR and PPM)

// CHECK-LABEL: "pbc_operations"
// CHECK:   "num_qubits": 2
// CHECK:   "operations"
// CHECK-DAG: "PPR-pi/4(1)": 3
// CHECK-DAG: "PPR-pi/8(1)": 1
// CHECK-DAG: "PPM(1)": 2
func.func @pbc_operations() {
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

// Static for loop

// CHECK-LABEL: "static_for_loop"
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 5
func.func @static_for_loop(%arg0: !quantum.bit) -> !quantum.bit {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c5 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "Hadamard"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Dynamic for loop (counted as 1 iteration)

// CHECK-LABEL: "dynamic_for_loop"
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1
func.func @dynamic_for_loop(%arg0: !quantum.bit, %n: index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %n step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "PauliX"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Dynamic for loop with estimated_iterations

// CHECK-LABEL: "estimated_iterations_loop"
// CHECK: "operations"
// CHECK-DAG: "PauliZ(1)": 10
func.func @estimated_iterations_loop(%arg0: !quantum.bit, %n: index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %n step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "PauliZ"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    } {estimated_iterations = 10 : i16}

    return %q : !quantum.bit
}

// -----

// If-else branching (take max per op)

// CHECK-LABEL: "if_else_branching"
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 3
// CHECK-DAG: "PauliX(1)": 2
func.func @if_else_branching(%arg0: !quantum.bit, %cond: i1) -> !quantum.bit {
    %q = scf.if %cond -> !quantum.bit {
        // True branch: 2 Hadamard, 1 PauliX
        %t1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %t2 = quantum.custom "Hadamard"() %t1 : !quantum.bit
        %t3 = quantum.custom "PauliX"() %t2 : !quantum.bit
        scf.yield %t3 : !quantum.bit
    } else {
        // False branch: 3 Hadamard, 2 PauliX -> max(2,3)=3 Hadamard, max(1,2)=2 PauliX
        %f1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %f2 = quantum.custom "Hadamard"() %f1 : !quantum.bit
        %f3 = quantum.custom "Hadamard"() %f2 : !quantum.bit
        %f4 = quantum.custom "PauliX"() %f3 : !quantum.bit
        %f5 = quantum.custom "PauliX"() %f4 : !quantum.bit
        scf.yield %f5 : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Nested static for loops

// CHECK-LABEL: "nested_static_for"
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 15
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

// Measurements

// CHECK-LABEL: "measurement_ops"
// CHECK: "measurements"
// CHECK-DAG: "MidCircuitMeasure": 1
// CHECK: "num_qubits": 1
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

// Function call resolution

// CHECK-LABEL: "caller_func"
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 2
func.func @caller_func(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode} {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}


// -----

// Function call resolution with nested calls

// CHECK-LABEL: "nested_caller_func"
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 60

func.func @nested_caller_func(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode} {
    %r1 = func.call @nested_helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @nested_helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // for (i = 0; i < 3; i++) {
    %q = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %arg0) -> (!quantum.bit) {
        // for (j = 0; j < 5; j++) {
        %q2 = scf.for %j = %c0 to %c5 step %c1 iter_args(%b = %a) -> (!quantum.bit) {
            %out = quantum.custom "PauliX"() %b : !quantum.bit
            scf.yield %out : !quantum.bit
        }
        scf.yield %q2 : !quantum.bit
    }
    return %q : !quantum.bit
}

func.func private @nested_helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    // total: 15 PauliX operations
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    // total: 15 PauliX operations
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

// -----

// Mixed quantum and PBC ops

// CHECK-LABEL: "mixed_ops"
// CHECK: "num_qubits": 2
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1
// CHECK-DAG: "PPR-pi/4(1)": 1
// CHECK-DAG: "PPM(1)": 1
func.func @mixed_ops() {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %4 = pbc.ppr ["Z"](4) %2 : !quantum.bit
    %mres, %out = pbc.ppm ["Z"] %4 : i1, !quantum.bit
    %5 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    return
}

// -----

// Pass statistics output

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-tracker)" -mlir-pass-statistics -mlir-pass-statistics-display=list --split-input-file %s 2>&1 | FileCheck %s --check-prefix=STATS

// STATS: ResourceTrackerPass
// STATS: 1 total-classical-ops
// STATS: 1 total-gates
// STATS: 1 total-measurements
// STATS: 2 total-qubits
func.func @stats_test() {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %mres, %out = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}

// -----

// Multiple qnode functions: the first qnode is the entry function.

// STATS: ResourceTrackerPass
// STATS: 4 total-function-calls
// STATS: 4 total-gates
func.func @first_qnode(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode} {
    %r1 = func.call @shared_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @shared_helper(%r1) : (!quantum.bit) -> !quantum.bit
    %r3 = func.call @second_qnode(%r2) : (!quantum.bit) -> !quantum.bit
    return %r3 : !quantum.bit
}

func.func private @shared_helper(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}


func.func @second_qnode(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode} {
    %r1 = func.call @shared_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    %out = quantum.custom "PauliX"() %r1 : !quantum.bit
    return %out : !quantum.bit
}
