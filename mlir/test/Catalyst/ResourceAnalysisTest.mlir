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

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-analysis{output-json=true})" --split-input-file %s | FileCheck %s


// Basic gate counting

// CHECK-LABEL: "basic_gates"
// CHECK:   "num_alloc_qubits": 2
// CHECK:   "num_arg_qubits": 0
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
// CHECK:   "num_alloc_qubits": 2
// CHECK:   "num_arg_qubits": 0
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

// Static for loop: the body is lifted into a synthetic "for_loop_1"
// entry with one-iteration counts; the parent records a function_calls
// entry with the trip count.
//
// JSON entries are emitted in alphabetical order.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1

// CHECK-LABEL: "static_for_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 5
// CHECK: "operations": {}
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

// Input-driven dynamic for loop: the upper bound is a func.func block
// argument, so the body is lifted into "dyn_for_loop_1" and the parent
// records var_function_calls + has_dyn_loop=true.

// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "dynamic_for_loop": {
// CHECK: "has_dyn_loop": true
// CHECK: "operations": {}
// CHECK: "var_function_calls"
// CHECK: "dyn_for_loop_1"
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

// Dynamic for loop with estimated_iterations: treated as static for the
// purposes of counting, so it lifts into "for_loop_1" with the
// attribute's iteration count as the call multiplier.

// CHECK-LABEL: "estimated_iterations_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 10
// CHECK: "operations": {}

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliZ(1)": 1
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

// For loop with indirect bounds resolvable through index_cast + addi.
// Trip count = 7 -> static lift.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1

// CHECK-LABEL: "resolve_constant_index_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 7
func.func @resolve_constant_index_loop(%arg0: !quantum.bit) -> !quantum.bit {
    %c0_i64 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %ub_i64 = arith.addi %c3_i64, %c4_i64 : i64
    %c0 = arith.index_cast %c0_i64 : i64 to index
    %ub = arith.index_cast %ub_i64 : i64 to index
    %c1 = arith.index_cast %c1_i64 : i64 to index

    %q = scf.for %iter = %c0 to %ub step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "Hadamard"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// CHECK-LABEL: "for_loop_1": {
// CHECK: "function_calls"
// CHECK: "gate_inside_loop_helper": 1
// CHECK: "operations": {}

// CHECK-LABEL: "gate_inside_loop_helper": {
// CHECK: "operations"
// CHECK-DAG: "PauliZ(1)": 1

// CHECK-LABEL: "static_for_with_call_in_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 4
// CHECK: "operations": {}
func.func private @gate_inside_loop_helper(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "PauliZ"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

func.func @static_for_with_call_in_loop(%arg0: !quantum.bit) -> !quantum.bit {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c4 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = func.call @gate_inside_loop_helper(%arg1) : (!quantum.bit) -> !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliZ(1)": 1

// CHECK-LABEL: "dyn_for_loop_2": {
// CHECK: "function_calls"
// CHECK: "nested_inner_loop_helper": 1
// CHECK: "operations": {}

// CHECK-LABEL: "nested_inner_loop_helper": {
// CHECK: "has_dyn_loop": true
// CHECK: "operations": {}
// CHECK: "var_function_calls"
// CHECK: "dyn_for_loop_1": "{{0x[0-9a-f]+}}"

// CHECK-LABEL: "outer_dyn_loop_calls_nested_helper": {
// CHECK: "has_dyn_loop": true
// CHECK: "operations": {}
// CHECK: "var_function_calls"
// CHECK: "dyn_for_loop_2": "{{0x[0-9a-f]+}}"
func.func private @nested_inner_loop_helper(%arg0: !quantum.bit, %c4 : index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c4 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "PauliZ"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }
    return %q : !quantum.bit
}

func.func @outer_dyn_loop_calls_nested_helper(%arg0: !quantum.bit, %c4 : index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c4 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = func.call @nested_inner_loop_helper(%arg1, %c4) : (!quantum.bit, index) -> !quantum.bit
        scf.yield %out : !quantum.bit
    }

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

// Nested static for loops: the inner loop is lifted into for_loop_1
// with the gate; the outer loop is lifted into for_loop_2 and records
// function_calls={ for_loop_1: 5 }; the parent records
// function_calls={ for_loop_2: 3 }.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "for_loop_2": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 5
// CHECK: "operations": {}

// CHECK-LABEL: "nested_static_for": {
// CHECK: "function_calls"
// CHECK: "for_loop_2": 3
// CHECK: "operations": {}
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

// quantum.adjoint over a static for-loop: the adjoint flag must be
// threaded into the lifted body, so the gate name picks up the
// "Adjoint(...)" prefix in the synthetic for_loop_<N> entry.

// CHECK-LABEL: "adjoint_static_for_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 5

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "Adjoint(Hadamard)(1)": 1
func.func @adjoint_static_for_loop(%arg0: !quantum.reg) -> !quantum.reg {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64

    %r = quantum.adjoint(%arg0) : !quantum.reg {
    ^bb0(%a0: !quantum.reg):
        %r2 = scf.for %i = %c0 to %c5 step %c1 iter_args(%a = %a0) -> (!quantum.reg) {
            %q = quantum.extract %a[%c0_i64] : !quantum.reg -> !quantum.bit
            %h = quantum.custom "Hadamard"() %q : !quantum.bit
            %a1 = quantum.insert %a[%c0_i64], %h : !quantum.reg, !quantum.bit
            scf.yield %a1 : !quantum.reg
        }
        quantum.yield %r2 : !quantum.reg
    }
    return %r : !quantum.reg
}

// -----

// Synthetic name collision: a user function literally named
// `for_loop_1` must not be overwritten by the lifted body. The lifted
// entry should bump the counter and pick a distinct name.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "for_loop_2": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1

// CHECK-LABEL: "user_caller": {
// CHECK: "function_calls"
// CHECK-DAG: "for_loop_2": 5
// CHECK: "operations": {}
func.func @user_caller(%arg0: !quantum.bit) -> !quantum.bit {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c5 step %c1 iter_args(%a = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "Hadamard"() %a : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// User function whose name happens to match the synthetic naming
// scheme. The pass should NOT clobber it.
func.func private @for_loop_1(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "PauliX"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

// -----

// Measurement and qubit allocation inside a lifted for-loop body
// stay local to the synthetic entry instead of being inlined into
// the parent.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "measurements"
// CHECK-DAG: "MidCircuitMeasure": 1
// CHECK: "num_alloc_qubits": 1
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1

// CHECK-LABEL: "loop_with_measurement_and_alloc": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 3
// CHECK: "measurements": {}
// CHECK: "num_alloc_qubits": 0
// CHECK: "operations": {}
func.func @loop_with_measurement_and_alloc() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index

    scf.for %i = %c0 to %c3 step %c1 {
        %qb = quantum.alloc_qb : !quantum.bit
        %h = quantum.custom "Hadamard"() %qb : !quantum.bit
        %mres, %out = quantum.measure %h : i1, !quantum.bit
        quantum.dealloc_qb %out : !quantum.bit
    }

    return
}

// -----

// Measurements at the top level (not inside a loop)

// CHECK-LABEL: "measurement_ops"
// CHECK: "measurements"
// CHECK-DAG: "MidCircuitMeasure": 1
// CHECK: "num_alloc_qubits": 1
// CHECK: "num_arg_qubits": 0
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

// Observable-based measurements

// CHECK-LABEL: "observable_measurements"
// CHECK: "measurements"
// CHECK-DAG: "expval(Hamiltonian(num_terms=2))": 1
// CHECK-DAG: "expval(Prod(num_terms=2))": 1
// CHECK-DAG: "sample(1 wires)": 1
// CHECK-DAG: "sample(all wires)": 1
// CHECK-DAG: "probs(all wires)": 1
// CHECK-DAG: "expval(PauliZ)": 1
// CHECK-DAG: "var(PauliX)": 1
func.func @observable_measurements() {
    %0 = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // expval(Hamiltonian(num_terms=2))
    %obs_z0 = quantum.namedobs %q0[ PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[ PauliX] : !quantum.obs
    %coeffs = arith.constant dense<[0.2, -0.543]> : tensor<2xf64>
    %ham = quantum.hamiltonian(%coeffs : tensor<2xf64>) %obs_z0, %obs_x1 : !quantum.obs
    %ev1 = quantum.expval %ham : f64

    // expval(Prod(num_terms=2))
    %obs_z0b = quantum.namedobs %q0[ PauliZ] : !quantum.obs
    %obs_z1 = quantum.namedobs %q1[ PauliZ] : !quantum.obs
    %tensor_obs = quantum.tensor %obs_z0b, %obs_z1 : !quantum.obs
    %ev2 = quantum.expval %tensor_obs : f64

    // sample(1 wires) -- computational basis with 1 qubit
    %cb1 = quantum.compbasis qubits %q2 : !quantum.obs
    %s1 = quantum.sample %cb1 : tensor<10x1xf64>

    // sample(all wires) -- computational basis with no explicit qubits
    %cb_all = quantum.compbasis : !quantum.obs
    %s2 = quantum.sample %cb_all : tensor<10x3xf64>

    // probs(all wires)
    %cb_all2 = quantum.compbasis : !quantum.obs
    %p = quantum.probs %cb_all2 : tensor<3xf64>

    // expval(PauliZ)
    %obs_z2 = quantum.namedobs %q2[ PauliZ] : !quantum.obs
    %ev3 = quantum.expval %obs_z2 : f64

    // var(PauliX)
    %obs_x0 = quantum.namedobs %q0[ PauliX] : !quantum.obs
    %v1 = quantum.var %obs_x0 : f64

    quantum.dealloc %0 : !quantum.reg
    return
}

// -----

// Function-call resolution: the callee's gates STAY in the callee, the
// caller exposes only its direct function_calls.

// CHECK-LABEL: "caller_func": {
// CHECK: "function_calls"
// CHECK: "helper_func": 2
// CHECK: "operations": {}

// CHECK-LABEL: "helper_func": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1
func.func @caller_func(%arg0: !quantum.bit) -> !quantum.bit {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}


// -----

// Function call resolution with nested calls: gates stay in the deepest
// (lifted) entry; structural function_calls show direct call counts.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "for_loop_2": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 5
// CHECK: "operations": {}

// CHECK-LABEL: "helper_func": {
// CHECK: "function_calls"
// CHECK: "for_loop_2": 3
// CHECK: "operations": {}

// CHECK-LABEL: "nested_caller_func": {
// CHECK: "function_calls"
// CHECK: "nested_helper_func": 2
// CHECK: "operations": {}

// CHECK-LABEL: "nested_helper_func": {
// CHECK: "function_calls"
// CHECK: "helper_func": 2
// CHECK: "operations": {}

func.func @nested_caller_func(%arg0: !quantum.bit) -> !quantum.bit {
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
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

// -----

// Mixed quantum and PBC ops

// CHECK-LABEL: "mixed_ops"
// CHECK: "num_alloc_qubits": 2
// CHECK: "num_arg_qubits": 0
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

// Qubit arguments on the entry function are counted toward num_qubits.

// CHECK-LABEL: "multi_qubit_args"
// CHECK: "num_alloc_qubits": 0
// CHECK: "num_arg_qubits": 2
// CHECK: "num_qubits": 2
// CHECK: "operations"
// CHECK-DAG: "CNOT(2)": 1
func.func @multi_qubit_args(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %0:2 = quantum.custom "CNOT"() %q0, %q1 : !quantum.bit, !quantum.bit
    return %0#0, %0#1 : !quantum.bit, !quantum.bit
}

// -----

// Mixed: both allocated qubits and argument qubits contribute to num_qubits.

// CHECK-LABEL: "mixed_alloc_and_arg_qubits"
// CHECK: "num_alloc_qubits": 2
// CHECK: "num_arg_qubits": 1
// CHECK: "num_qubits": 3
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1
// CHECK-DAG: "CNOT(2)": 1
func.func @mixed_alloc_and_arg_qubits(%q0: !quantum.bit) -> !quantum.bit {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %4:2 = quantum.custom "CNOT"() %q0, %3 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %0[ 0], %4#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    return %4#0 : !quantum.bit
}

// -----

// Pass statistics output

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-analysis)" -mlir-pass-statistics -mlir-pass-statistics-display=list --split-input-file %s 2>&1 | FileCheck %s --check-prefix=STATS

// STATS: ResourceAnalysisPass
// STATS: 2 total-alloc-qubits
// STATS: 0 total-arg-qubits
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
// Stats walk the structural call graph: 4 gates total =
//   first_qnode (0) +
//     2 * shared_helper (1 H) +
//     1 * second_qnode (1 PauliX + 1 * shared_helper (1 H)) = 4.

// STATS: ResourceAnalysisPass
// STATS: 0 total-alloc-qubits
// STATS: 1 total-arg-qubits
// STATS: 4 total-function-calls
// STATS: 4 total-gates
// STATS: 1 total-qubits
func.func @first_qnode(%arg0: !quantum.bit) -> !quantum.bit {
    %r1 = func.call @shared_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @shared_helper(%r1) : (!quantum.bit) -> !quantum.bit
    %r3 = func.call @second_qnode(%r2) : (!quantum.bit) -> !quantum.bit
    return %r3 : !quantum.bit
}

func.func private @shared_helper(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}


func.func @second_qnode(%arg0: !quantum.bit) -> !quantum.bit {
    %r1 = func.call @shared_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    %out = quantum.custom "PauliX"() %r1 : !quantum.bit
    return %out : !quantum.bit
}

// -----

// qnode attribute marking: the qnode keeps `qnode: true`, helper keeps
// its own gate (no inlining into the qnode).

// CHECK-LABEL: "my_helper": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1
// CHECK: "qnode": false

// CHECK-LABEL: "my_qnode": {
// CHECK: "function_calls"
// CHECK: "my_helper": 1
// CHECK: "operations": {}
// CHECK: "qnode": true
func.func @my_qnode(%arg0: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %r = func.call @my_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    return %r : !quantum.bit
}

func.func private @my_helper(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}
