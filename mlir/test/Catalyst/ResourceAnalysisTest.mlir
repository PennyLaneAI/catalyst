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
// RUN: quantum-opt --pass-pipeline="builtin.module(resource-analysis)" --split-input-file %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN


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

// Operator2 gates

// CHECK-LABEL: "operator2_gates"
// CHECK:    "num_alloc_qubits": 10
// CHECK:    "num_arg_qubits": 0
// CHECK:    "num_qubits": 10
// CHECK:    "operations"
// CHECK-DAG: "DummyOp(4)": 1
// CHECK-DAG: "DummyOp(5)": 1

func.func @operator2_gates(){
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %0 = quantum.alloc( 10) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %out_qubits:5 = quantum.operator "DummyOp"(%cst: tensor<f64>) qubits(%1, %2, %3, %4, %5)
      static_data = {metadata = "word"}
      param_map = {phi = [0]} qubit_map = {reg1 = [0, 1], reg2 = [2, 3, 4]}
    %6 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 2], %out_qubits#2 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[ 3], %out_qubits#3 : !quantum.reg, !quantum.bit
    %10 = quantum.insert %9[ 4], %out_qubits#4 : !quantum.reg, !quantum.bit
    %11 = quantum.extract %10[ 2] : !quantum.reg -> !quantum.bit
    %12 = quantum.extract %10[ 3] : !quantum.reg -> !quantum.bit
    %13 = quantum.extract %10[ 4] : !quantum.reg -> !quantum.bit
    %14 = quantum.extract %10[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits_0:4 = quantum.operator "DummyOp"(%cst: tensor<f64>) qubits(%11, %12, %13, %14)
      static_data = {metadata = "word"}
      param_map = {phi = [0]} qubit_map = {reg1 = [0, 1, 2], reg2 = [3]}
    %15 = quantum.insert %10[ 2], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %16 = quantum.insert %15[ 3], %out_qubits_0#1 : !quantum.reg, !quantum.bit
    %17 = quantum.insert %16[ 4], %out_qubits_0#2 : !quantum.reg, !quantum.bit
    %18 = quantum.insert %17[ 0], %out_qubits_0#3 : !quantum.reg, !quantum.bit
    quantum.dealloc %18 : !quantum.reg
    return
}

// -----

// MBQC Operations

// CHECK-LABEL: "mbqc_gates"
// CHECK: "num_alloc_qubits": 4
// CHECK: "operations"
// CHECK-DAG: "mbqc.graph_state_prep(0)": 1,
// CHECK-DAG: "mbqc.measure_in_basis(0)": 1
func.func @mbqc_gates() {
    %adj_matrix = arith.constant dense<[1, 0, 1, 0, 0, 1]> : tensor<6xi1>
    %graph_reg = mbqc.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg

    %q0 = quantum.extract %graph_reg[0] : !quantum.reg -> !quantum.bit

    %angle = arith.constant 4.0 : f64
    %mbqc_meas, %out = mbqc.measure_in_basis [ZX, %angle] %q0 : i1, !quantum.bit

    return
}

// -----

// PBC operations (PPR and PPM)

// CHECK-LABEL: "pbc_operations"
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 4
// CHECK-DAG: "qubit_disjoint_depth": 4
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
// records var_function_calls (hasDynLoop is tracked internally, not emitted).

// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "dynamic_for_loop": {
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

// PBC ops inside a dynamic for loop: parent reports empty depth; lifted body
// reports single-iteration depth.

// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 1
// CHECK-DAG: "qubit_disjoint_depth": 1
// CHECK-DAG: "PPR-pi/4(1)": 1

// CHECK-LABEL: "pbc_in_dyn_loop"
// CHECK: "depth": {}
func.func @pbc_in_dyn_loop(%arg0: !quantum.bit, %n: index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %n step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = pbc.ppr ["Z"](4) %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Inline PBC ops after a dynamic for loop: parent depth covers only the
// post-loop ops; loop body depth stays in dyn_for_loop_<N>.

// CHECK-LABEL: "dyn_for_loop_1": {
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 1
// CHECK-DAG: "qubit_disjoint_depth": 1
// CHECK-DAG: "PPR-pi/4(1)": 1

// CHECK-LABEL: "pbc_in_dyn_loop_post_ppr": {
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 1
// CHECK-DAG: "qubit_disjoint_depth": 1
// CHECK-DAG: "PPR-pi/4(1)": 1
// CHECK: "var_function_calls"
// CHECK: "dyn_for_loop_1"
func.func @pbc_in_dyn_loop_post_ppr(%arg0: !quantum.bit, %n: index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %n step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = pbc.ppr ["Z"](4) %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    %q3 = pbc.ppr ["X"](4) %q : !quantum.bit

    return %q3 : !quantum.bit
}

// -----

// CHECK: "test_independent_extract_between_pprs": 
// CHECK:   "depth":
// CHECK:      "any_commuting_depth": 1
// CHECK:      "qubit_disjoint_depth": 2
func.func public @test_independent_extract_between_pprs() {
    %0 = quantum.alloc(4) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[2] : !quantum.reg -> !quantum.bit
    %4:3 = pbc.ppr ["X", "X", "X"](4) %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %5 = quantum.extract %0[3] : !quantum.reg -> !quantum.bit
    %6:3 = pbc.ppr ["X", "X", "Y"](4) %4#1, %4#2, %5 : !quantum.bit, !quantum.bit, !quantum.bit
    %7 = quantum.insert %0[0], %4#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[1], %6#0 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[2], %6#1 : !quantum.reg, !quantum.bit
    %10 = quantum.insert %9[3], %6#2 : !quantum.reg, !quantum.bit
    quantum.dealloc %10 : !quantum.reg
    return
}

// -----

// CHECK: "for_loop_1"
// CHECK:   "depth":
// CHECK:     "any_commuting_depth": 1
// CHECK:     "qubit_disjoint_depth": 2
// CHECK:   "operations":
// CHECK:     "PPR-pi/8(1)": 2

// CHECK: "for_loop_2":
// CHECK:   "depth":
// CHECK:     "any_commuting_depth": 1
// CHECK:     "qubit_disjoint_depth": 1
// CHECK:   "function_calls":
// CHECK:     "for_loop_1": 5
// CHECK:   "operations":
// CHECK:     "PPR-pi/8(1)": 1

// CHECK: "static_for_loop_nested":
// CHECK:   "function_calls":
// CHECK:     "for_loop_2": 6

func.func public @static_for_loop_nested(%arg0: !quantum.bit) {
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // for_loop_2
    %q = scf.for %iter = %c0 to %c6 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {

        // for_loop_1
        %qp_inner = scf.for %iter_inner = %c0 to %c5 step %c1 iter_args(%arg1_qp = %arg1) -> (!quantum.bit) {
          %qp0 = pbc.ppr ["Z"](8) %arg1_qp : !quantum.bit
          %qp1 = pbc.ppr ["Z"](8) %qp0 : !quantum.bit
          scf.yield %qp1 : !quantum.bit
        }

        %qp2 = pbc.ppr ["Z"](8) %qp_inner : !quantum.bit
        scf.yield %qp2 : !quantum.bit
    }

    return
}

// -----


// CHECK: "for_loop_1":
// CHECK:   "depth":
// CHECK:     "any_commuting_depth": 1
// CHECK:     "qubit_disjoint_depth": 1
// CHECK:   "operations": {
// CHECK:     "PPR-pi/4(1)": 1

// CHECK: "pbc_estimated_iterations_loop":
// CHECK:  "function_calls":
// CHECK:    "for_loop_1": 10

func.func @pbc_estimated_iterations_loop(%arg0: !quantum.bit, %n: index) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %n step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = pbc.ppr ["Z"](4) %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    } {catalyst.estimated_iterations = 10 : i16}

    return %q : !quantum.bit
}

// -----

// PBC depth is per-function only; callers without inline PBC ops report no depth
// even when they call a helper that contains PBC ops.

// CHECK-LABEL: "depth_caller": {
// CHECK: "depth": {}
// CHECK: "function_calls"
// CHECK: "depth_helper": 1

// CHECK-LABEL: "depth_helper": {
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 1
// CHECK-DAG: "qubit_disjoint_depth": 1
func.func private @depth_helper(%arg0: !quantum.bit) -> !quantum.bit {
    %out = pbc.ppr ["Z"](4) %arg0 : !quantum.bit
    return %out : !quantum.bit
}

func.func @depth_caller(%arg0: !quantum.bit) -> !quantum.bit {
    %out = func.call @depth_helper(%arg0) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

// -----

// Dynamic for loop with catalyst.estimated_iterations: treated as static for the
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
    } {catalyst.estimated_iterations = 10 : i16}

    return %q : !quantum.bit
}

// -----

// Reverse for-loop bounds lowered via subi + ceildivsi (PennyLane for_loop
// start/stop/step pattern). Trip count = ceildivsi(subi(-1, 9), -1) = 10.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 1

// CHECK-LABEL: "reverse_for_loop_ceildivsi": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 10
// CHECK: "operations": {}
func.func @reverse_for_loop_ceildivsi(%arg0: !quantum.reg) -> !quantum.reg {
    %c9_i64 = arith.constant 9 : i64
    %cm1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %start = arith.index_cast %c9_i64 : i64 to index
    %stop = arith.index_cast %cm1_i64 : i64 to index
    %neg_step = arith.index_cast %cm1_i64 : i64 to index
    %lb = arith.index_cast %c0_i64 : i64 to index
    %step = arith.index_cast %c1_i64 : i64 to index
    %range = arith.subi %stop, %start : index
    %ub = arith.ceildivsi %range, %neg_step : index

    %r = scf.for %iter = %lb to %ub step %step iter_args(%reg = %arg0) -> (!quantum.reg) {
        %idx = arith.muli %iter, %neg_step : index
        %wire = arith.addi %start, %idx : index
        %wire_i64 = arith.index_cast %wire : index to i64
        %q = quantum.extract %reg[%wire_i64] : !quantum.reg -> !quantum.bit
        %out = quantum.custom "PauliX"() %q : !quantum.bit
        %reg1 = quantum.insert %reg[%wire_i64], %out : !quantum.reg, !quantum.bit
        scf.yield %reg1 : !quantum.reg
    }

    return %r : !quantum.reg
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
// CHECK: "operations": {}
// CHECK: "var_function_calls"
// CHECK: "dyn_for_loop_1": "{{0x[0-9a-f]+}}"

// CHECK-LABEL: "outer_dyn_loop_calls_nested_helper": {
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
// CHECK: "depth"
// CHECK-DAG: "any_commuting_depth": 1
// CHECK-DAG: "qubit_disjoint_depth": 2
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
func.func @multi_qubit_args(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {llvm.emit_c_interface} {
    %0:2 = quantum.custom "CNOT"() %q0, %q1 : !quantum.bit, !quantum.bit
    return %0#0, %0#1 : !quantum.bit, !quantum.bit
}

// -----

// Entry function is `llvm.emit_c_interface`, not the first function in the module.

// CHECK-LABEL: "helper_with_qubit_args": {
// CHECK: "num_arg_qubits": 0

// CHECK-LABEL: "jit_entry_with_qubit_args": {
// CHECK: "num_arg_qubits": 1
func.func private @helper_with_qubit_args(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %q0 : !quantum.bit
    return %out : !quantum.bit
}

func.func @jit_entry_with_qubit_args(%q0: !quantum.bit) -> !quantum.bit attributes {llvm.emit_c_interface} {
    %r = func.call @helper_with_qubit_args(%q0, %q0) : (!quantum.bit, !quantum.bit) -> !quantum.bit
    return %r : !quantum.bit
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
func.func @mixed_alloc_and_arg_qubits(%q0: !quantum.bit) -> !quantum.bit attributes {llvm.emit_c_interface} {
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

// Stats use the `llvm.emit_c_interface` entry and its flattened call graph:
//   first_qnode (0) +
//     2 * shared_helper (1 H) +
//     1 * second_qnode (1 PauliX + 1 * shared_helper (1 H)) = 4.

// STATS: ResourceAnalysisPass
// STATS: 0 total-alloc-qubits
// STATS: 1 total-arg-qubits
// STATS: 4 total-function-calls
// STATS: 4 total-gates
// STATS: 1 total-qubits
func.func @first_qnode(%arg0: !quantum.bit) -> !quantum.bit attributes {llvm.emit_c_interface} {
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

// -----

// Recursive calls are skipped when flattening resources, so the pass should
// warn that resource counts may be incomplete.

// WARN: ResourceAnalysis encountered recursive call to 'recursive'. Recursive calls are not flattened, so resource counts may be incomplete.
func.func @recursive(%arg0: !quantum.bit) -> !quantum.bit {
    %out = func.call @recursive(%arg0) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

// -----

// auto_qubit_management flag is reported in the result.
// The function uses {auto_qubit_management} on the device, so the flag is true.

// CHECK-LABEL: "auto_qm_flag_set"
// CHECK:   "auto_qubit_management": true
// CHECK:   "device_name": "NullQubit"
func.func @auto_qm_flag_set() {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["librtd_null_qubit.so", "NullQubit", "{}"] {auto_qubit_management}
    %0 = quantum.alloc( 0) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "Hadamard"() %1 : !quantum.bit
    %2 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %2 : !quantum.reg
    quantum.device_release
    return
}

// -----

// quantum.device present but without auto_qubit_management: flag is false.

// CHECK-LABEL: "auto_qm_flag_unset"
// CHECK:   "auto_qubit_management": false
func.func @auto_qm_flag_unset() {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["librtd_null_qubit.so", "NullQubit", "{}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %q = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "Hadamard"() %q : !quantum.bit
    %r = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %r : !quantum.reg
    quantum.device_release
    return
}

// -----

// general operations in reference semantics

// CHECK-LABEL: "qref"

// CHECK: "measurements"
// CHECK-DAG: "MidCircuitMeasure": 1

// CHECK:   "num_alloc_qubits": 6
// CHECK:   "num_arg_qubits": 3
// CHECK:   "num_qubits": 9

// CHECK:   "operations"
// CHECK-DAG: "Adjoint(Hadamard)(1)": 1
// CHECK-DAG: "CNOT(2)": 1
// CHECK-DAG: "Adjoint(T)(1)": 1
// CHECK-DAG: "S(1)": 1
// CHECK-PPM: "PPM(0)": 1,
// CHECK-DAG: "mbqc.ref.graph_state_prep(0)": 1,
// CHECK-DAG: "mbqc.ref.measure_in_basis(0)": 1

func.func @qref(%arg0: !qref.bit, %arg1: !qref.reg<2>) {
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
    ^bb0():
        qref.custom "Hadamard"() %1 : !qref.bit
    }
    qref.custom "T"() %1 adj : !qref.bit
    qref.custom "S"() %2 : !qref.bit
    qref.custom "CNOT"() %1, %2 : !qref.bit, !qref.bit

    %meas = qref.measure %1 : i1

    %pbc_meas = pbc.ref.ppm ["X", "Z"] %1, %2: i1

    %angle = arith.constant 4.0 : f64
    %mbqc_meas = mbqc.ref.measure_in_basis [ZX, %angle] %2 : i1

    %adj_matrix = arith.constant dense<[1, 0, 1, 0, 0, 1]> : tensor<6xi1>
    %graph_reg = mbqc.ref.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !qref.reg<4>

    qref.dealloc %0 : !qref.reg<2>
    return
}

// -----

// scf.if with an `estimated_probability` hint uses the expected (probability-
// weighted) resource counts instead of the worst case. With p(then) = 0.75 the
// then-branch has 4 Hadamards and the else-branch has 8, so the expected count
// is 0.75*4 + 0.25*8 = 5 (the worst case would be 8).

// CHECK-LABEL: "if_estimated_probability"
// CHECK: "has_branches": true
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 5
func.func @if_estimated_probability(%arg0: !quantum.bit, %cond: i1) -> !quantum.bit {
    %q = scf.if %cond -> !quantum.bit {
        %t1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %t2 = quantum.custom "Hadamard"() %t1 : !quantum.bit
        %t3 = quantum.custom "Hadamard"() %t2 : !quantum.bit
        %t4 = quantum.custom "Hadamard"() %t3 : !quantum.bit
        scf.yield %t4 : !quantum.bit
    } else {
        %f1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %f2 = quantum.custom "Hadamard"() %f1 : !quantum.bit
        %f3 = quantum.custom "Hadamard"() %f2 : !quantum.bit
        %f4 = quantum.custom "Hadamard"() %f3 : !quantum.bit
        %f5 = quantum.custom "Hadamard"() %f4 : !quantum.bit
        %f6 = quantum.custom "Hadamard"() %f5 : !quantum.bit
        %f7 = quantum.custom "Hadamard"() %f6 : !quantum.bit
        %f8 = quantum.custom "Hadamard"() %f7 : !quantum.bit
        scf.yield %f8 : !quantum.bit
    } {catalyst.estimated_probability = 0.75 : f64}
    return %q : !quantum.bit
}

// -----

// scf.if with only a then-branch and `estimated_probability` = 0.5: the (empty)
// else-branch contributes nothing, so the expected Hadamard count is
// 0.5 * 3 = 1.5. Fractional expected counts are surfaced as floats in the JSON.

// CHECK-LABEL: "if_estimated_probability_then_only"
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1.5
func.func @if_estimated_probability_then_only(%arg0: !quantum.bit, %cond: i1) {
    scf.if %cond {
        %t1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %t2 = quantum.custom "Hadamard"() %t1 : !quantum.bit
        %t3 = quantum.custom "Hadamard"() %t2 : !quantum.bit
        scf.yield
    } {catalyst.estimated_probability = 0.5 : f64}
    return
}

// -----

// Qubit allocations are probability-weighted like every other count. Here the
// then-branch allocates 1 qubit and the (empty) else-branch allocates none,
// with p(then) = 0.5, so the expected allocation count is 0.5.

// CHECK-LABEL: "if_estimated_probability_qubits"
// CHECK: "num_alloc_qubits": 0.5
// CHECK: "num_qubits": 0.5
func.func @if_estimated_probability_qubits(%cond: i1) {
    scf.if %cond {
        %r = quantum.alloc(1) : !quantum.reg
        %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
        %h = quantum.custom "Hadamard"() %q : !quantum.bit
        scf.yield
    } {catalyst.estimated_probability = 0.5 : f64}
    return
}

// -----

// A probabilistic conditional inside a loop body: the fractional expected count
// (0.5 Hadamard per iteration, p(then) = 0.5) must survive lifting into the
// for_loop_1 body and only be combined with the trip count downstream. The
// lifted body therefore reports a fractional count, and the parent records
// function_calls = { for_loop_1: 10 }.

// CHECK-LABEL: "for_loop_1": {
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 0.5

// CHECK-LABEL: "prob_if_in_loop": {
// CHECK: "function_calls"
// CHECK: "for_loop_1": 10
func.func @prob_if_in_loop(%arg0: !quantum.bit, %cond: i1) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %q = scf.for %i = %c0 to %c10 step %c1 iter_args(%a = %arg0) -> (!quantum.bit) {
        scf.if %cond {
            %h = quantum.custom "Hadamard"() %a : !quantum.bit
            scf.yield
        } {catalyst.estimated_probability = 0.5 : f64}
        scf.yield %a : !quantum.bit
    }
    return %q : !quantum.bit
}

// -----

// scf.index_switch with an `estimated_probabilities` hint (one entry per case,
// in case order). The default case probability is computed automatically as the
// remaining mass: 1 - (0.2 + 0.3) = 0.5. With case 0 = 5, case 1 = 10 and
// default = 2 PauliX gates, the expected count is
// 0.2*5 + 0.3*10 + 0.5*2 = 5.

// CHECK-LABEL: "switch_estimated_probabilities"
// CHECK: "has_branches": true
// CHECK: "operations"
// CHECK-DAG: "PauliX(1)": 5
func.func @switch_estimated_probabilities(%arg0: !quantum.bit, %sel: index) -> !quantum.bit {
    %q = scf.index_switch %sel {catalyst.estimated_probabilities = [0.2 : f64, 0.3 : f64]} -> !quantum.bit
    case 0 {
        %c1 = quantum.custom "PauliX"() %arg0 : !quantum.bit
        %c2 = quantum.custom "PauliX"() %c1 : !quantum.bit
        %c3 = quantum.custom "PauliX"() %c2 : !quantum.bit
        %c4 = quantum.custom "PauliX"() %c3 : !quantum.bit
        %c5 = quantum.custom "PauliX"() %c4 : !quantum.bit
        scf.yield %c5 : !quantum.bit
    }
    case 1 {
        %d1 = quantum.custom "PauliX"() %arg0 : !quantum.bit
        %d2 = quantum.custom "PauliX"() %d1 : !quantum.bit
        %d3 = quantum.custom "PauliX"() %d2 : !quantum.bit
        %d4 = quantum.custom "PauliX"() %d3 : !quantum.bit
        %d5 = quantum.custom "PauliX"() %d4 : !quantum.bit
        %d6 = quantum.custom "PauliX"() %d5 : !quantum.bit
        %d7 = quantum.custom "PauliX"() %d6 : !quantum.bit
        %d8 = quantum.custom "PauliX"() %d7 : !quantum.bit
        %d9 = quantum.custom "PauliX"() %d8 : !quantum.bit
        %d10 = quantum.custom "PauliX"() %d9 : !quantum.bit
        scf.yield %d10 : !quantum.bit
    }
    default {
        %e1 = quantum.custom "PauliX"() %arg0 : !quantum.bit
        %e2 = quantum.custom "PauliX"() %e1 : !quantum.bit
        scf.yield %e2 : !quantum.bit
    }
    return %q : !quantum.bit
}

// -----

// Resource analysis should include functions inside nested modules.

// CHECK-LABEL: "circuit"
// CHECK: "num_alloc_qubits": 1
// CHECK: "operations"
// CHECK-DAG: "Hadamard(1)": 1

// CHECK-LABEL: "jit_circuit"
// CHECK: "catalyst.launch_kernel": 1
module @nested_resource_module {
  func.func public @jit_circuit() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_circuit::@circuit() : () -> ()
    return
  }

  module @module_circuit {
    func.func public @circuit() attributes {quantum.node} {
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
      quantum.dealloc %3 : !quantum.reg
      return
    }
  }
}
