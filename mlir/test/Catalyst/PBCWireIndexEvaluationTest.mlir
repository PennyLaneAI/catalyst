// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s
// RUN: %PYTHON %S/Inputs/pbc_wire_index_chain.py | quantum-opt -resource-analysis=output-json -o /dev/null | FileCheck %s --check-prefix=CHAIN

// Long classical DAGs are evaluated without a recursive stack. A known select
// condition still resolves its chosen operand when the other one is unknown.
// CHAIN-LABEL: "long_local_classical_chain": {
// CHAIN: "qubit_disjoint_depth_status": "exact"
// CHAIN: "qubit_disjoint_depth_total": 1

// Constant arithmetic and boolean expressions determine the chosen graph.
// CHECK-LABEL: "integer_boolean_arithmetic": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @integer_boolean_arithmetic(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %left = stablehlo.constant dense<[-1, 1]> : tensor<2xi64>
    %right = stablehlo.constant dense<1> : tensor<2xi64>
    %zero = stablehlo.constant dense<0> : tensor<2xi64>
    %sum = stablehlo.add %left, %right : tensor<2xi64>
    %positive = stablehlo.compare GT, %sum, %zero, SIGNED : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %nonpositive = stablehlo.not %positive : tensor<2xi1>
    %condition = tensor.extract %nonpositive[%c0] : tensor<2xi1>
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// Dynamic slicing of a known multidimensional tensor gives a known wire.
// CHECK-LABEL: "multidimensional_dynamic_slice": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @multidimensional_dynamic_slice() {
    %c1 = arith.constant 1 : index
    %one = stablehlo.constant dense<1> : tensor<i64>
    %table = stablehlo.iota dim = 1 : tensor<2x3xi64>
    %slice = stablehlo.dynamic_slice %table, %one, %one, sizes = [1, 2] : (tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<1x2xi64>
    %flat = stablehlo.reshape %slice : (tensor<1x2xi64>) -> tensor<2xi64>
    %wire = tensor.extract %flat[%c1] : tensor<2xi64>
    %reg = quantum.alloc(3) : !quantum.reg
    %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["X"](8) %q0 : !quantum.bit
    return
}

// -----

// Out-of-bounds starts are clamped according to stablehlo.dynamic_slice.
// CHECK-LABEL: "clamped_dynamic_slice": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @clamped_dynamic_slice() {
    %start = stablehlo.constant dense<99> : tensor<i64>
    %table = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi64>
    %slice = stablehlo.dynamic_slice %table, %start, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %wire = tensor.extract %scalar[] : tensor<i64>
    %reg = quantum.alloc(3) : !quantum.reg
    %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["X"](8) %q0 : !quantum.bit
    return
}

// -----

// Concatenation and static slicing retain row-major multidimensional indices.
// CHECK-LABEL: "concatenate_and_slice": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @concatenate_and_slice() {
    %c0 = arith.constant 0 : index
    %left = stablehlo.constant dense<[[0, 1]]> : tensor<1x2xi64>
    %right = stablehlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
    %table = stablehlo.concatenate %left, %right, dim = 0 : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<2x2xi64>
    %slice = stablehlo.slice %table [1:2, 0:2] : (tensor<2x2xi64>) -> tensor<1x2xi64>
    %flat = stablehlo.reshape %slice : (tensor<1x2xi64>) -> tensor<2xi64>
    %wire = tensor.extract %flat[%c0] : tensor<2xi64>
    %reg = quantum.alloc(4) : !quantum.reg
    %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["X"](8) %q0 : !quantum.bit
    return
}

// -----

// A concrete reduction can determine a loop bound independently of its hint.
// CHECK-LABEL: "integer_reduction_bound": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @integer_reduction_bound(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %table = stablehlo.constant dense<[1, 3, 2]> : tensor<3xi64>
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %maximum = stablehlo.reduce(%table init: %zero) applies stablehlo.maximum across dimensions = [0] : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %count = tensor.extract %maximum[] : tensor<i64>
    %upper = arith.index_cast %count : i64 to index
    %out = scf.for %i = %c0 to %upper step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } {catalyst.estimated_iterations = 1.5 : f64}
    return
}

// -----

// Narrow integer operations use their declared width, including wrapping.
// CHECK-LABEL: "narrow_integer_arithmetic": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @narrow_integer_arithmetic(%q: !quantum.bit) {
    %max = stablehlo.constant dense<127> : tensor<i8>
    %one = stablehlo.constant dense<1> : tensor<i8>
    %zero = stablehlo.constant dense<0> : tensor<i8>
    %wrapped = stablehlo.add %max, %one : tensor<i8>
    %negative = stablehlo.compare LT, %wrapped, %zero, SIGNED : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %condition = tensor.extract %negative[] : tensor<i1>
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// Boolean addition is logical OR, unlike wrapping arith.addi on i1.
// CHECK-LABEL: "boolean_addition": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @boolean_addition(%q: !quantum.bit) {
    %true = stablehlo.constant dense<true> : tensor<i1>
    %sum = stablehlo.add %true, %true : tensor<i1>
    %condition = tensor.extract %sum[] : tensor<i1>
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// An omitted comparison type is inferred from the boolean operand type.
// CHECK-LABEL: "inferred_boolean_comparison": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @inferred_boolean_comparison(%q: !quantum.bit) {
    %true = stablehlo.constant dense<true> : tensor<i1>
    %false = stablehlo.constant dense<false> : tensor<i1>
    %greater = stablehlo.compare GT, %true, %false : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %condition = tensor.extract %greater[] : tensor<i1>
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// The local evaluator must not choose a tree for a non-associative reduction.
// These cases run without the optional native classical evaluator.
// CHECK-LABEL: "non_associative_reduction": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @non_associative_reduction(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %table = stablehlo.constant dense<[4, 3, 2]> : tensor<3xi64>
    %initial = stablehlo.constant dense<10> : tensor<i64>
    %reduced = "stablehlo.reduce"(%table, %initial) ({
      ^bb0(%left: tensor<i64>, %right: tensor<i64>):
        %difference = stablehlo.subtract %left, %right : tensor<i64>
        stablehlo.return %difference : tensor<i64>
    }) {dimensions = array<i64: 0>} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %count = tensor.extract %reduced[] : tensor<i64>
    %upper = arith.index_cast %count : i64 to index
    %out = scf.for %i = %c0 to %upper step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// An implementation may insert a reduction's initial value multiple times.
// CHECK-LABEL: "non_neutral_reduction_initial_value": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @non_neutral_reduction_initial_value(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %table = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
    %initial = stablehlo.constant dense<1> : tensor<i64>
    %reduced = stablehlo.reduce(%table init: %initial) applies stablehlo.add across dimensions = [0] : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %count = tensor.extract %reduced[] : tensor<i64>
    %upper = arith.index_cast %count : i64 to index
    %out = scf.for %i = %c0 to %upper step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// Rebinding either loop invalidates its own computations and nested regions.
// The outer offset remains invariant while the inner loop chooses two wires.
// CHECK-LABEL: "nested_region_indices": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @nested_region_indices() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(4) : !quantum.reg
    scf.for %i = %c0 to %c2 step %c1 {
        %offset = arith.muli %i, %c2 : index
        scf.for %j = %c0 to %c2 step %c1 {
            %index = arith.addi %offset, %j : index
            %wire = arith.index_cast %index : index to i64
            %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
            %out = pbc.ppr ["X"](8) %q : !quantum.bit
        }
    }
    return
}

// -----

// An if result is rebound in the enclosing region before its consumers run.
// CHECK-LABEL: "branch_result_rebinding": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @branch_result_rebinding() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(2) : !quantum.reg
    scf.for %i = %c0 to %c2 step %c1 {
        %first = arith.cmpi eq, %i, %c0 : index
        %selected = scf.if %first -> (index) {
            scf.yield %c0 : index
        } else {
            scf.yield %c1 : index
        }
        %index = arith.addi %selected, %c0 : index
        %wire = arith.index_cast %index : index to i64
        %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
    }
    return
}

// -----

// A nested loop's result can differ each time the enclosing loop runs.
// CHECK-LABEL: "loop_result_rebinding": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @loop_result_rebinding() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(3) : !quantum.reg
    scf.for %i = %c0 to %c2 step %c1 {
        %selected = scf.for %j = %c0 to %c1 step %c1 iter_args(%value = %i) -> (index) {
            %next = arith.addi %value, %c1 : index
            scf.yield %next : index
        }
        %index = arith.addi %selected, %c0 : index
        %wire = arith.index_cast %index : index to i64
        %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
    }
    return
}

// -----

// Callee arguments and caller results both change between calls.
// CHECK-LABEL: "call_result_rebinding": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @call_result_rebinding() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(3) : !quantum.reg
    scf.for %i = %c0 to %c2 step %c1 {
        %selected = func.call @increment(%i) : (index) -> index
        %index = arith.addi %selected, %c0 : index
        %wire = arith.index_cast %index : index to i64
        %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
    }
    return
}
func.func private @increment(%value: index) -> index {
    %c1 = arith.constant 1 : index
    %next = arith.addi %value, %c1 : index
    return %next : index
}

// -----

// A formerly concrete result must not survive an unknown binding.
// CHECK-LABEL: "unknown_after_known_binding": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @unknown_after_known_binding(%unknown: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(2) : !quantum.reg
    scf.for %i = %c0 to %c2 step %c1 {
        %first = arith.cmpi eq, %i, %c0 : index
        %selected = scf.if %first -> (index) {
            scf.yield %c0 : index
        } else {
            scf.yield %unknown : index
        }
        %index = arith.addi %selected, %c0 : index
        %wire = arith.index_cast %index : index to i64
        %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
    }
    return
}

// -----

// A no-wrap promise must not be interpreted as ordinary wrapping arithmetic.
// CHECK-LABEL: "poison_integer_overflow": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @poison_integer_overflow(%q: !quantum.bit) {
    %max = arith.constant 127 : i8
    %one = arith.constant 1 : i8
    %zero = arith.constant 0 : i8
    %sum = arith.addi %max, %one overflow<nsw> : i8
    %condition = arith.cmpi slt, %sum, %zero : i8
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}
