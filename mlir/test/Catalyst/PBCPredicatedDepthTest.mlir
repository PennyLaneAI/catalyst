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

// Both representations describe the same predicated circuit. Classical
// feed-forward latency is outside this qubit-only logical depth metric.
// CHECK-LABEL: "explicit_disjoint_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @explicit_disjoint_correction(%meter: !quantum.bit, %data: !quantum.bit) {
    %a = pbc.ppr ["X"](2) %meter : !quantum.bit
    %b = pbc.ppr ["Z"](2) %a : !quantum.bit
    %m, %after = pbc.ppm ["Z"] %b : i1, !quantum.bit
    %corrected = pbc.ppr ["X"](2) %data cond(%m) : !quantum.bit
    return
}
// CHECK-LABEL: "structured_disjoint_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @structured_disjoint_correction(%meter: !quantum.bit, %data: !quantum.bit) {
    %a = pbc.ppr ["X"](2) %meter : !quantum.bit
    %b = pbc.ppr ["Z"](2) %a : !quantum.bit
    %m, %after = pbc.ppm ["Z"] %b : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}

// -----

// Reusing the measured qubit imposes a quantum dependency in both forms.
// CHECK-LABEL: "explicit_same_wire_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @explicit_same_wire_correction(%q: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %q : i1, !quantum.bit
    %corrected = pbc.ppr ["X"](2) %after cond(%m) : !quantum.bit
    return
}
// CHECK-LABEL: "structured_same_wire_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @structured_same_wire_correction(%q: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %q : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %after : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %after : !quantum.bit
    }
    return
}

// -----

// The measurement predicate can travel through scalar/tensor conversions and
// a classical helper, as in the emitted TemporaryAND correction.
// CHECK-LABEL: "condition_through_helper": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @condition_through_helper(%meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["X"] %meter : i1, !quantum.bit
    %tensor = tensor.from_elements %m : tensor<i1>
    %condition_tensor = func.call @z_condition(%tensor) : (tensor<i1>) -> tensor<i1>
    %condition = tensor.extract %condition_tensor[] : tensor<i1>
    %corrected = scf.if %condition -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}
func.func private @z_condition(%input: tensor<i1>) -> tensor<i1> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %integer = stablehlo.convert %input : (tensor<i1>) -> tensor<i64>
    %condition = stablehlo.compare NE, %integer, %zero, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    return %condition : tensor<i1>
}

// -----

// A unitary helper retains parallelism inside a multi-qubit correction.
// CHECK-LABEL: "explicit_pair_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @explicit_pair_correction(%meter: !quantum.bit, %a: !quantum.bit, %b: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %a1 = pbc.ppr ["Z"](4) %a cond(%m) : !quantum.bit
    %b1 = pbc.ppr ["Z"](4) %b cond(%m) : !quantum.bit
    %out:2 = pbc.ppr ["Z", "Z"](4) %a1, %b1 cond(%m) : !quantum.bit, !quantum.bit
    return
}
// CHECK-LABEL: "structured_pair_correction": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @structured_pair_correction(%meter: !quantum.bit, %a: !quantum.bit, %b: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected:2 = scf.if %m -> (!quantum.bit, !quantum.bit) {
        %out:2 = func.call @z_pair(%a, %b) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        scf.yield %out#0, %out#1 : !quantum.bit, !quantum.bit
    } else {
        scf.yield %a, %b : !quantum.bit, !quantum.bit
    }
    return
}
func.func private @z_pair(%a: !quantum.bit, %b: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %a1 = pbc.ppr ["Z"](4) %a : !quantum.bit
    %b1 = pbc.ppr ["Z"](4) %b : !quantum.bit
    %out:2 = pbc.ppr ["Z", "Z"](4) %a1, %b1 : !quantum.bit, !quantum.bit
    return %out#0, %out#1 : !quantum.bit, !quantum.bit
}

// -----

// Measurement provenance is retained by a function result.
// CHECK-LABEL: "measurement_from_helper": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @measurement_from_helper(%q: !quantum.bit) {
    %m, %after = func.call @z_measure(%q) : (!quantum.bit) -> (i1, !quantum.bit)
    %corrected = scf.if %m -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %after : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %after : !quantum.bit
    }
    return
}
func.func private @z_measure(%q: !quantum.bit) -> (i1, !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %q : i1, !quantum.bit
    return %m, %after : i1, !quantum.bit
}

// -----

// Unknown classical graph selection is not relabeled as a measured correction.
// CHECK-LABEL: "unknown_classical_predicate": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @unknown_classical_predicate(%condition: i1, %q: !quantum.bit) {
    %corrected = scf.if %condition -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %q : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }
    return
}

// -----

// A measurement argument alone does not make an unrelated helper result a
// measured predicate.
// CHECK-LABEL: "unrelated_helper_result": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @unrelated_helper_result(%unknown: i1, %meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %condition = func.call @z_unrelated(%m, %unknown) : (i1, i1) -> i1
    %corrected = scf.if %condition -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}
func.func private @z_unrelated(%measurement: i1, %unknown: i1) -> i1 {
    return %unknown : i1
}

// -----

// Every nonconstant predicate input must be measurement-derived.
// CHECK-LABEL: "mixed_unknown_predicate": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @mixed_unknown_predicate(%unknown: i1, %meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %condition = arith.andi %m, %unknown : i1
    %corrected = scf.if %condition -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}

// -----

// This restricted normalization requires an identity else arm.
// CHECK-LABEL: "nonidentity_else_arm": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @nonidentity_else_arm(%meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %a = pbc.ppr ["X"](2) %data : !quantum.bit
        %b = pbc.ppr ["Z"](2) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } else {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    }
    return
}

// -----

// A predicated block can contain a measurement whose result does not escape.
// CHECK-LABEL: "predicated_measurement": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @predicated_measurement(%meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %other_m, %out = pbc.ppm ["X"] %data : i1, !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}

// -----

// Internal ancillas and their correction predicates stay inside the block.
// CHECK-LABEL: "predicated_injection": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @predicated_injection(%meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %ancilla = pbc.prepare zero : !quantum.bit
        %pair:2 = pbc.ppr ["X", "Z"](8) %data, %ancilla : !quantum.bit, !quantum.bit
        %inner_m, %inner_after = pbc.ppm ["Z"] %pair#1 : i1, !quantum.bit
        %out = pbc.ppr ["Z"](2) %pair#0 cond(%inner_m) : !quantum.bit
        quantum.dealloc_qb %inner_after : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}

// -----

// A conditional classical result is not replaced with the then-arm value.
// CHECK-LABEL: "escaping_classical_result": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @escaping_classical_result(%meter: !quantum.bit, %data: !quantum.bit) {
    %false = arith.constant false
    %true = arith.constant true
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected, %result = scf.if %m -> (!quantum.bit, i1) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out, %true : !quantum.bit, i1
    } else {
        scf.yield %data, %false : !quantum.bit, i1
    }
    return
}

// -----

// A cached helper result must retain which unknown inputs came from a
// measurement; an unrelated runtime input must use a different cache entry.
// CHECK-LABEL: "cached_measurement_provenance": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @cached_measurement_provenance(%unknown: i1, %meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %first = func.call @z_forward(%m) : (i1) -> i1
    %second = func.call @z_forward(%unknown) : (i1) -> i1
    %corrected = scf.if %second -> (!quantum.bit) {
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}
func.func private @z_forward(%input: i1) -> i1 {
    return %input : i1
}

// -----

// A conditional allocation changes identities, rather than predicating gates.
// CHECK-LABEL: "allocating_then_arm": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @allocating_then_arm(%meter: !quantum.bit, %data: !quantum.bit) {
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        %fresh = pbc.prepare zero : !quantum.bit
        %out = pbc.ppr ["X"](2) %fresh : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}

// -----

// Classical side effects cannot be made unconditional with a quantum predicate.
// CHECK-LABEL: "mutating_then_arm": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total":
// CHECK: "function_calls":
func.func @mutating_then_arm(%meter: !quantum.bit, %data: !quantum.bit) {
    %zero = arith.constant 0 : i64
    %stack = catalyst.list_init : <i64>
    %m, %after = pbc.ppm ["Z"] %meter : i1, !quantum.bit
    %corrected = scf.if %m -> (!quantum.bit) {
        catalyst.list_push %zero, %stack : <i64>
        %out = pbc.ppr ["X"](2) %data : !quantum.bit
        scf.yield %out : !quantum.bit
    } else {
        scf.yield %data : !quantum.bit
    }
    return
}
