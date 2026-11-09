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

// RUN: quantum-opt --instrument-classical-control --split-input-file %s | FileCheck %s

// m = measure(0); cond(m, Z)(1)

// CHECK-DAG: func.func private @__catalyst__rt__measurement_token() -> i64
// CHECK-DAG: func.func private @__catalyst__rt__begin_classical_control(i64)
// CHECK-DAG: func.func private @__catalyst__rt__end_classical_control()

// CHECK-LABEL: func.func @cond_on_measurement
func.func @cond_on_measurement(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[M:%.+]], [[Q0:%.+]] = quantum.measure
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token() : () -> i64
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit

    // CHECK: scf.if [[M]]
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]]) : (i64) -> ()
    // CHECK-NEXT: quantum.custom "PauliZ"()
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control() : () -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: } else {
    // CHECK-NOT: call @__catalyst__rt__begin_classical_control
    // CHECK: scf.yield
    %q1_1 = scf.if %m -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q1 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    return %q0_1, %q1_1 : !quantum.bit, !quantum.bit
}

// -----

// Conditions that do not depend on measurements are not instrumented.

// CHECK-LABEL: func.func @cond_on_argument
func.func @cond_on_argument(%c: i1, %q: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: call @__catalyst__rt__
    %q_1 = scf.if %c -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }
    return %q_1 : !quantum.bit
}

// -----

// Values computed from the quantum state do not depend on earlier measurements through the
// qubits they act on.

// CHECK-LABEL: func.func @cond_on_expval
func.func @cond_on_expval(%q0: !quantum.bit, %q1: !quantum.bit, %t: f64) -> !quantum.bit {
    // CHECK-NOT: call @__catalyst__rt__
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %obs = quantum.namedobs %q0_1[ PauliZ] : !quantum.obs
    %e = quantum.expval %obs : f64
    %c = arith.cmpf ogt, %e, %t : f64
    %q1_1 = scf.if %c -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q1 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    return %q1_1 : !quantum.bit
}

// -----

// A condition on several measurements is controlled by the latest of them.

// CHECK-LABEL: func.func @cond_on_two_measurements
func.func @cond_on_two_measurements(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> !quantum.bit {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[T0:%.+]] = call @__catalyst__rt__measurement_token()
    // CHECK: quantum.measure
    // CHECK-NEXT: [[T1:%.+]] = call @__catalyst__rt__measurement_token()
    %m0, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %m1, %q1_1 = quantum.measure %q1 : i1, !quantum.bit
    %c = arith.andi %m0, %m1 : i1

    // CHECK: scf.if
    // CHECK-NEXT: [[TOKEN:%.+]] = arith.maxsi [[T1]], [[T0]] : i64
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    %q2_1 = scf.if %c -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q2 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q2 : !quantum.bit
    }
    return %q2_1 : !quantum.bit
}

// -----

// Measurement results are traced through tensors, as emitted by the frontend.

// CHECK-LABEL: func.func @cond_on_tensor
func.func @cond_on_tensor(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %t = tensor.from_elements %m : tensor<i1>
    %c = tensor.extract %t[] : tensor<i1>

    // CHECK: scf.if
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    %q1_1 = scf.if %c -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q1 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    return %q1_1 : !quantum.bit
}

// -----

// Nested conditionals are each controlled by their own measurements, and the runtime combines
// them with the enclosing region.

// CHECK-LABEL: func.func @nested_cond
func.func @nested_cond(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[T0:%.+]] = call @__catalyst__rt__measurement_token()
    %m0, %q0_1 = quantum.measure %q0 : i1, !quantum.bit

    // CHECK: scf.if
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[T0]])
    // CHECK: quantum.measure
    // CHECK-NEXT: [[T1:%.+]] = {{(func\.)?}}call @__catalyst__rt__measurement_token()
    // CHECK: scf.if
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[T1]])
    // CHECK-NEXT: quantum.custom "PauliZ"()
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    // CHECK-NEXT: scf.yield
    // CHECK: call @__catalyst__rt__end_classical_control()
    // CHECK-NEXT: scf.yield
    %r:2 = scf.if %m0 -> (!quantum.bit, !quantum.bit) {
        %m1, %q1_1 = quantum.measure %q1 : i1, !quantum.bit
        %q2_1 = scf.if %m1 -> !quantum.bit {
            %z = quantum.custom "PauliZ"() %q2 : !quantum.bit
            scf.yield %z : !quantum.bit
        } else {
            scf.yield %q2 : !quantum.bit
        }
        scf.yield %q1_1, %q2_1 : !quantum.bit, !quantum.bit
    } else {
        scf.yield %q1, %q2 : !quantum.bit, !quantum.bit
    }
    return %r#0, %r#1 : !quantum.bit, !quantum.bit
}

// -----

// Branches of an index switch on a measurement are classically controlled.

// CHECK-LABEL: func.func @switch_on_measurement
func.func @switch_on_measurement(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %i = arith.index_cast %m : i1 to index

    // CHECK: scf.index_switch
    // CHECK: case 0 {
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    // CHECK-NEXT: quantum.custom "PauliX"()
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    // CHECK: default {
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    // CHECK-NEXT: quantum.custom "PauliZ"()
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    %q1_1 = scf.index_switch %i -> !quantum.bit
    case 0 {
        %x = quantum.custom "PauliX"() %q1 : !quantum.bit
        scf.yield %x : !quantum.bit
    }
    default {
        %z = quantum.custom "PauliZ"() %q1 : !quantum.bit
        scf.yield %z : !quantum.bit
    }
    return %q1_1 : !quantum.bit
}

// -----

// The body of a repeat-until-success loop is controlled by the measurement of the previous
// iteration, which is carried by the loop and represented by the most recent measurement.

// CHECK-LABEL: func.func @repeat_until_success
func.func @repeat_until_success(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[T0:%.+]] = call @__catalyst__rt__measurement_token()
    %m0, %q0_1 = quantum.measure %q0 : i1, !quantum.bit

    // CHECK: scf.while
    // CHECK-NOT: call @__catalyst__rt__begin_classical_control
    // CHECK: scf.condition
    // CHECK: } do {
    // CHECK-NEXT: ^bb0
    // CHECK-NEXT: [[LATEST:%.+]] = {{(func\.)?}}call @__catalyst__rt__measurement_token()
    // CHECK-NEXT: [[TOKEN:%.+]] = arith.maxsi [[T0]], [[LATEST]] : i64
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    // CHECK-NEXT: quantum.custom "Hadamard"()
    // CHECK-NEXT: quantum.measure
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    // CHECK-NEXT: scf.yield
    %r:3 = scf.while (%m = %m0, %qa = %q0_1, %qb = %q1) : (i1, !quantum.bit, !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
        scf.condition(%m) %m, %qa, %qb : i1, !quantum.bit, !quantum.bit
    } do {
    ^bb0(%m_1: i1, %qa_1: !quantum.bit, %qb_1: !quantum.bit):
        %h = quantum.custom "Hadamard"() %qa_1 : !quantum.bit
        %m_2, %qa_2 = quantum.measure %h : i1, !quantum.bit
        scf.yield %m_2, %qa_2, %qb_1 : i1, !quantum.bit, !quantum.bit
    }
    return %r#1, %r#2 : !quantum.bit, !quantum.bit
}

// -----

// The body of a loop whose bounds depend on a measurement is classically controlled.

// CHECK-LABEL: func.func @for_on_measurement
func.func @for_on_measurement(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    // CHECK: quantum.measure
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %m : i1 to index

    // CHECK: scf.for
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    // CHECK-NEXT: quantum.custom "PauliX"()
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    // CHECK-NEXT: scf.yield
    %q1_1 = scf.for %i = %c0 to %n step %c1 iter_args(%q = %q1) -> !quantum.bit {
        %x = quantum.custom "PauliX"() %q : !quantum.bit
        scf.yield %x : !quantum.bit
    }
    return %q1_1 : !quantum.bit
}

// -----

// Regions that do not apply quantum operations are not instrumented.

// CHECK-LABEL: func.func @classical_branches
func.func @classical_branches(%q0: !quantum.bit, %a: f64, %b: f64) -> f64 {
    // CHECK-NOT: call @__catalyst__rt__
    %m, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %r = scf.if %m -> f64 {
        %s = arith.addf %a, %b : f64
        scf.yield %s : f64
    } else {
        scf.yield %a : f64
    }
    return %r : f64
}

// -----

// A Pauli measurement whose basis is selected by a measurement is classically controlled.

// CHECK-LABEL: func.func @select_ppm_on_measurement
func.func @select_ppm_on_measurement(%q0: !quantum.bit, %q1: !quantum.bit) -> i1 {
    // CHECK: pbc.ppm
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
    %m, %q0_1 = pbc.ppm ["Z"] %q0 : i1, !quantum.bit

    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    // CHECK-NEXT: pbc.select.ppm
    // CHECK-NEXT: call @__catalyst__rt__end_classical_control()
    %mres, %oq = pbc.select.ppm (%m ? ["X"] : ["Z"]) %q1 : i1, !quantum.bit
    return %mres : i1
}
