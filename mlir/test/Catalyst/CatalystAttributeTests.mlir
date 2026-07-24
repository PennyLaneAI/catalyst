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

// Verification of the Catalyst dialect's resource-estimation probability hints.
// RUN: quantum-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Valid hints round-trip unchanged.
// CHECK-LABEL: @valid_if
func.func @valid_if(%arg0: !quantum.bit, %cond: i1) -> !quantum.bit {
    // CHECK: catalyst.estimated_probability = 7.500000e-01
    %q = scf.if %cond -> !quantum.bit {
        %t = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        scf.yield %t : !quantum.bit
    } else {
        scf.yield %arg0 : !quantum.bit
    } {catalyst.estimated_probability = 0.75 : f64}
    return %q : !quantum.bit
}

// -----

// CHECK-LABEL: @valid_switch
func.func @valid_switch(%sel: index) {
    // CHECK: catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]
    scf.index_switch %sel {catalyst.estimated_probabilities = [0.2 : f64, 0.3 : f64]}
    case 0 {
        scf.yield
    }
    case 1 {
        scf.yield
    }
    default {
        scf.yield
    }
    return
}

// -----

// A valid loop trip-count hint round-trips unchanged.
// CHECK-LABEL: @valid_iterations
func.func @valid_iterations(%arg0: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: catalyst.estimated_iterations = 10 : i16
    scf.for %i = %c0 to %n step %c1 {
        scf.yield
    } {catalyst.estimated_iterations = 10 : i16}
    return
}

// -----

func.func @iterations_not_int(%arg0: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{'catalyst.estimated_iterations' must be an integer attribute}}
    scf.for %i = %c0 to %n step %c1 {
        scf.yield
    } {catalyst.estimated_iterations = 1.0 : f64}
    return
}

// -----

func.func @iterations_negative(%arg0: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{'catalyst.estimated_iterations' must be non-negative, but got -1}}
    scf.for %i = %c0 to %n step %c1 {
        scf.yield
    } {catalyst.estimated_iterations = -1 : i64}
    return
}

// -----

// `catalyst.estimated_iterations` must be attached to an scf.for or scf.while.
func.func @iterations_wrong_op(%arg0: !quantum.bit, %cond: i1) {
    // expected-error @+1 {{'catalyst.estimated_iterations' is only valid on 'scf.for' or 'scf.while'}}
    scf.if %cond {
        scf.yield
    } {catalyst.estimated_iterations = 10 : i64}
    return
}

// -----

func.func @prob_not_float(%arg0: !quantum.bit, %cond: i1) {
    // expected-error @+1 {{'catalyst.estimated_probability' must be a float attribute}}
    scf.if %cond {
        scf.yield
    } {catalyst.estimated_probability = 1 : i64}
    return
}

// -----

func.func @prob_out_of_range(%arg0: !quantum.bit, %cond: i1) {
    // expected-error @+1 {{'catalyst.estimated_probability' must be a probability in [0, 1], but got 1.5}}
    scf.if %cond {
        scf.yield
    } {catalyst.estimated_probability = 1.5 : f64}
    return
}

// -----

func.func @probs_not_array(%sel: index) {
    // expected-error @+1 {{'catalyst.estimated_probabilities' must be an array attribute}}
    scf.index_switch %sel {catalyst.estimated_probabilities = 0.5 : f64}
    case 0 {
        scf.yield
    }
    default {
        scf.yield
    }
    return
}

// -----

func.func @probs_element_out_of_range(%sel: index) {
    // expected-error @+1 {{'catalyst.estimated_probabilities' must be a probability in [0, 1], but got 1.5}}
    scf.index_switch %sel {catalyst.estimated_probabilities = [0.2 : f64, 1.5 : f64]}
    case 0 {
        scf.yield
    }
    case 1 {
        scf.yield
    }
    default {
        scf.yield
    }
    return
}

// -----

func.func @probs_sum_too_large(%sel: index) {
    // expected-error @+1 {{'catalyst.estimated_probabilities' entries must sum to at most 1, but got 1.2}}
    scf.index_switch %sel {catalyst.estimated_probabilities = [0.6 : f64, 0.6 : f64]}
    case 0 {
        scf.yield
    }
    case 1 {
        scf.yield
    }
    default {
        scf.yield
    }
    return
}

// -----

func.func @probs_size_mismatch(%sel: index) {
    // expected-error @+1 {{'catalyst.estimated_probabilities' has 1 entries but the switch has 2 case(s)}}
    scf.index_switch %sel {catalyst.estimated_probabilities = [0.5 : f64]}
    case 0 {
        scf.yield
    }
    case 1 {
        scf.yield
    }
    default {
        scf.yield
    }
    return
}

// -----

// `catalyst.estimated_probability` must be attached to an scf.if.
func.func @prob_wrong_op(%arg0: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error @+1 {{'catalyst.estimated_probability' is only valid on 'scf.if'}}
    scf.for %i = %c0 to %c4 step %c1 {
        scf.yield
    } {catalyst.estimated_probability = 0.5 : f64}
    return
}

// -----

// `catalyst.estimated_probabilities` must be attached to an scf.index_switch.
func.func @probs_wrong_op(%arg0: !quantum.bit, %cond: i1) {
    // expected-error @+1 {{'catalyst.estimated_probabilities' is only valid on 'scf.index_switch'}}
    scf.if %cond {
        scf.yield
    } {catalyst.estimated_probabilities = [0.5 : f64]}
    return
}
