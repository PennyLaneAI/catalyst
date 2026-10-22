// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test that resource analysis hints on SCF ops must survive upstream mlir canonicalization.

// RUN: quantum-opt --canonicalize --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_preserves_compiler_hints
func.func @test_preserves_compiler_hints(%cond: i1, %idx: index) attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    %q = qref.alloc_qb : !qref.bit

    // CHECK: scf.for
    // CHECK: {catalyst.estimated_iterations = 1.000000e+01 : f64}
    scf.for %i = %start to %stop step %step {
        qref.custom "Hadamard"() %q : !qref.bit
        scf.yield
    } {catalyst.estimated_iterations = 1.000000e+01 : f64}

    // CHECK: scf.while
    // CHECK: {catalyst.estimated_iterations = 2.500000e+00 : f64}
    scf.while () : () -> () {
        scf.condition(%cond)
    } do {
        qref.custom "PauliX"() %q : !qref.bit
        scf.yield
    } attributes {catalyst.estimated_iterations = 2.500000e+00 : f64}

    // CHECK: scf.if
    // CHECK: {catalyst.estimated_probability = 7.500000e-01 : f64}
    scf.if %cond {
        qref.custom "PauliY"() %q : !qref.bit
        scf.yield
    } else {
        scf.yield
    } {catalyst.estimated_probability = 7.500000e-01 : f64}

    // CHECK: scf.index_switch {{.+}} {catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]}
    scf.index_switch %idx {catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]}
    case 0 {
        qref.custom "PauliZ"() %q : !qref.bit
        scf.yield
    }
    case 1 {
        scf.yield
    }
    default {
        scf.yield
    }

    qref.dealloc_qb %q : !qref.bit
    return
}
