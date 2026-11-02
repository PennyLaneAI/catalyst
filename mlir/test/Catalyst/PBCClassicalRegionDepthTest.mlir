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

// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// A classical region's captured measurement must remain a dependency.
// CHECK-LABEL: "classical_capture": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @classical_capture(%q0: !quantum.bit, %q1: !quantum.bit) {
    %m, %out = pbc.ppm ["Z"] %q0 : i1, !quantum.bit
    %condition = scf.execute_region -> i1 {
        scf.yield %m : i1
    }
    %r = pbc.ppr ["X"](8) %q1 cond(%condition) : !quantum.bit
    return
}

// -----

// The dependence also survives a memory write inside a classical region.
// CHECK-LABEL: "classical_memory": {
// CHECK: "qubit_disjoint_depth_total": 3
func.func @classical_memory(%q0: !quantum.bit, %q1: !quantum.bit, %buffer: memref<i1>) {
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %a : i1, !quantum.bit
    scf.execute_region {
        memref.store %m, %buffer[] : memref<i1>
        scf.yield
    }
    %condition = memref.load %buffer[] : memref<i1>
    %r = pbc.ppr ["X"](8) %q1 cond(%condition) : !quantum.bit
    return
}

// -----

// No quantum work may be hidden by a region with only classical results.
// CHECK-LABEL: "hidden_quantum_region": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @hidden_quantum_region(%q: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %m = scf.execute_region -> i1 {
        %fresh = quantum.alloc_qb : !quantum.bit
        %measurement, %out = pbc.ppm ["Z"] %fresh : i1, !quantum.bit
        scf.yield %measurement : i1
    }
    return
}

// -----

// Calls inside an otherwise classical region are not assumed quantum-free.
// CHECK-LABEL: "hidden_call_region": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @hidden_call_region(%q: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    scf.execute_region {
        func.call @opaque() : () -> ()
        scf.yield
    }
    return
}
func.func private @opaque()

// -----

// Calls are supported after recursively proving that every reachable body is
// classical. A captured measurement still constrains the following PPR.
// CHECK-LABEL: "classical_calls": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @classical_calls(%q0: !quantum.bit, %q1: !quantum.bit) {
    %m, %out = pbc.ppm ["Z"] %q0 : i1, !quantum.bit
    %condition = scf.execute_region -> i1 {
        %value = func.call @classical_outer(%m) : (i1) -> i1
        scf.yield %value : i1
    }
    %r = pbc.ppr ["X"](8) %q1 cond(%condition) : !quantum.bit
    return
}

func.func private @classical_outer(%condition: i1) -> i1 {
    %out = func.call @classical_inner(%condition) : (i1) -> i1
    return %out : i1
}

func.func private @classical_inner(%condition: i1) -> i1 {
    %false = arith.constant false
    %out = arith.xori %condition, %false : i1
    return %out : i1
}

// -----

// A resolved call can still hide quantum operations despite a classical type.
// CHECK-LABEL: "quantum_call_region": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @quantum_call_region(%q: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    scf.execute_region {
        func.call @quantum_helper() : () -> ()
        scf.yield
    }
    return
}

func.func private @quantum_helper() {
    %q = quantum.alloc_qb : !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    quantum.dealloc_qb %a : !quantum.bit
    return
}
