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

// RUN: not catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module( graph-decomposition{gate-set=RX=1.0 bytecode-rules="%BYTECODE_PATH"})' %s 2>&1 | FileCheck %s --implicit-check-not=libc++abi --implicit-check-not='uncaught exception'

func.func @circuit(%q0: !quantum.bit) {
    %pi = arith.constant 3.14 : f64
    %out = quantum.pcphase (%pi, dim : 3) %q0 : !quantum.bit

    // CHECK: UserWarning: Python decomposition rule compilation failed for operator 'PCPhase' (id: PCPhase[f64][1]{dim:3})
    // CHECK-SAME:  it will be treated as non-decomposable by the graph solver
    // CHECK: Decomposition rule not found for operator 'pcphase
    return
}

// -----

func.func public @toffoli_circuit() attributes {quantum.node} {
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %out_qubits:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %4 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %out_qubits#2 : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    // CHECK-NOT: libc++abi
    // CHECK-NOT: uncaught exception
    // CHECK: Decomposition rule not found for operator 'Toffoli[w:3][p:0]'
    // CHECK-NOT: libc++abi
    // CHECK-NOT: uncaught exception
    return
}
