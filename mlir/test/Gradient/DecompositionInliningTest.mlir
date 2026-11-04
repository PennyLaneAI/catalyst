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

// RUN: quantum-opt %s --lower-gradients --split-input-file | FileCheck %s --implicit-check-not='call @decomposition' --implicit-check-not='call @rotation'

// Inline nested decomposition calls inside the QNode, retaining calls to the QNode itself.
// CHECK-LABEL: func.func @circuit(
// CHECK: quantum.custom "RX"
// CHECK: quantum.expval
// CHECK: return
func.func @circuit(%angle: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %reg = quantum.alloc(1) : !quantum.reg
    %out = func.call @decomposition(%reg, %angle) : (!quantum.reg, f64) -> !quantum.reg
    %qubit = quantum.extract %out[0] : !quantum.reg -> !quantum.bit
    %obs = quantum.namedobs %qubit[PauliZ] : !quantum.obs
    %result = quantum.expval %obs : f64
    quantum.dealloc %out : !quantum.reg
    return %result : f64
}

// CHECK-LABEL: func.func private @circuit.pcount(
// CHECK-NOT: quantum.
// CHECK: return
func.func private @decomposition(%reg: !quantum.reg, %angle: f64) -> !quantum.reg {
    %out = func.call @rotation(%reg, %angle) : (!quantum.reg, f64) -> !quantum.reg
    return %out : !quantum.reg
}

func.func private @rotation(%reg: !quantum.reg, %angle: f64) -> !quantum.reg {
    %qubit = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "RX"(%angle) %qubit : !quantum.bit
    %outreg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    return %outreg : !quantum.reg
}

// CHECK-LABEL: func.func @workflow(
// CHECK: call @circuit(
func.func @workflow(%angle: f64) -> f64 {
    %result = func.call @circuit(%angle) : (f64) -> f64
    return %result : f64
}

func.func @gradient(%angle: f64) -> f64 {
    %result = gradient.grad "auto" @workflow(%angle) : (f64) -> f64
    return %result : f64
}

// -----

// CHECK-LABEL: func.func @circuit(
// CHECK: quantum.custom "RX"
// CHECK: quantum.expval
// CHECK: return
func.func @circuit(%angle: f64) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %reg = quantum.alloc(1) : !quantum.reg
    %out = func.call @decomposition(%reg, %angle) : (!quantum.reg, f64) -> !quantum.reg
    %qubit = quantum.extract %out[0] : !quantum.reg -> !quantum.bit
    %obs = quantum.namedobs %qubit[PauliZ] : !quantum.obs
    %result = quantum.expval %obs : f64
    quantum.dealloc %out : !quantum.reg
    return %result : f64
}

// CHECK-LABEL: func.func private @circuit.pcount(
// CHECK-NOT: quantum.
// CHECK: return
func.func private @decomposition(%reg: !quantum.reg, %angle: f64) -> !quantum.reg {
    %out = func.call @rotation(%reg, %angle) : (!quantum.reg, f64) -> !quantum.reg
    return %out : !quantum.reg
}

func.func private @rotation(%reg: !quantum.reg, %angle: f64) -> !quantum.reg {
    %qubit = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "RX"(%angle) %qubit : !quantum.bit
    %outreg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    return %outreg : !quantum.reg
}

// CHECK-LABEL: func.func @workflow(
// CHECK: call @circuit(
func.func @workflow(%angle: f64) -> f64 {
    %result = func.call @circuit(%angle) : (f64) -> f64
    return %result : f64
}

func.func @gradient(%angle: f64) -> f64 {
    %result = gradient.grad "auto" @workflow(%angle) : (f64) -> f64
    return %result : f64
}
