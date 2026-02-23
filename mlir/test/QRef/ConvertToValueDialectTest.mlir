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

// Test conversion to value semantics quantum dialect.
//
// RUN: quantum-opt --convert-to-value-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_expval_circuit
func.func @test_expval_circuit(%arg0: f64, %arg1: f64, %arg2: i1, %arg3: i64) -> f64 {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][%arg3] : !quantum.reg -> !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[%arg3] : !qref.reg<2>, i64 -> !qref.bit

    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[RX:%.+]] = quantum.custom "RX"(%arg0) [[CNOT]]#0 : !quantum.bit
    qref.custom "RX"(%arg0) %q0 : !qref.bit

    // CHECK: [[ROT:%.+]], [[ROTctrl:%.+]] = quantum.custom "Rot"(%arg0, %arg1) [[RX]] adj ctrls([[CNOT]]#1) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.custom "Rot"(%arg0, %arg1) %q0 adj ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: [[PPR:%.+]], [[PPRctrl:%.+]] = quantum.paulirot ["Y"](%arg0) [[ROT]] ctrls([[ROTctrl]]) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.paulirot ["Y"](%arg0) %q0 ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: quantum.gphase(%arg0)
    qref.gphase(%arg0) : f64

    // CHECK: [[GPHASEctrl:%.+]] = quantum.gphase(%arg0) ctrls([[PPR]]) ctrlvals(%arg2) : ctrls !quantum.bit
    qref.gphase(%arg0) ctrls (%q0) ctrlvals (%arg2) : f64 ctrls !qref.bit

    // CHECK: [[namedobs:%.+]] = quantum.namedobs [[PPRctrl]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[namedobs]]
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert1:%.+]] = quantum.insert [[qreg]][%arg3], [[PPRctrl]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert0:%.+]] = quantum.insert [[insert1]][ 0], [[GPHASEctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return %expval : f64
}

// -----
