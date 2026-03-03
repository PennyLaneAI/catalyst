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


// CHECK-LABEL: test_for_loop
func.func @test_for_loop(%nqubits: i64) -> f64 attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    %a = qref.alloc(%nqubits) : !qref.reg<?>

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<?> -> !qref.bit
    qref.custom "PauliX"() %q0 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    // CHECK: [[preLoopInsert:%.+]] = quantum.insert [[qreg]][ 0], [[X]] : !quantum.reg, !quantum.bit
    // CHECK: [[loopOutReg:%.+]] = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = [[preLoopInsert]]) -> (!quantum.reg) {
    scf.for %i = %start to %stop step %step {
        // CHECK: [[indexCast:%.+]] = index.casts %arg1 : index to i64
        %int = index.casts %i : index to i64

        // CHECK: [[dynamicQ:%.+]] = quantum.extract %arg2[[[indexCast]]] : !quantum.reg -> !quantum.bit
        //%this_q = qref.get %a[%int] : !qref.reg<?>, i64 -> !qref.bit
        %this_q = qref.get %a[1] : !qref.reg<?> -> !qref.bit

        // CHECK: [[dynamicIndexInsert:%.+]] = quantum.insert %arg2[[[indexCast]]], [[dynamicQ]] : !quantum.reg, !quantum.bit
        // CHECK: [[dynamicQ:%.+]] = quantum.extract [[dynamicIndexInsert]][[[indexCast]]] : !quantum.reg -> !quantum.bit
        // CHECK: [[Hadamard:%.+]] = quantum.custom "Hadamard"() [[dynamicQ]] : !quantum.bit
        // CHECK: [[dynamicQInsert:%.+]] = quantum.insert [[dynamicIndexInsert]][[[indexCast]]], [[Hadamard]] : !quantum.reg, !quantum.bit
        qref.custom "Hadamard"() %this_q : !qref.bit

        // CHECK: [[staticQ:%.+]] = quantum.extract [[dynamicQInsert]][ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[Z:%.+]] = quantum.custom "PauliZ"() [[staticQ]] : !quantum.bit
        qref.custom "PauliZ"() %q0 : !qref.bit

        // CHECK: [[staticQInsert:%.+]] = quantum.insert [[dynamicQInsert]][ 0], [[Z]] : !quantum.reg, !quantum.bit
        // CHECK: scf.yield [[staticQInsert]] : !quantum.reg
        scf.yield
    }

    // CHECK: [[extract:%.+]] = quantum.extract [[loopOutReg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs:%.+]] = quantum.namedobs [[extract]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[obs]]
    %obs = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert:%.+]] = quantum.insert [[loopOutReg]][ 0], [[extract]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %a : !qref.reg<?>
    return %expval : f64
}
