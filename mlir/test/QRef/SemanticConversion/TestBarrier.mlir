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

// RUN: quantum-opt --convert-to-value-semantics --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @all_registers
func.func @all_registers(%n: i64) attributes {quantum.node} {
    // CHECK: [[A:%.+]] = quantum.alloc
    %a = qref.alloc(2) : !qref.reg<2>
    // CHECK: [[B:%.+]] = quantum.alloc
    %b = qref.alloc(%n) : !qref.reg<?>
    // CHECK: [[BARRIER:%.+]]:2 = quantum.barrier [[A]], [[B]] : !quantum.reg, !quantum.reg
    qref.barrier :
    // CHECK: quantum.dealloc [[BARRIER]]#1 : !quantum.reg
    qref.dealloc %b : !qref.reg<?>
    // CHECK: [[A2:%.+]] = quantum.barrier [[BARRIER]]#0 : !quantum.reg
    qref.barrier :
    // CHECK: quantum.dealloc [[A2]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}

// -----

// CHECK-LABEL: @explicit_wire
func.func @explicit_wire() attributes {quantum.node} {
    %a = qref.alloc(2) : !qref.reg<2>
    %b = qref.alloc(1) : !qref.reg<1>
    %q = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    // CHECK: quantum.barrier {{%.+}} : !quantum.bit
    qref.barrier %q : !qref.bit
    qref.dealloc %a : !qref.reg<2>
    qref.dealloc %b : !qref.reg<1>
    return
}

// -----

// CHECK-LABEL: @nested_barrier
func.func @nested_barrier(%condition: i1) attributes {quantum.node} {
    %a = qref.alloc(2) : !qref.reg<2>
    // CHECK: [[RESULT:%.+]] = scf.if
    scf.if %condition {
        %b = qref.alloc(1) : !qref.reg<1>
        // CHECK: quantum.barrier {{%.+}}, {{%.+}} : !quantum.reg, !quantum.reg
        qref.barrier :
        qref.dealloc %b : !qref.reg<1>
    }
    // CHECK: quantum.barrier [[RESULT]] : !quantum.reg
    qref.barrier :
    qref.dealloc %a : !qref.reg<2>
    return
}

// -----

// CHECK-LABEL: @no_registers
func.func @no_registers() attributes {quantum.node} {
    // CHECK: quantum.barrier :
    qref.barrier :
    return
}

// -----

// CHECK-LABEL: @loop_barrier
func.func @loop_barrier() attributes {quantum.node} {
    %a = qref.alloc(2) : !qref.reg<2>
    %start = arith.constant 0 : index
    %stop = arith.constant 3 : index
    %step = arith.constant 1 : index
    // CHECK: [[RESULT:%.+]] = scf.for {{.*}} iter_args([[ARG:%.+]] = {{%.+}}) -> (!quantum.reg)
    scf.for %i = %start to %stop step %step {
        // CHECK: [[NEXT:%.+]] = quantum.barrier [[ARG]] : !quantum.reg
        qref.barrier :
        // CHECK: scf.yield [[NEXT]] : !quantum.reg
    }
    // CHECK: quantum.dealloc [[RESULT]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}
