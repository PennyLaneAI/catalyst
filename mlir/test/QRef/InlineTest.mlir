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


// RUN: quantum-opt --split-input-file --inline %s | FileCheck %s

func.func @my_helper() -> f64 {
    %0 = arith.constant 4.2 : f64
    return %0 : f64
}

// CHECK-LABEL: test_ctrl_op
// CHECK: [[param:%.+]] = arith.constant 4.200000e+00 : f64
// CHECK: qref.ctrl
// CHECK-NOT: func.call
// CHECK: qref.custom "gate"([[param]])
func.func @test_ctrl_op(%q: !qref.bit, %r: !qref.reg<1>)
{
    %true = llvm.mlir.constant (1 : i1) :i1
    qref.ctrl (%q) ctrlvals (%true){
    ^bb0():
        %target = qref.get %r[0] : !qref.reg<1> -> !qref.bit
        %param = func.call @my_helper() : () -> f64
        qref.custom "gate"(%param) %target : !qref.bit
    }
    return
}

// -----

// CHECK-LABEL: @test_unannotated
// CHECK-SAME: ([[Q:%.*]]: !qref.bit)
// CHECK-NEXT: qref.custom "PauliX"() [[Q]]
// CHECK-NEXT: return
func.func @test_unannotated(%q: !qref.bit) {
    func.call @helper(%q) : (!qref.bit) -> ()
    return
}

func.func private @helper(%q: !qref.bit) {
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

// CHECK-LABEL: @test_qnode_attribute
// CHECK-SAME: ([[Q:%.*]]: !qref.bit)
// CHECK-NEXT: qref.custom "PauliX"() [[Q]]
// CHECK-NEXT: return
func.func @test_qnode_attribute(%q: !qref.bit) {
    func.call @helper(%q) : (!qref.bit) -> ()
    return
}

func.func private @helper(%q: !qref.bit) attributes {quantum.node} {
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

// CHECK-LABEL: @test_mixed_dialects
// CHECK-SAME: ([[Q:%.*]]: !qref.bit)
// CHECK-NEXT: qref.custom "PauliX"() [[Q]]
// CHECK-NEXT: [[SIZE:%.*]] = quantum.num_qubits : i64
// CHECK-NEXT: return [[SIZE]]
func.func @test_mixed_dialects(%q: !qref.bit) -> i64 {
    %size = func.call @helper(%q) : (!qref.bit) -> i64
    return %size : i64
}

func.func private @helper(%q: !qref.bit) -> i64 {
    qref.custom "PauliX"() %q : !qref.bit
    %size = quantum.num_qubits : i64
    return %size : i64
}

// -----


func.func @my_helper() -> f64 {
    %0 = arith.constant 4.2 : f64
    return %0 : f64
}

// CHECK-LABEL: test_adjoint_op
// CHECK: [[param:%.+]] = arith.constant 4.200000e+00 : f64
// CHECK: qref.adjoint
// CHECK-NOT: func.call
// CHECK: qref.custom "gate"([[param]])
func.func @test_adjoint_op(%q: !qref.bit)
{
    qref.adjoint{
    ^bb0():
        %param = func.call @my_helper() : () -> f64
        qref.custom "gate"(%param) %q : !qref.bit
    }
    return
}
