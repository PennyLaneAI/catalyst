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

// CHECK-LABEL: @test_unannotated
// CHECK-SAME: ([[Q:%.*]]: !quantum.bit)
// CHECK-NEXT: [[OUT:%.*]] = quantum.custom "PauliX"() [[Q]]
// CHECK-NEXT: return [[OUT]]
func.func @test_unannotated(%q: !quantum.bit) -> !quantum.bit {
    %out = func.call @helper(%q) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_multiple_calls
// CHECK-SAME: ([[Q0:%.*]]: !quantum.bit, [[Q1:%.*]]: !quantum.bit)
// CHECK-NEXT: [[OUT0:%.*]] = quantum.custom "PauliX"() [[Q0]]
// CHECK-NEXT: [[OUT1:%.*]] = quantum.custom "PauliX"() [[Q1]]
// CHECK-NEXT: return [[OUT0]], [[OUT1]]
func.func @test_multiple_calls(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %out0 = func.call @helper(%q0) : (!quantum.bit) -> !quantum.bit
    %out1 = func.call @helper(%q1) : (!quantum.bit) -> !quantum.bit
    return %out0, %out1 : !quantum.bit, !quantum.bit
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_qnode_attribute
// CHECK-SAME: ([[Q:%.*]]: !quantum.bit)
// CHECK-NEXT: [[OUT:%.*]] = quantum.custom "PauliX"() [[Q]]
// CHECK-NEXT: return [[OUT]]
func.func @test_qnode_attribute(%q: !quantum.bit) -> !quantum.bit {
    %out = func.call @helper(%q) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_target_gate_attribute
// CHECK-SAME: ([[Q:%.*]]: !quantum.bit)
// CHECK-NEXT: [[OUT:%.*]] = quantum.custom "PauliX"() [[Q]]
// CHECK-NEXT: return [[OUT]]
func.func @test_target_gate_attribute(%q: !quantum.bit) -> !quantum.bit {
    %out = func.call @helper(%q) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit attributes {target_gate = "PauliX"} {
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_ctrl_region
// CHECK: quantum.ctrl
// CHECK: ^bb0([[Q:%.*]]: !quantum.bit):
// CHECK-NEXT: [[OUT:%.*]] = quantum.custom "PauliX"() [[Q]]
// CHECK-NEXT: quantum.yield [[OUT]]
func.func @test_ctrl_region(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %true = arith.constant true
    %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
    ^bb0(%arg: !quantum.bit):
        %out = func.call @helper(%arg) : (!quantum.bit) -> !quantum.bit
        quantum.yield %out : !quantum.bit
    }
    return %outc, %outq : !quantum.bit, !quantum.bit
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_adjoint_region
// CHECK: quantum.adjoint
// CHECK: ^bb0([[REG:%.*]]: !quantum.reg):
// CHECK-NEXT: [[Q:%.*]] = quantum.extract [[REG]][{{ *}}0]
// CHECK-NEXT: [[OUT:%.*]] = quantum.custom "PauliX"() [[Q]]
// CHECK-NEXT: [[OUTREG:%.*]] = quantum.insert [[REG]][{{ *}}0], [[OUT]]
// CHECK-NEXT: quantum.yield [[OUTREG]]
func.func @test_adjoint_region(%reg: !quantum.reg) -> !quantum.reg {
    %out = quantum.adjoint(%reg) : !quantum.reg {
    ^bb0(%arg: !quantum.reg):
        %result = func.call @helper(%arg) : (!quantum.reg) -> !quantum.reg
        quantum.yield %result : !quantum.reg
    }
    return %out : !quantum.reg
}

func.func private @helper(%reg: !quantum.reg) -> !quantum.reg {
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "PauliX"() %q : !quantum.bit
    %outreg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    return %outreg : !quantum.reg
}
