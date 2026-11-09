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

// RUN: quantum-opt %s --delegate-feedforward --split-input-file --verify-diagnostics | FileCheck %s

// The pauli-corrected pi/8 gadget: a measurement-dependent basis, then a conditional Pauli.

// CHECK-LABEL: @pauli_corrected
func.func @pauli_corrected(%q: !quantum.bit) -> !quantum.bit {
    %shots = arith.constant 1000 : i64
    // CHECK: quantum.device
    // CHECK-NEXT: quantum.delegate fill_shots
    quantum.device shots(%shots) ["librtd_null_qubit.so", "NullQubit", "{}"]
    %magic = pbc.fabricate magic : !quantum.bit
    // CHECK: [[m0:%.+]], [[o:%.+]]:2 = pbc.ppm ["Z", "Z"] {{.*}} : !quantum.res, !quantum.bit, !quantum.bit
    %m0, %o:2 = pbc.ppm ["Z", "Z"] %q, %magic : i1, !quantum.bit, !quantum.bit
    // CHECK: [[sel:%.+]]:2 = quantum.predicated [[m0]]([[o]]#1) : (!quantum.bit) -> (!quantum.bit, !quantum.res) {
    // CHECK:   pbc.ppm ["Y"] {{.*}} : !quantum.res, !quantum.bit
    // CHECK: } else {
    // CHECK:   pbc.ppm ["X"] {{.*}} : !quantum.res, !quantum.bit
    // CHECK: }
    // CHECK: quantum.result.release [[m0]]
    %m1, %a = pbc.select.ppm (%m0 ? ["Y"] : ["X"]) %o#1 : i1, !quantum.bit
    // CHECK: [[corrected:%.+]] = quantum.predicated [[sel]]#1([[o]]#0) : (!quantum.bit) -> (!quantum.bit) {
    // CHECK:   pbc.ppr ["Z"](2) %{{.+}} : !quantum.bit
    // CHECK-NOT: cond
    // CHECK: } else {
    // CHECK:   quantum.yield
    // CHECK: }
    // CHECK: quantum.result.release [[sel]]#1
    %out = pbc.ppr ["Z"](2) %o#0 cond(%m1) : !quantum.bit
    quantum.dealloc_qb %a : !quantum.bit
    quantum.device_release
    return %out : !quantum.bit
}

// -----

// A branch that releases an auxiliary qubit in both arms returns it instead, and the qubit is
// released once after the predicated op. i1 arithmetic on outcomes becomes result ops.

// CHECK-LABEL: @branch_with_release
func.func @branch_with_release(%q: !quantum.bit) -> !quantum.bit {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    %aux = quantum.alloc_qb : !quantum.bit
    %m0, %o:2 = pbc.ppm ["Z", "Z"] %q, %aux : i1, !quantum.bit, !quantum.bit
    %true = arith.constant true
    // CHECK: [[not:%.+]] = quantum.result.not
    %n = arith.xori %m0, %true : i1
    // CHECK: [[out:%.+]]:3 = quantum.predicated [[not]]({{.*}}) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.res) {
    // CHECK-NOT: quantum.dealloc_qb
    // CHECK: } else {
    // CHECK-NOT: quantum.dealloc_qb
    // CHECK: }
    // CHECK: quantum.dealloc_qb [[out]]#1
    %r:2 = scf.if %n -> (!quantum.bit, i1) {
        %m1, %a = pbc.ppm ["X"] %o#1 : i1, !quantum.bit
        %c = pbc.ppr ["X"](2) %o#0 cond(%m1) : !quantum.bit
        quantum.dealloc_qb %a : !quantum.bit
        scf.yield %c, %m1 : !quantum.bit, i1
    } else {
        %false = arith.constant false
        quantum.dealloc_qb %o#1 : !quantum.bit
        scf.yield %o#0, %false : !quantum.bit, i1
    }
    // CHECK: [[and:%.+]] = quantum.result.select [[out]]#2, %{{.+}}, %{{.+}}
    // CHECK: quantum.predicated [[and]]
    %both = arith.andi %r#1, %m0 : i1
    %final = pbc.ppr ["Z"](2) %r#0 cond(%both) : !quantum.bit
    return %final : !quantum.bit
}

// -----

// A user-level qp.cond on a mid-circuit measurement.

// CHECK-LABEL: @user_cond
func.func @user_cond(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    // CHECK: quantum.measure {{.*}} : !quantum.res, !quantum.bit
    %m, %a = quantum.measure %q0 : i1, !quantum.bit
    // CHECK: quantum.predicated
    // CHECK: quantum.custom "PauliX"
    %b = scf.if %m -> (!quantum.bit) {
        %x = quantum.custom "PauliX"() %q1 : !quantum.bit
        scf.yield %x : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    return %a, %b : !quantum.bit, !quantum.bit
}

// -----

func.func @returned_outcome(%q: !quantum.bit) -> i1 {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    %m, %a = pbc.ppm ["Z"] %q : i1, !quantum.bit
    // expected-error@+1 {{a measurement outcome is used as a classical value}}
    return %m : i1
}

// -----

func.func @classical_branch_result(%q: !quantum.bit) -> f64 {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    %m, %a = pbc.ppm ["Z"] %q : i1, !quantum.bit
    // expected-error@+1 {{a conditional branch returns a classical value}}
    %r = scf.if %m -> (f64) {
        %one = arith.constant 1.0 : f64
        scf.yield %one : f64
    } else {
        %zero = arith.constant 0.0 : f64
        scf.yield %zero : f64
    }
    return %r : f64
}

// -----

func.func @register_in_branch(%q: !quantum.bit, %reg: !quantum.reg) -> !quantum.reg {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    %m, %a = pbc.ppm ["Z"] %q : i1, !quantum.bit
    %r = scf.if %m -> (!quantum.reg) {
        // expected-error@+1 {{this op cannot run on only some of the shots}}
        %b = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
        %x = quantum.custom "PauliX"() %b : !quantum.bit
        %reg2 = quantum.insert %reg[0], %x : !quantum.reg, !quantum.bit
        scf.yield %reg2 : !quantum.reg
    } else {
        scf.yield %reg : !quantum.reg
    }
    return %r : !quantum.reg
}

// -----

func.func @released_in_one_branch(%q: !quantum.bit, %aux: !quantum.bit) -> !quantum.bit {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"]
    %m, %a = pbc.ppm ["Z"] %q : i1, !quantum.bit
    // expected-error@+1 {{a qubit leaves the two branches of a conditional differently}}
    %r = scf.if %m -> (!quantum.bit) {
        quantum.dealloc_qb %aux : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        scf.yield %a : !quantum.bit
    }
    return %r : !quantum.bit
}

// -----

// A function without a device is not a quantum program, and is left alone.

// CHECK-LABEL: @no_device
func.func @no_device(%m: i1, %q: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: quantum.delegate
    // CHECK: pbc.ppr ["Z"](2) {{.*}} cond
    %out = pbc.ppr ["Z"](2) %q cond(%m) : !quantum.bit
    return %out : !quantum.bit
}
