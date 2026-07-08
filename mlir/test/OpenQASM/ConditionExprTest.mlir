// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test feedforward condition expressions: negation (xori), conjunction
// (andi), disjunction (ori), and integer comparison (cmpi eq/ne).

func.func @main() {
    %cn = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %i0 = arith.constant 0 : i64
    %i1 = arith.constant 1 : i64
    %i2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%i0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%i1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%i2] : !quantum.reg -> !quantum.bit

    // CHECK: c[0] = measure q0[0];
    %m0, %q0_m = quantum.measure %q0 {creg_idx = 0 : i64, creg_name = "c", creg_size = 2 : i64} : i1, !quantum.bit
    // CHECK: c[1] = measure q0[1];
    %m1, %q1_m = quantum.measure %q1 {creg_idx = 1 : i64, creg_name = "c", creg_size = 2 : i64} : i1, !quantum.bit

    %true = arith.constant true

    // A conjunction tagged by the importer as a register-equality fold
    // (qasm3_creg_eq) is reconstructed as a whole-register comparison.
    // CHECK: if (c == 2) {
    %n0 = arith.xori %m0, %true : i1
    %and = arith.andi %n0, %m1 {qasm3_creg_eq} : i1
    %q2_a = scf.if %and -> !quantum.bit {
        %q2_x = quantum.custom "x"() %q2 : !quantum.bit
        scf.yield %q2_x : !quantum.bit
    } else {
        scf.yield %q2 : !quantum.bit
    }

    // An UNTAGGED conjunction is a generic logic AND: it must NOT become a
    // register comparison (it says nothing about unmentioned bits).
    // CHECK: if ((c[0] && c[1])) {
    %and2 = arith.andi %m0, %m1 : i1
    %q2_a2 = scf.if %and2 -> !quantum.bit {
        %q2_y = quantum.custom "y"() %q2_a : !quantum.bit
        scf.yield %q2_y : !quantum.bit
    } else {
        scf.yield %q2_a : !quantum.bit
    }

    // CHECK: if ((c[0] || c[1])) {
    %or = arith.ori %m0, %m1 : i1
    %q2_b = scf.if %or -> !quantum.bit {
        %q2_z = quantum.custom "z"() %q2_a2 : !quantum.bit
        scf.yield %q2_z : !quantum.bit
    } else {
        scf.yield %q2_a2 : !quantum.bit
    }

    // CHECK: if (c[0] == 1) {
    %cmp = arith.cmpi eq, %m0, %true : i1
    %q2_c = scf.if %cmp -> !quantum.bit {
        %q2_h = quantum.custom "h"() %q2_b : !quantum.bit
        scf.yield %q2_h : !quantum.bit
    } else {
        scf.yield %q2_b : !quantum.bit
    }

    // CHECK: measure q0[2];
    %m2, %q2_m = quantum.measure %q2_c : i1, !quantum.bit

    return
}
