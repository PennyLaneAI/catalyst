// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test measurement operations

func.func @test_single_measurement() {
    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    %q1 = quantum.custom "h"() %q0 : !quantum.bit

    // CHECK: measure q
    %m, %q_out = quantum.measure %q1 : i1, !quantum.bit

    return
}

func.func @test_multiple_measurements() {
    %c3 = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%c2] : !quantum.reg -> !quantum.bit

    %q0_h = quantum.custom "h"() %q0 : !quantum.bit
    %q1_h = quantum.custom "h"() %q1 : !quantum.bit
    %q2_h = quantum.custom "h"() %q2 : !quantum.bit

    // CHECK: measure q
    // CHECK: measure q
    // CHECK: measure q
    %m0, %q0_out = quantum.measure %q0_h : i1, !quantum.bit
    %m1, %q1_out = quantum.measure %q1_h : i1, !quantum.bit
    %m2, %q2_out = quantum.measure %q2_h : i1, !quantum.bit

    return
}
