// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test control flow structures

func.func @test_if_statement() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    %q0_h = quantum.custom "h"() %q0 : !quantum.bit
    %m, %q0_m = quantum.measure %q0_h : i1, !quantum.bit

    // CHECK: if
    %q1_out = scf.if %m -> !quantum.bit {
        %q1_x = quantum.custom "x"() %q1 : !quantum.bit
        scf.yield %q1_x : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }

    return
}

func.func @test_for_loop() {
    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    %lb = arith.constant 0 : index
    %ub = arith.constant 5 : index
    %step = arith.constant 1 : index

    // CHECK: for
    %q_final = scf.for %i = %lb to %ub step %step iter_args(%q_iter = %q0) -> !quantum.bit {
        %q_next = quantum.custom "h"() %q_iter : !quantum.bit
        scf.yield %q_next : !quantum.bit
    }

    return
}
