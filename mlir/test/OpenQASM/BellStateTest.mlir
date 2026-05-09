// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test Bell state preparation

func.func @test_bell_state() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: h q
    %q0_h = quantum.custom "h"() %q0 : !quantum.bit

    // CHECK: cx q{{.*}}, q
    %q0_out, %q1_out = quantum.custom "cnot"() %q0_h, %q1 : !quantum.bit, !quantum.bit

    // CHECK: measure q
    // CHECK: measure q
    %m0, %q0_m = quantum.measure %q0_out : i1, !quantum.bit
    %m1, %q1_m = quantum.measure %q1_out : i1, !quantum.bit

    return
}
