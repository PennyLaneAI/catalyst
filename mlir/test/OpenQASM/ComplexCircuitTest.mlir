// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test complex circuit with multiple operations

func.func @test_complex_circuit() {
    %c3 = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%c2] : !quantum.reg -> !quantum.bit

    // CHECK: OPENQASM 3.0
    // CHECK: h q
    %q0_h = quantum.custom "h"() %q0 : !quantum.bit

    // CHECK: cx q{{.*}}, q
    %q0_out1, %q1_out1 = quantum.custom "cnot"() %q0_h, %q1 : !quantum.bit, !quantum.bit

    // CHECK: cx q{{.*}}, q
    %q1_out2, %q2_out1 = quantum.custom "cnot"() %q1_out1, %q2 : !quantum.bit, !quantum.bit

    // CHECK: measure q
    // CHECK: measure q
    // CHECK: measure q
    %m0, %q0_m = quantum.measure %q0_out1 : i1, !quantum.bit
    %m1, %q1_m = quantum.measure %q1_out2 : i1, !quantum.bit
    %m2, %q2_m = quantum.measure %q2_out1 : i1, !quantum.bit

    return
}
