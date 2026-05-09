// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test two-qubit gates

func.func @test_cnot() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: cx q{{.*}}, q
    %q0_out, %q1_out = quantum.custom "cnot"() %q0, %q1 : !quantum.bit, !quantum.bit

    return
}

func.func @test_swap() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: swap q{{.*}}, q
    %q0_out, %q1_out = quantum.custom "swap"() %q0, %q1 : !quantum.bit, !quantum.bit

    return
}

func.func @test_cz() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: cz q{{.*}}, q
    %q0_out, %q1_out = quantum.custom "cz"() %q0, %q1 : !quantum.bit, !quantum.bit

    return
}
