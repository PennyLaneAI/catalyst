// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test basic single-qubit gates

func.func @test_hadamard() {
    %c5 = arith.constant 5 : i64
    %reg = quantum.alloc(5) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    // CHECK: h q
    %q1 = quantum.custom "h"() %q0 : !quantum.bit

    return
}

func.func @test_pauli_gates() {
    %c3 = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%c2] : !quantum.reg -> !quantum.bit

    // CHECK: x q
    %q0_out = quantum.custom "x"() %q0 : !quantum.bit

    // CHECK: y q
    %q1_out = quantum.custom "y"() %q1 : !quantum.bit

    // CHECK: z q
    %q2_out = quantum.custom "z"() %q2 : !quantum.bit

    return
}

func.func @test_phase_gates() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: s q
    %q0_out = quantum.custom "s"() %q0 : !quantum.bit

    // CHECK: t q
    %q1_out = quantum.custom "t"() %q1 : !quantum.bit

    return
}
