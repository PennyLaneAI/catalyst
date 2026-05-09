// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test qubit declarations

func.func @test_single_qubit() {
    // CHECK: qubit
    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    return
}

func.func @test_multiple_qubits() {
    // CHECK: qubit
    %c5 = arith.constant 5 : i64
    %reg = quantum.alloc(5) : !quantum.reg

    return
}
