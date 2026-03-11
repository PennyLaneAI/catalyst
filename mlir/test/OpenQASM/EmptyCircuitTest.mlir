// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test empty circuit (minimal case)

func.func @test_empty() {
    // CHECK: OPENQASM 3.0
    // CHECK: include "stdgates.inc"

    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    return
}
