// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test that output contains required headers

func.func @test_headers() {
    // CHECK: OPENQASM 3.0
    // CHECK: include "stdgates.inc"

    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    %q1 = quantum.custom "h"() %q0 : !quantum.bit

    return
}
