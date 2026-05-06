
// RUN: quantum-translate %s --mlir-to-qasm3 | FileCheck %s

module {
  func.func @test_circuit() {
    %c2 = arith.constant 2 : i64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    
    %r = quantum.alloc(%c2) : !quantum.reg
    %q0 = quantum.extract %r[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[%c1] : !quantum.reg -> !quantum.bit
    
    %q0_h = quantum.custom "h"() %q0 : !quantum.bit
    %q0_cx, %q1_cx = quantum.custom "cx"() %q0_h, %q1 : !quantum.bit, !quantum.bit
    
    quantum.dealloc %r : !quantum.reg
    return
  }
}

// CHECK: OPENQASM 3.0;
// CHECK: qubit[2] [[REG:[a-z][a-z0-9]*]];
// CHECK: h [[REG]][0];
// CHECK: cx [[REG]][0], [[REG]][1];
