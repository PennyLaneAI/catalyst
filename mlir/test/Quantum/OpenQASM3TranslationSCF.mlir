
// RUN: quantum-translate %s --mlir-to-qasm3 | FileCheck %s

module {
  func.func @test_scf() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    
    %r = quantum.alloc( %c1_i64 ) : !quantum.reg
    %q0 = quantum.extract %r[ %c0_i64 ] : !quantum.reg -> !quantum.bit

    scf.for %i = %c0 to %c10 step %c1 {
        %q0_h = quantum.custom "h"() %q0 : !quantum.bit
    }
    
    quantum.dealloc %r : !quantum.reg
    return
  }
}

// CHECK: OPENQASM 3.0;
// CHECK: qubit [[REG:.*]][1];
// CHECK: for [[VAR:.*]] in [0:1:10] {
// CHECK: h [[REG]][0];
// CHECK: }
