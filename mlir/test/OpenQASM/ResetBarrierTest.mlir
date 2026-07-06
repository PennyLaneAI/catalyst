// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s
// RUN: quantum-opt --pass-pipeline="builtin.module(apply-transform-sequence,canonicalize,merge-rotations)" %s | quantum-translate --mlir-to-qasm3 | FileCheck %s

// Test reset and barrier statements. Both are represented as quantum.custom
// markers by the importer; the second RUN line locks in that they survive the
// standard quantum-opt pipeline as long as their results are consumed.

func.func @main() {
    %c2 = arith.constant 2 : i64
    %reg = quantum.alloc(2) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit

    // CHECK: h q0[0];
    %q0_h = quantum.custom "h"() %q0 : !quantum.bit

    // CHECK-NEXT: reset q0[0];
    %q0_r = quantum.custom "reset"() %q0_h : !quantum.bit

    // CHECK-NEXT: barrier q0[0], q0[1];
    %q0_b, %q1_b = quantum.custom "barrier"() %q0_r, %q1 : !quantum.bit, !quantum.bit

    // CHECK-NEXT: x q0[0];
    %q0_x = quantum.custom "x"() %q0_b : !quantum.bit

    // CHECK: measure q0[0];
    %m, %q0_m = quantum.measure %q0_x : i1, !quantum.bit

    return
}
