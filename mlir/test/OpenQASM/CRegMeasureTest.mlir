// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s
// RUN: quantum-opt --pass-pipeline="builtin.module(apply-transform-sequence,canonicalize,merge-rotations)" %s | quantum-translate --mlir-to-qasm3 | FileCheck %s

// Test classical register declarations and measurement assignment.
// Measurements tagged with creg_name/creg_idx/creg_size attributes assign into
// a pre-declared bit[n] register; untagged measurements fall back to an
// anonymous bit. The second RUN line locks in that the attributes survive the
// standard quantum-opt pipeline.

// CHECK: bit[3] c;
func.func @main() {
    %c3 = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%c2] : !quantum.reg -> !quantum.bit

    %q0_h = quantum.custom "h"() %q0 : !quantum.bit

    // CHECK: c[0] = measure q0[0];
    %m0, %q0_m = quantum.measure %q0_h {creg_idx = 0 : i64, creg_name = "c", creg_size = 3 : i64} : i1, !quantum.bit

    // CHECK: c[1] = measure q0[1];
    %m1, %q1_m = quantum.measure %q1 {creg_idx = 1 : i64, creg_name = "c", creg_size = 3 : i64} : i1, !quantum.bit

    // Untagged measurement: anonymous bit fallback.
    // CHECK: bit m_
    // CHECK: = measure q0[2];
    %m2, %q2_m = quantum.measure %q2 : i1, !quantum.bit

    return
}
