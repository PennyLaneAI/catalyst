// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test scf.while → QASM3 while translation. The loop-carried condition bit is
// named via creg attributes so the in-body measurement re-assigns the same
// c[0] that the while condition reads — QASM3's named-variable semantics
// implement the loop-carry that SSA does in MLIR.

// CHECK: bit[1] c;
func.func @main() {
    %cn = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg
    %i0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%i0] : !quantum.reg -> !quantum.bit

    // CHECK: h q0[0];
    %q0_h = quantum.custom "h"() %q0 : !quantum.bit
    // CHECK: c[0] = measure q0[0];
    %m0, %q0_m = quantum.measure %q0_h {creg_idx = 0 : i64, creg_name = "c", creg_size = 1 : i64} : i1, !quantum.bit

    // Full-forwarding form (as emitted by the importer): scf.condition
    // forwards all loop-carried values.
    // CHECK: while (c[0]) {
    // CHECK: h q0[0];
    // CHECK: c[0] = measure q0[0];
    // CHECK: }
    %res:2 = scf.while (%arg0 = %m0, %arg1 = %q0_m) : (i1, !quantum.bit) -> (i1, !quantum.bit) {
        scf.condition(%arg0) %arg0, %arg1 : i1, !quantum.bit
    } do {
    ^bb0(%b0: i1, %b1: !quantum.bit):
        %qh = quantum.custom "h"() %b1 : !quantum.bit
        %m1, %qm = quantum.measure %qh {creg_idx = 0 : i64, creg_name = "c", creg_size = 1 : i64} : i1, !quantum.bit
        scf.yield %m1, %qm : i1, !quantum.bit
    }

    return
}

// Canonicalized-pruned form: canonicalize drops unused forwarded values, so
// scf.condition forwards only the qubit. After-region args must be mapped
// from the condition's forwarded operands, not positionally from the inits.
func.func @pruned() {
    %cn = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg
    %i0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%i0] : !quantum.reg -> !quantum.bit

    %m0, %q0_m = quantum.measure %q0 {creg_idx = 0 : i64, creg_name = "d", creg_size = 1 : i64} : i1, !quantum.bit

    // CHECK: while (d[0]) {
    // CHECK: x q1[0];
    // CHECK: d[0] = measure q1[0];
    // CHECK: }
    %res = scf.while (%arg0 = %m0, %arg1 = %q0_m) : (i1, !quantum.bit) -> (!quantum.bit) {
        scf.condition(%arg0) %arg1 : !quantum.bit
    } do {
    ^bb0(%b1: !quantum.bit):
        %qx = quantum.custom "x"() %b1 : !quantum.bit
        %m1, %qm = quantum.measure %qx {creg_idx = 0 : i64, creg_name = "d", creg_size = 1 : i64} : i1, !quantum.bit
        scf.yield %m1, %qm : i1, !quantum.bit
    }

    return
}
