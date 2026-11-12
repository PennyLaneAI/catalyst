// RUN: quantum-opt %s | FileCheck %s --check-prefixes=ALL,VALUE
// RUN: quantum-opt --convert-to-reference-semantics %s | FileCheck %s --check-prefixes=ALL,REF
// RUN: quantum-opt --convert-to-reference-semantics --convert-to-value-semantics %s | FileCheck %s --check-prefixes=ALL,VALUE
// RUN: quantum-opt --inline-nested-module --convert-quantum-to-llvm %s | FileCheck %s --check-prefixes=ALL,LLVM

// ALL-LABEL: @barrier_registers
// LLVM-NOT: Barrier
// LLVM-NOT: barrier
// LLVM: llvm.return
func.func @barrier_registers(%n: i64) attributes {quantum.node} {
    %a = quantum.alloc(2) : !quantum.reg
    %b = quantum.alloc(%n) : !quantum.reg
    // VALUE: quantum.barrier {{.*}} : !quantum.reg, !quantum.reg
    // REF: qref.barrier {{.*}} : !qref.reg<2>, !qref.reg<?>
    %a1, %b1 = quantum.barrier %a, %b : !quantum.reg, !quantum.reg
    quantum.dealloc %a1 : !quantum.reg
    quantum.dealloc %b1 : !quantum.reg
    return
}

// ALL-LABEL: @barrier_qubit_and_register
// LLVM-NOT: Barrier
// LLVM-NOT: barrier
// LLVM: llvm.return
func.func @barrier_qubit_and_register() attributes {quantum.node} {
    %a = quantum.alloc(1) : !quantum.reg
    %b = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %a[0] : !quantum.reg -> !quantum.bit
    // VALUE: quantum.barrier {{.*}} : !quantum.bit
    // REF: qref.barrier {{.*}} : !qref.bit
    %q1 = quantum.barrier %q : !quantum.bit
    // VALUE: quantum.barrier {{.*}} : !quantum.bit, !quantum.reg
    // REF: qref.barrier {{.*}} : !qref.bit, !qref.reg<2>
    %q2, %b1 = quantum.barrier %q1, %b : !quantum.bit, !quantum.reg
    %a1 = quantum.insert %a[0], %q2 : !quantum.reg, !quantum.bit
    quantum.dealloc %a1 : !quantum.reg
    quantum.dealloc %b1 : !quantum.reg
    return
}
