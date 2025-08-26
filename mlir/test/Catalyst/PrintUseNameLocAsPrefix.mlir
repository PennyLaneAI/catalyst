// RUN: quantum-opt %s -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s

func.func @test_alice() {
    %0 = quantum.alloc( 1) : !quantum.reg
    // CHECK: %alice = quantum.extract
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit loc("alice")
    quantum.dealloc %0 : !quantum.reg
    return
}

// -----

func.func @test_alice_and_bob() {
    %0 = quantum.alloc( 2) : !quantum.reg
    // CHECK: %alice = quantum.extract
    // CHECK: %bob = quantum.extract
    // CHECK: quantum.custom "CNOT"() %alice, %bob
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit loc("alice")
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit loc("bob")
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    quantum.dealloc %0 : !quantum.reg
    return
}
