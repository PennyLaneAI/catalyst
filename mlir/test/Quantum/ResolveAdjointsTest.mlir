// RUN: quantum-opt --resolve-adjoints %s | FileCheck %s

// CHECK-LABEL: test_hermitian_adjoint
func.func @test_hermitian_adjoint() -> !quantum.bit {
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ret:%.+]] = quantum.custom "Hadamard"() [[qubit]] : !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 {adjoint} : !quantum.bit
    // CHECK: return [[ret]]
    return %2 : !quantum.bit
}

// CHECK-LABEL: test_rotation_adjoint
func.func @test_rotation_adjoint(%arg0: f64) -> !quantum.bit {
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "RX"([[arg0neg]]) [[qubit]] : !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 {adjoint} : !quantum.bit
    // CHECK: return [[ret]]
    return %2 : !quantum.bit
}

// CHECK-LABEL: test_multirz_adjoint
func.func @test_multirz_adjoint(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]]:2 = quantum.multirz([[arg0neg]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    %3:2 = quantum.multirz(%arg0) %1, %2 {adjoint} : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}

// CHECK-LABEL: test_pcphase_adjoint
func.func @test_pcphase_adjoint(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]]:2 = quantum.pcphase([[arg0neg]], dim : 2) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    %3:2 = quantum.pcphase(%arg0, dim : 2) %1, %2 {adjoint} : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}
