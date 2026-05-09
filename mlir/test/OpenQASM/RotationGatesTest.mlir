// RUN: quantum-translate --mlir-to-qasm3 %s | FileCheck %s

// Test parameterized rotation gates

func.func @test_rotation_gates() {
    %c3 = arith.constant 3 : i64
    %reg = quantum.alloc(3) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64

    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%c1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[%c2] : !quantum.reg -> !quantum.bit

    %angle1 = arith.constant 0.5 : f64
    %angle2 = arith.constant 1.2 : f64
    %angle3 = arith.constant 3.14159 : f64

    // CHECK: rx({{.*}}) q
    %q0_out = quantum.custom "rx"(%angle1) %q0 : !quantum.bit

    // CHECK: ry({{.*}}) q
    %q1_out = quantum.custom "ry"(%angle2) %q1 : !quantum.bit

    // CHECK: rz({{.*}}) q
    %q2_out = quantum.custom "rz"(%angle3) %q2 : !quantum.bit

    return
}

func.func @test_zero_angle_rotations() {
    %c1 = arith.constant 1 : i64
    %reg = quantum.alloc(1) : !quantum.reg

    %c0 = arith.constant 0 : i64
    %q0 = quantum.extract %reg[%c0] : !quantum.reg -> !quantum.bit

    %zero = arith.constant 0.0 : f64

    // CHECK: rx(0
    %q_out = quantum.custom "rx"(%zero) %q0 : !quantum.bit

    return
}
