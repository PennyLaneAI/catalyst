OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Test rotation gates with various angles
rx(0.785398) q[0];  // pi/4
ry(2.35619) q[1];   // 3*pi/4
rz(6.28318) q[2];   // 2*pi
rx(1.0) q[3];       // arbitrary angle

measure q -> c;
