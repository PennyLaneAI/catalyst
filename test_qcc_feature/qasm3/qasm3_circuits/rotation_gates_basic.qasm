OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Test rotation gates with different angles
rx(0.0) q[0];
ry(1.5708) q[1];  // pi/2
rz(3.14159) q[2]; // pi

measure q -> c;
