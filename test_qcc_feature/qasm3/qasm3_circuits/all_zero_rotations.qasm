OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Zero-angle rotations (should be identity)
rx(0.0) q[0];
ry(0.0) q[1];
rz(0.0) q[2];

measure q -> c;
