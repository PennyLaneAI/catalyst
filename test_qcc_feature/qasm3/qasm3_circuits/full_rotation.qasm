OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Full 2π rotations (should return to original state)
rx(6.28318) q[0];
ry(6.28318) q[1];
rz(6.28318) q[2];

measure q -> c;
