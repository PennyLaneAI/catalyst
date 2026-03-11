OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];

// Sequence of rotations on same qubit
rx(0.1) q[0];
ry(0.2) q[0];
rz(0.3) q[0];
rx(0.4) q[0];
ry(0.5) q[0];
rz(0.6) q[0];

measure q[0] -> c[0];
