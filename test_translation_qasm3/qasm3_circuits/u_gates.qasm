OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Universal U gates
u1(0.5) q[0];
u2(0.5, 1.0) q[1];
u3(0.5, 1.0, 1.5) q[2];

measure q -> c;
