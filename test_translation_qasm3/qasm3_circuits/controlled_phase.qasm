OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Controlled phase gate
h q[0];
h q[1];
cp(0.7854) q[0], q[1];  // pi/4

measure q -> c;
