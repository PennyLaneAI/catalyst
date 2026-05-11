OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// All Clifford generators
h q[0];
s q[1];
cx q[0], q[1];
h q[2];
s q[3];
cx q[2], q[3];

measure q -> c;
