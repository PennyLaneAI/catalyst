OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Test two-qubit gates
cx q[0], q[1];
cz q[1], q[2];
swap q[2], q[3];
cy q[0], q[3];

measure q -> c;
