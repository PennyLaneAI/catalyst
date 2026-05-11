OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Double CNOT gate (decomposed)
cx q[0], q[1];
cx q[1], q[0];

measure q -> c;
