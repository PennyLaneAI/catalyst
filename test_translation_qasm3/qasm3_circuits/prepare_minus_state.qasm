OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Prepare |-⟩ states
x q[0];
h q[0];
x q[1];
h q[1];
x q[2];
h q[2];

measure q -> c;
