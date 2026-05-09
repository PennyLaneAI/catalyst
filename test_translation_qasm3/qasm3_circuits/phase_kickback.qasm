OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Phase kickback demonstration
h q[0];
x q[1];
h q[1];
cx q[0], q[1];
h q[0];

measure q -> c;
