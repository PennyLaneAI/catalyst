OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Identity gates
id q[0];
id q[1];

measure q -> c;
