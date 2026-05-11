OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Square root of X gate
sx q[0];
sx q[1];

measure q -> c;
