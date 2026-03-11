OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Fredkin (CSWAP) gate test
x q[0];
x q[1];
cswap q[0], q[1], q[2];

measure q -> c;
