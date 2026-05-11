OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[1];

// SWAP test circuit
h q[0];
cswap q[0], q[1], q[2];
h q[0];

measure q[0] -> c[0];
