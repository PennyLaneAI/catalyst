OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Test phase gates
s q[0];
t q[1];
sdg q[2];
tdg q[3];

measure q -> c;
