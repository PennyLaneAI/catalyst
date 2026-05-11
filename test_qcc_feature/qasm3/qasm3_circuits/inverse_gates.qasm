OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Gates followed by their inverses
h q[0];
h q[0];

s q[1];
sdg q[1];

t q[2];
tdg q[2];

x q[3];
x q[3];

measure q -> c;
