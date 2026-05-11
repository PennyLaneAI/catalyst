OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Mix of controlled gates
x q[0];
x q[1];
cx q[0], q[2];
cz q[1], q[3];
ccx q[0], q[1], q[2];

measure q -> c;
