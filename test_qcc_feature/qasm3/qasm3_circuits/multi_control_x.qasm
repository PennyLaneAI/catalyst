OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Multiple control qubits
x q[0];
x q[1];
x q[2];
ccx q[0], q[1], q[2];
ccx q[0], q[2], q[3];

measure q -> c;
