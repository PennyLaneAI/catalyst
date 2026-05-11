OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Test all Pauli gates
x q[0];
y q[1];
z q[2];

measure q -> c;
