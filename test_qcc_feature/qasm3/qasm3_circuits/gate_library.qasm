OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

h q[0];
x q[1];
y q[0];
z q[1];
s q[0];
t q[1];

// Parameterized gates
rx(0.5) q[0];
ry(1.2) q[1];
rz(3.14159) q[0];

cx q[0], q[1];
measure q -> c;
