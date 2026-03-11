OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Dense circuit with many gates
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[0];
rx(0.5) q[0];
ry(0.5) q[1];
rz(0.5) q[2];
rx(0.5) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[0];
h q[0];
h q[1];
h q[2];
h q[3];

measure q -> c;
