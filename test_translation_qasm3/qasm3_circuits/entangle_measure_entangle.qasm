OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];

// Entangle, measure, entangle again
h q[0];
cx q[0], q[1];
measure q[0] -> c0[0];

h q[1];
cx q[1], q[2];
measure q[1] -> c1[0];
