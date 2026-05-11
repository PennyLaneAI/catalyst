OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Superdense coding (transmit "11")
// Create Bell pair
h q[0];
cx q[0], q[1];

// Encode "11"
z q[0];
x q[0];

// Decode
cx q[0], q[1];
h q[0];

measure q -> c;
