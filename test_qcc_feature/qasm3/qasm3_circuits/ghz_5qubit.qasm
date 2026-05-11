OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

// GHZ state: (|00000⟩ + |11111⟩)/√2
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];

measure q -> c;
