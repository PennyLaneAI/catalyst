OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// GHZ state: (|0000⟩ + |1111⟩)/√2
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];

measure q -> c;
