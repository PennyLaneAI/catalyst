OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// GHZ state: (|000⟩ + |111⟩)/√2
h q[0];
cx q[0], q[1];
cx q[1], q[2];

measure q -> c;
