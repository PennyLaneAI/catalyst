OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Prepare |+⟩ states
h q[0];
h q[1];
h q[2];

measure q -> c;
