OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Echoed cross-resonance gate (decomposed using standard gates)
ry(1.5708) q[0];
rx(1.5708) q[1];
cx q[0], q[1];
rx(-1.5708) q[1];

measure q -> c;
