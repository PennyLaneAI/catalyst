OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Quantum Fourier Transform on 2 qubits
h q[0];
cu1(1.5708) q[1], q[0];  // pi/2
h q[1];
swap q[0], q[1];

measure q -> c;
