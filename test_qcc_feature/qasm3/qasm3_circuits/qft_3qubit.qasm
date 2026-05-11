OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// Quantum Fourier Transform on 3 qubits
h q[0];
cu1(1.5708) q[1], q[0];
cu1(0.785398) q[2], q[0];
h q[1];
cu1(1.5708) q[2], q[1];
h q[2];
swap q[0], q[2];

measure q -> c;
