OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Grover's algorithm for 2 qubits (1 iteration)
// Initialize
h q[0];
h q[1];

// Oracle (mark |11⟩)
cz q[0], q[1];

// Diffusion operator
h q[0];
h q[1];
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];
h q[0];
h q[1];

measure q -> c;
