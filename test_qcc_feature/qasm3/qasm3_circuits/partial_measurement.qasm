OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[2];

// Partial measurement (not all qubits measured)
h q[0];
cx q[0], q[1];
cx q[1], q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
// q[2] is not measured
