OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

// W state: (|100⟩ + |010⟩ + |001⟩)/√3
// Approximation using standard gates
ry(1.9106) q[0];
cx q[0], q[1];
x q[0];
cry(0.9553) q[1], q[0];
x q[1];
ccx q[0], q[1], q[2];

measure q -> c;
