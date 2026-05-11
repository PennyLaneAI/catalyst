OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

// Controlled rotation gates
x q[0];
crx(0.5) q[0], q[1];
cry(1.2) q[0], q[2];
crz(2.1) q[0], q[3];

measure q -> c;
