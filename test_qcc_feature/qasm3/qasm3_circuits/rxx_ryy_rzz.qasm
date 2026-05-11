OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];

// Two-qubit rotation gates (rzz is in qelib1)
rzz(0.5) q[0], q[1];
rzz(0.5) q[2], q[3];
rzz(0.5) q[4], q[5];

measure q -> c;
