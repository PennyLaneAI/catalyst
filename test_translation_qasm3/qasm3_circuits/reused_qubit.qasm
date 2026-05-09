OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c0[1];
creg c1[1];

h q[0];
measure q[0] -> c0[0];

reset q[0];
x q[0];
measure q[0] -> c1[0];
