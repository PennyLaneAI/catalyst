OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

// Conditional Z gate
h q[0];
measure q[0] -> c[0];
if(c==1) z q[1];
h q[1];
measure q[1] -> c[1];
