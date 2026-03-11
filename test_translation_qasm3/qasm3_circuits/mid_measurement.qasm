OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c0[1];
creg c1[1];

h q[0];
measure q[0] -> c0[0];
if(c0==1) x q[1];
measure q[1] -> c1[0];
