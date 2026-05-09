OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[3];
creg meas[3];
h q[1];
swap q[1],q[2];
cz q[1],q[2];
id q[0];
sdg q[2];
iswap q[1],q[0];
h q[0];
cy q[1],q[2];
y q[1];
h q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];