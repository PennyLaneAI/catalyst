OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[3];
creg meas[3];
s q[2];
ecr q[0],q[2];
y q[2];
h q[2];
s q[0];
x q[1];
y q[1];
cx q[1],q[2];
sx q[1];
ecr q[0],q[1];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];