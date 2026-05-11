OPENQASM 2.0;
include "qelib1.inc";
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; rz(-pi/2) q0; sx q0; rz(pi/2) q0; s q1; cx q0,q1; ry(param0/2) q0; ry(-param0/2) q1; cx q0,q1; sdg q1; rz(-pi/2) q0; sxdg q0; rz(pi/2) q0; rz(param1) q1; }
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[3];
creg c[3];
y q[0];
swap q[1],q[2];
u1(6.0000408753696295) q[0];
cu3(2.584907341708092,5.903943627034251,5.822933769757058) q[2],q[1];
id q[2];
rzz(3.291527172216614) q[1],q[0];
ry(3.4319162097782208) q[1];
xx_minus_yy(1.2781499546688175,1.8930001924522162) q[2],q[0];
ryy(4.8516280798538505) q[0],q[2];
tdg q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];