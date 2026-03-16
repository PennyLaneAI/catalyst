OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - param1) q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; rz(-pi/2) q0; sx q0; rz(pi/2) q0; s q1; cx q0,q1; ry(param0/2) q0; ry(-param0/2) q1; cx q0,q1; sdg q1; rz(-pi/2) q0; sxdg q0; rz(pi/2) q0; rz(param1) q1; }
qreg q[3];
creg c[3];
u2(0.0853624864807074,4.877860452562436) q[0];
crz(6.256509953199805) q[2],q[1];
r(2.338909940516581,4.515469484840054) q[2];
ecr q[0],q[1];
crx(4.96310929455875) q[0],q[2];
id q[1];
xx_minus_yy(0.6339349974087864,5.274566127282956) q[0],q[1];
u2(3.639550681202242,5.4692650980573445) q[2];
sdg q[0];
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];