OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - param1) q0; }
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[3];
creg c[3];
u(5.423122828661231,3.7060928635807873,3.3173542427595915) q[0];
z q[2];
y q[1];
x q[0];
x q[2];
tdg q[1];
r(4.4609606923418985,4.649402395428016) q[2];
s q[0];
y q[1];
id q[0];
cp(2.9166456805744834) q[1],q[2];
ryy(2.192493939063765) q[2],q[1];
r(5.98023372319067,0.4539559844656487) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];