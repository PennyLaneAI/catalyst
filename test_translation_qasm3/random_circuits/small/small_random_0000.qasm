OPENQASM 2.0;
include "qelib1.inc";
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; rz(-pi/2) q0; sx q0; rz(pi/2) q0; s q1; cx q0,q1; ry(param0/2) q0; ry(-param0/2) q1; cx q0,q1; sdg q1; rz(-pi/2) q0; sxdg q0; rz(pi/2) q0; rz(param1) q1; }
qreg q[3];
creg c[3];
xx_minus_yy(4.782381792256834,4.938987693414485) q[0],q[2];
sdg q[1];
u3(4.0455238622962515,5.169563679814652,2.7860535790666945) q[2];
u(1.42778299794038,3.4845589853632135,0.40097564589827117) q[0];
crz(6.0990755645863075) q[0],q[1];
rxx(0.27522717759345167) q[1],q[2];
rx(0.9694294696110178) q[0];
swap q[0],q[1];
t q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];