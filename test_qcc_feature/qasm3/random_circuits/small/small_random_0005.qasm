OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[3];
creg c[3];
t q[1];
x q[0];
h q[0];
cu3(1.12619847870723,4.715308001236008,4.98242910008768) q[2],q[1];
csx q[2],q[0];
u2(1.9498594748899725,4.488875773956584) q[1];
h q[1];
u2(3.4624647622828415,0.706435948551442) q[2];
sx q[0];
ecr q[2],q[1];
u3(4.3797115221527365,3.448935460708652,1.8289191686778474) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];