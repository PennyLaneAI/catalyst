OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }
gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; rz(-pi/2) q0; sx q0; rz(pi/2) q0; s q1; cx q0,q1; ry(param0/2) q0; ry(-param0/2) q1; cx q0,q1; sdg q1; rz(-pi/2) q0; sxdg q0; rz(pi/2) q0; rz(param1) q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - param1) q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[5];
creg c[5];
y q[1];
swap q[2],q[4];
rx(2.123477552128429) q[3];
y q[4];
csdg q[1],q[2];
cs q[3],q[0];
cp(3.4319162097782208) q[0],q[4];
s q[3];
ry(1.2781499546688175) q[2];
ryy(1.808800876832255) q[0],q[1];
tdg q[3];
p(1.3296653935803973) q[4];
cp(5.688398583111862) q[2],q[1];
h q[3];
cs q[4],q[0];
cz q[2],q[4];
t q[3];
cz q[0],q[1];
cy q[1],q[0];
xx_minus_yy(2.6624955337168363,4.576821710770722) q[3],q[2];
t q[4];
iswap q[0],q[1];
sdg q[4];
rxx(3.7803362271511762) q[3],q[2];
csx q[0],q[1];
sdg q[2];
x q[4];
sx q[3];
cu3(1.4437961905327492,0.4888929159353887,5.23320523732292) q[1],q[2];
x q[3];
cy q[4],q[0];
ry(5.437347074421612) q[3];
cx q[0],q[1];
rzz(1.718427277766935) q[4],q[2];
h q[0];
t q[4];
rxx(4.423948886502224) q[3],q[1];
r(2.6547164490311124,3.893510934476593) q[3];
dcx q[1],q[4];
h q[0];
cry(1.3964254178398898) q[4],q[3];
dcx q[1],q[2];
h q[0];
rzx(2.365850064133139) q[1],q[3];
xx_minus_yy(1.2097239895951337,0.8596502349106676) q[0],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];