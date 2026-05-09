OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; rz(-pi/2) q1; sx q1; rz(pi/2) q1; s q0; cx q1,q0; ry(-param0/2) q1; ry(-param0/2) q0; cx q1,q0; sdg q0; rz(-pi/2) q1; sxdg q1; rz(pi/2) q1; rz(-param1) q0; }
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - param1) q0; }
gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[5];
creg c[5];
t q[4];
x q[2];
cs q[0],q[1];
ry(5.389459934636325) q[3];
csx q[1],q[2];
u2(5.668530697260247,1.2165438858040165) q[3];
sxdg q[4];
ry(3.8339793444716905) q[0];
rz(4.593412443618399) q[0];
dcx q[1],q[4];
ecr q[3],q[2];
sxdg q[0];
rzz(2.8454962385915743) q[2],q[4];
iswap q[1],q[3];
xx_plus_yy(0.16192953736665422,0.31910824315352604) q[2],q[3];
cs q[4],q[1];
id q[0];
x q[3];
ecr q[1],q[2];
tdg q[4];
z q[0];
p(3.129487991529346) q[3];
s q[2];
r(2.2359046548020385,6.060446246521122) q[0];
dcx q[1],q[4];
sx q[0];
xx_plus_yy(0.28240063743317084,1.9446048504986042) q[1],q[3];
y q[2];
sdg q[4];
rx(3.6748149508417263) q[0];
sx q[2];
id q[4];
sx q[3];
sxdg q[1];
cy q[0],q[4];
t q[3];
rzz(4.839309491343004) q[1],q[2];
x q[4];
r(4.045941736780225,4.300408811874294) q[2];
r(2.967796991239352,2.5264359306186925) q[1];
u3(2.4863873569200363,6.040178324310168,2.81589673605595) q[3];
s q[0];
u(4.438608035106058,6.1918032510995245,2.998489685045923) q[2];
csdg q[0],q[1];
sx q[3];
rz(5.004701208402125) q[4];
t q[3];
r(5.516657578135731,2.8012724625860224) q[4];
ryy(4.126100801163652) q[0],q[1];
ry(3.2843403176383585) q[2];
ecr q[0],q[4];
rx(3.6302051136660642) q[3];
rz(4.442217058871079) q[2];
rz(3.7904485011334352) q[1];
u1(5.718349563233436) q[2];
cu3(6.096657434379249,0.5191900436304341,2.927002599071349) q[3],q[0];
sxdg q[4];
tdg q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];