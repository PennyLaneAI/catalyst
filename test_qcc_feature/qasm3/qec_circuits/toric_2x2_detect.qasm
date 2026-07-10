// 2x2 toric code [[8,2,2]] — the prototypical quantum LDPC (qLDPC) code:
// constant-weight (4) sparse checks on a periodic lattice, k=2 logical qubits.
//
// Edge qubits (vertices (r,c), r,c in {0,1}, periodic boundary):
//   horizontal h(r,c): h(0,0)=q0 h(0,1)=q1 h(1,0)=q2 h(1,1)=q3
//   vertical   v(r,c): v(0,0)=q4 v(0,1)=q5 v(1,0)=q6 v(1,1)=q7
// Vertex X-checks (stars):     A0={q0,q1,q4,q6}  A1={q0,q1,q5,q7}
//                              A2={q2,q3,q4,q6}  A3={q2,q3,q5,q7}
// Plaquette Z-checks:          B0={q0,q2,q4,q5}  B1={q1,q3,q4,q5}
//                              B2={q0,q2,q6,q7}  B3={q1,q3,q6,q7}
// Products A0A1A2A3 = B0B1B2B3 = I, so only 3 of each are independent;
// we measure all 4 anyway — the 4th outcome is a parity-consistency bit.
// Logicals: Zbar_a=Z0Z1, Zbar_b=Z4Z6, Xbar_a=X0X2, Xbar_b=X4X5 (weight 2 -> d=2).
//
// Injected X q[4] fires B0 and B1 -> zs == 3 (zs[3] consistency bit = 0).
// NOTE (fixture, not a decoder): at distance 2 the syndrome zs==3 is shared
// by X q4 and X q5, which differ by the LOGICAL Xbar_b — a real decoder
// cannot choose. This test knows the injected error and undoes it to get a
// deterministic readout. Production qLDPC decoding (BP+OSD) requires runtime
// classical computation ('extern' territory) and is out of pipeline scope;
// this file tests the syndrome-extraction circuit shape + lookup feedforward.
//
// Deterministic expectation: xs = "0000", zs = "0011", out = "00000000".
OPENQASM 3.0;
include "stdgates.inc";

qubit[8] q;
qubit[1] anc;
bit[4] xs;
bit[4] zs;
bit[8] out;

reset q;
reset anc;

// encode |00>_L: independent X-generators {A0, A2, A0*A1={q4,q5,q6,q7}}
// with exclusive pivots q0, q2, q5
h q[0];
h q[2];
h q[5];
cx q[0], q[1];
cx q[0], q[4];
cx q[0], q[6];
cx q[2], q[3];
cx q[2], q[4];
cx q[2], q[6];
cx q[5], q[4];
cx q[5], q[6];
cx q[5], q[7];

// inject error
x q[4];
barrier q;

// four vertex X-checks
h anc[0];
cx anc[0], q[0];
cx anc[0], q[1];
cx anc[0], q[4];
cx anc[0], q[6];
h anc[0];
xs[0] = measure anc[0];
reset anc[0];
h anc[0];
cx anc[0], q[0];
cx anc[0], q[1];
cx anc[0], q[5];
cx anc[0], q[7];
h anc[0];
xs[1] = measure anc[0];
reset anc[0];
h anc[0];
cx anc[0], q[2];
cx anc[0], q[3];
cx anc[0], q[4];
cx anc[0], q[6];
h anc[0];
xs[2] = measure anc[0];
reset anc[0];
h anc[0];
cx anc[0], q[2];
cx anc[0], q[3];
cx anc[0], q[5];
cx anc[0], q[7];
h anc[0];
xs[3] = measure anc[0];

// four plaquette Z-checks
reset anc[0];
cx q[0], anc[0];
cx q[2], anc[0];
cx q[4], anc[0];
cx q[5], anc[0];
zs[0] = measure anc[0];
reset anc[0];
cx q[1], anc[0];
cx q[3], anc[0];
cx q[4], anc[0];
cx q[5], anc[0];
zs[1] = measure anc[0];
reset anc[0];
cx q[0], anc[0];
cx q[2], anc[0];
cx q[6], anc[0];
cx q[7], anc[0];
zs[2] = measure anc[0];
reset anc[0];
cx q[1], anc[0];
cx q[3], anc[0];
cx q[6], anc[0];
cx q[7], anc[0];
zs[3] = measure anc[0];

// fixture undo of the known injected error (see NOTE above)
if (zs == 3) x q[4];

// decode (same CXs, then H on pivots)
cx q[0], q[1];
cx q[0], q[4];
cx q[0], q[6];
cx q[2], q[3];
cx q[2], q[4];
cx q[2], q[6];
cx q[5], q[4];
cx q[5], q[6];
cx q[5], q[7];
h q[0];
h q[2];
h q[5];
out = measure q;
