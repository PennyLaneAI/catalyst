// Rotated distance-3 surface code [[9,1,3]] ("surface-17" layout, with one
// reused syndrome ancilla): one full stabilizer round + weight-1 lookup
// decoding of both syndromes.
//
// Data qubits on a 3x3 grid, q[3*row + col]:
//     q0 q1 q2
//     q3 q4 q5
//     q6 q7 q8
// Z-plaquettes -> zs:  Zp1={q0,q3} zs[0]; Zp2={q1,q2,q4,q5} zs[1];
//                      Zp3={q3,q4,q6,q7} zs[2]; Zp4={q5,q8} zs[3].
// X-plaquettes -> xs:  Xp1={q0,q1,q3,q4} xs[0]; Xp2={q4,q5,q7,q8} xs[1];
//                      Xp3={q1,q2} xs[2]; Xp4={q6,q7} xs[3].
// (All X/Z support overlaps are even -> generators commute; 8 independent
// generators on 9 qubits -> k=1. Logicals: Z_L=Z0Z1Z2, X_L=X0X3X6.)
//
// Weight-1 lookup (value = sum of fired bits * 2^i). Degenerate entries are
// resolved by a STABILIZER (never a logical), as distance 3 requires:
//   zs: 1->x q0 | 2->x q1 (X q2 equivalent: X1X2 = Xp3) | 5->x q3 | 6->x q4
//       | 10->x q5 | 4->x q6 (X q7 equivalent: X6X7 = Xp4) | 8->x q8
//   xs: 1->z q0 (Z q3 equivalent: Z0Z3 = Zp1) | 5->z q1 | 4->z q2 | 3->z q4
//       | 2->z q5 (Z q8 equivalent: Z5Z8 = Zp4) | 8->z q6 | 10->z q7
//
// Injected X q[4] -> Zp2,Zp3 fire -> zs == 6 -> x q[4] (exact).
// Injected Z q[7] -> Xp2,Xp4 fire -> xs == 10 -> z q[7] (exact).
// Deterministic expectation: zs = "0110", xs = "1010", out = "000000000".
OPENQASM 3.0;
include "stdgates.inc";

def zcheck2(qubit d0, qubit d1, qubit an) -> bit {
  reset an;
  cx d0, an;
  cx d1, an;
  return measure an;
}

def zcheck4(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  cx d0, an;
  cx d1, an;
  cx d2, an;
  cx d3, an;
  return measure an;
}

def xcheck2(qubit d0, qubit d1, qubit an) -> bit {
  reset an;
  h an;
  cx an, d0;
  cx an, d1;
  h an;
  return measure an;
}

def xcheck4(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  h an;
  cx an, d0;
  cx an, d1;
  cx an, d2;
  cx an, d3;
  h an;
  return measure an;
}

qubit[9] q;
qubit[1] anc;
bit[4] zs;
bit[4] xs;
bit[9] out;

reset q;
reset anc;

// encode |0>_L: pivots q0 (Xp1), q8 (Xp2), q2 (Xp3), q6 (Xp4)
h q[0];
h q[2];
h q[6];
h q[8];
cx q[0], q[1];
cx q[0], q[3];
cx q[0], q[4];
cx q[8], q[4];
cx q[8], q[5];
cx q[8], q[7];
cx q[2], q[1];
cx q[6], q[7];

// inject errors
x q[4];
z q[7];
barrier q;

// Z-plaquette round
zs[0] = zcheck2(q[0], q[3], anc[0]);
zs[1] = zcheck4(q[1], q[2], q[4], q[5], anc[0]);
zs[2] = zcheck4(q[3], q[4], q[6], q[7], anc[0]);
zs[3] = zcheck2(q[5], q[8], anc[0]);
// X-plaquette round
xs[0] = xcheck4(q[0], q[1], q[3], q[4], anc[0]);
xs[1] = xcheck4(q[4], q[5], q[7], q[8], anc[0]);
xs[2] = xcheck2(q[1], q[2], anc[0]);
xs[3] = xcheck2(q[6], q[7], anc[0]);

// weight-1 X-error lookup on the Z syndrome
if (zs == 1) x q[0];
if (zs == 2) x q[1];
if (zs == 5) x q[3];
if (zs == 6) x q[4];
if (zs == 10) x q[5];
if (zs == 4) x q[6];
if (zs == 8) x q[8];
// weight-1 Z-error lookup on the X syndrome
if (xs == 1) z q[0];
if (xs == 5) z q[1];
if (xs == 4) z q[2];
if (xs == 3) z q[4];
if (xs == 2) z q[5];
if (xs == 8) z q[6];
if (xs == 10) z q[7];

// decode (same CXs — disjoint controls/targets — then H on pivots)
cx q[0], q[1];
cx q[0], q[3];
cx q[0], q[4];
cx q[8], q[4];
cx q[8], q[5];
cx q[8], q[7];
cx q[2], q[1];
cx q[6], q[7];
h q[0];
h q[2];
h q[6];
h q[8];
out = measure q;
