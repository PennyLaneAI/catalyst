// Steane code [[7,1,3]] — CSS code from the [7,4] Hamming code.
// Qubit i corresponds to Hamming parity-check column i+1, so a nonzero
// syndrome value k directly names the flipped qubit q[k-1].
//
// Stabilizer generators (same supports for X-type and Z-type):
//   row1 (syndrome bit 0): {q0, q2, q4, q6}
//   row2 (syndrome bit 1): {q1, q2, q5, q6}
//   row3 (syndrome bit 2): {q3, q4, q5, q6}
// All pairwise support overlaps are even, so X- and Z-generators commute.
//
// Encode |0>_L (pivots q0, q1, q3 — each exclusive to its row), inject
// X on q[2] (column 3) and Z on q[5] (column 6), extract both syndromes
// with one reused ancilla, correct via full 7-entry lookup chains, decode.
//
// Deterministic expectation: zsyn = "011" (3), xsyn = "110" (6), out = "0000000".
OPENQASM 3.0;
include "stdgates.inc";

def zpar4(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  cx d0, an;
  cx d1, an;
  cx d2, an;
  cx d3, an;
  return measure an;
}

def xpar4(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  h an;
  cx an, d0;
  cx an, d1;
  cx an, d2;
  cx an, d3;
  h an;
  return measure an;
}

qubit[7] q;
qubit[1] anc;
bit[3] zsyn;
bit[3] xsyn;
bit[7] out;

reset q;
reset anc;

// encode |0>_L
h q[0];
h q[1];
h q[3];
cx q[0], q[2];
cx q[0], q[4];
cx q[0], q[6];
cx q[1], q[2];
cx q[1], q[5];
cx q[1], q[6];
cx q[3], q[4];
cx q[3], q[5];
cx q[3], q[6];

// inject errors: X on q2 (column 3 = 011b), Z on q5 (column 6 = 110b)
x q[2];
z q[5];
barrier q;

// Z-type syndrome (detects X errors)
zsyn[0] = zpar4(q[0], q[2], q[4], q[6], anc[0]);
zsyn[1] = zpar4(q[1], q[2], q[5], q[6], anc[0]);
zsyn[2] = zpar4(q[3], q[4], q[5], q[6], anc[0]);
// X-type syndrome (detects Z errors)
xsyn[0] = xpar4(q[0], q[2], q[4], q[6], anc[0]);
xsyn[1] = xpar4(q[1], q[2], q[5], q[6], anc[0]);
xsyn[2] = xpar4(q[3], q[4], q[5], q[6], anc[0]);

// lookup: syndrome value k names the flipped qubit q[k-1]
if (zsyn == 1) x q[0];
if (zsyn == 2) x q[1];
if (zsyn == 3) x q[2];
if (zsyn == 4) x q[3];
if (zsyn == 5) x q[4];
if (zsyn == 6) x q[5];
if (zsyn == 7) x q[6];
if (xsyn == 1) z q[0];
if (xsyn == 2) z q[1];
if (xsyn == 3) z q[2];
if (xsyn == 4) z q[3];
if (xsyn == 5) z q[4];
if (xsyn == 6) z q[5];
if (xsyn == 7) z q[6];

// decode (encoder CXs have disjoint controls/targets, so they commute;
// inverse = same CX list, then H on the pivots)
cx q[3], q[4];
cx q[3], q[5];
cx q[3], q[6];
cx q[1], q[2];
cx q[1], q[5];
cx q[1], q[6];
cx q[0], q[2];
cx q[0], q[4];
cx q[0], q[6];
h q[0];
h q[1];
h q[3];
out = measure q;
