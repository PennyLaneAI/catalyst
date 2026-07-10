// 2x2 toric code [[8,2,2]] (qLDPC) — MULTI-ROUND syndrome extraction:
// the same three independent checks of each type are re-measured for
// const ROUNDS rounds (for-loop, unrolled at compile time). Measuring
// stabilizers of a stabilizer eigenstate is repeatable, so every round
// yields the same syndrome (syndrome persistence); the registers hold the
// final round's values.
//
// Code layout, checks, logicals, encoder and the d=2 degeneracy note are
// identical to toric_2x2_detect.qasm. Injected X q[4] -> zs == 3 in every
// round; the `if (zs == 3) x q[4];` undo is a test fixture, not a decoder
// (production qLDPC decoding needs runtime classical compute / extern).
//
// Deterministic expectation: xs = "000", zs = "011", out = "00000000".
OPENQASM 3.0;
include "stdgates.inc";

const uint ROUNDS = 3;

def xstab(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  h an;
  cx an, d0;
  cx an, d1;
  cx an, d2;
  cx an, d3;
  h an;
  return measure an;
}

def zstab(qubit d0, qubit d1, qubit d2, qubit d3, qubit an) -> bit {
  reset an;
  cx d0, an;
  cx d1, an;
  cx d2, an;
  cx d3, an;
  return measure an;
}

qubit[8] q;
qubit[1] anc;
bit[3] xs;
bit[3] zs;
bit[8] out;

reset q;
reset anc;

// encode |00>_L (pivots q0, q2, q5)
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

// repeated rounds over the 3 independent checks of each type
for uint r in [1:ROUNDS] {
  xs[0] = xstab(q[0], q[1], q[4], q[6], anc[0]);  // A0
  xs[1] = xstab(q[0], q[1], q[5], q[7], anc[0]);  // A1
  xs[2] = xstab(q[2], q[3], q[4], q[6], anc[0]);  // A2
  zs[0] = zstab(q[0], q[2], q[4], q[5], anc[0]);  // B0
  zs[1] = zstab(q[1], q[3], q[4], q[5], anc[0]);  // B1
  zs[2] = zstab(q[0], q[2], q[6], q[7], anc[0]);  // B2
}

// fixture undo of the known injected error (see toric_2x2_detect.qasm)
if (zs == 3) x q[4];

// decode
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
