// Distance-2 surface code [[4,1,2]] — the smallest surface-code patch.
// Stabilizers: X0X1X2X3 (-> det[0]), Z0Z1 (-> det[1]), Z2Z3 (-> det[2]).
// Logicals: X_L = X0X1, Z_L = Z0Z2. |0>_L = (|0000> + |1111>)/sqrt2 (GHZ-4).
//
// A distance-2 code only DETECTS single errors (e.g. X0 and X1 give the
// same syndrome and differ by the logical X_L), so instead of correcting
// we use the repeat-until-clean pattern: while any detector fired, discard
// the state, re-prepare |0>_L from scratch, and re-measure all detectors.
//
// Injected X q[0] fires det[1] (value 2); the while body re-prepares once
// and exits with a clean syndrome.
// Deterministic expectation: det = "000"; out is "0000" or "1111" (GHZ).
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
qubit[1] anc;
bit[3] det;
bit[4] out;

reset q;
reset anc;

// encode |0>_L = GHZ-4
h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];

// inject error
x q[0];
barrier q;

// detection round: XXXX, Z0Z1, Z2Z3
h anc[0];
cx anc[0], q[0];
cx anc[0], q[1];
cx anc[0], q[2];
cx anc[0], q[3];
h anc[0];
det[0] = measure anc[0];
reset anc[0];
cx q[0], anc[0];
cx q[1], anc[0];
det[1] = measure anc[0];
reset anc[0];
cx q[2], anc[0];
cx q[3], anc[0];
det[2] = measure anc[0];

// repeat-until-clean state preparation
while (det != 0) {
  reset q;
  reset anc[0];
  h q[0];
  cx q[0], q[1];
  cx q[0], q[2];
  cx q[0], q[3];
  h anc[0];
  cx anc[0], q[0];
  cx anc[0], q[1];
  cx anc[0], q[2];
  cx anc[0], q[3];
  h anc[0];
  det[0] = measure anc[0];
  reset anc[0];
  cx q[0], anc[0];
  cx q[1], anc[0];
  det[1] = measure anc[0];
  reset anc[0];
  cx q[2], anc[0];
  cx q[3], anc[0];
  det[2] = measure anc[0];
}

out = measure q;
