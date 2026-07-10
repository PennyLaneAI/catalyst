// 3-qubit bit-flip code with ADAPTIVE repeat-until-clean correction:
// keep correcting and re-measuring the syndrome while it is nonzero.
// Encode |1>_L, inject X on q[2] (syndrome value 2); the while body runs
// exactly once (correct -> re-extract -> syndrome 0 -> exit).
//
// Deterministic expectation: syn = "00", out = "111".
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
qubit[2] a;
bit[2] syn;
bit[3] out;

reset q;
reset a;

// encode |1>_L
x q[0];
cx q[0], q[1];
cx q[0], q[2];

// inject error
x q[2];
barrier q;

// first syndrome extraction (condition bits must be measured before the loop)
cx q[0], a[0];
cx q[1], a[0];
syn[0] = measure a[0];
cx q[1], a[1];
cx q[2], a[1];
syn[1] = measure a[1];

while (syn != 0) {
  if (syn == 1) x q[0];
  if (syn == 3) x q[1];
  if (syn == 2) x q[2];
  reset a[0];
  reset a[1];
  cx q[0], a[0];
  cx q[1], a[0];
  syn[0] = measure a[0];
  cx q[1], a[1];
  cx q[2], a[1];
  syn[1] = measure a[1];
}

out = measure q;
