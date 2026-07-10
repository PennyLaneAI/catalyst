// 3-qubit bit-flip repetition code [[3,1,1]]-style (protects against X errors).
// Encode |1>_L = |111>, inject X on q[1], measure the two Z-parity checks,
// correct via feedforward register comparisons, and read out.
//
// Syndrome table (syn is little-endian: value = syn[0] + 2*syn[1]):
//   X q[0] -> Z0Z1 fires            -> syn == 1 -> x q[0]
//   X q[1] -> Z0Z1 and Z1Z2 fire    -> syn == 3 -> x q[1]
//   X q[2] -> Z1Z2 fires            -> syn == 2 -> x q[2]
// Deterministic expectation: syn = "11" (value 3), out = "111".
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
x q[1];
barrier q;

// Z0Z1 -> syn[0]
cx q[0], a[0];
cx q[1], a[0];
syn[0] = measure a[0];
// Z1Z2 -> syn[1]
cx q[1], a[1];
cx q[2], a[1];
syn[1] = measure a[1];

// feedforward correction
if (syn == 1) x q[0];
if (syn == 3) x q[1];
if (syn == 2) x q[2];

out = measure q;
