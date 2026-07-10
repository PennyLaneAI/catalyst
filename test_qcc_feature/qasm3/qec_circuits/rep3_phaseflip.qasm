// 3-qubit phase-flip repetition code (protects against Z errors).
// Encode |0>_L = |+++>, inject Z on q[1], measure the two XX-parity checks
// with a subroutine, correct via a nested if/else chain, decode with H^3.
//
// Syndrome table (X_a X_b anticommutes with Z on its support):
//   Z q[0] -> X0X1 fires            -> syn == 1 -> z q[0]
//   Z q[1] -> X0X1 and X1X2 fire    -> syn == 3 -> z q[1]
//   Z q[2] -> X1X2 fires            -> syn == 2 -> z q[2]
// Deterministic expectation: syn = "11" (value 3), out = "000".
OPENQASM 3.0;
include "stdgates.inc";

def xxsyn(qubit d0, qubit d1, qubit an) -> bit {
  reset an;
  h an;
  cx an, d0;
  cx an, d1;
  h an;
  return measure an;
}

qubit[3] q;
qubit[2] a;
bit[2] syn;
bit[3] out;

let data = q;

reset q;
reset a;

// encode |0>_L = |+++>
h q[0];
h q[1];
h q[2];

// inject error
z q[1];
barrier q;

syn[0] = xxsyn(data[0], data[1], a[0]);  // X0X1
syn[1] = xxsyn(data[1], data[2], a[1]);  // X1X2

// nested if/else correction chain
if (syn == 1) {
  z q[0];
} else {
  if (syn == 3) {
    z q[1];
  } else {
    if (syn == 2) {
      z q[2];
    }
  }
}

// decode to computational basis
h q[0];
h q[1];
h q[2];
out = measure q;
