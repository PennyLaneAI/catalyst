// Shor code [[9,1,3]] — full cycle: encode |1>_L, inject a Y error,
// extract all 8 stabilizers with ONE reused ancilla, correct both the
// bit-flip and phase-flip components, decode, and read out.
//
// Blocks: {q0,q1,q2}, {q3,q4,q5}, {q6,q7,q8}.
// Z-type stabilizers (bit-flip checks, per block): Z0Z1, Z1Z2 / Z3Z4, Z4Z5 / Z6Z7, Z7Z8.
// X-type stabilizers (phase checks): X0X1X2X3X4X5, X3X4X5X6X7X8.
// Every Z-generator overlaps every X-generator on 0 or 2 qubits -> commute.
//
// Injected Y q[4] = i * X4 * Z4:
//   X4 flips Z3Z4 and Z4Z5           -> s1 == 3  -> x q[4]  (exact)
//   Z4 flips both X-checks           -> xs == 3  -> z q[3]  (exact up to the
//                                       stabilizer Z3Z4, so the state is restored)
// Per-block bit-flip lookup: value 1 -> first qubit, 3 -> middle, 2 -> last.
// Phase lookup: xs == 1 -> block 0, 3 -> block 1, 2 -> block 2 (any Z in block).
//
// Deterministic expectation: s0="00", s1="11", s2="00", xs="11", out="000000001".
OPENQASM 3.0;
include "stdgates.inc";

const uint HI1 = 5;  // upper bound of the first X-check CX chain

qubit[9] q;
qubit[1] anc;
bit[2] s0;
bit[2] s1;
bit[2] s2;
bit[2] xs;
bit[9] out;

reset q;
reset anc;

// encode |1>_L = ((|000> - |111>)/sqrt2)^(x3)
x q[0];
cx q[0], q[3];
cx q[0], q[6];
h q[0];
h q[3];
h q[6];
cx q[0], q[1];
cx q[0], q[2];
cx q[3], q[4];
cx q[3], q[5];
cx q[6], q[7];
cx q[6], q[8];

// inject Y error on q4
y q[4];
barrier q;

// six Z-checks with one reused ancilla
reset anc[0];
cx q[0], anc[0];
cx q[1], anc[0];
s0[0] = measure anc[0];
reset anc[0];
cx q[1], anc[0];
cx q[2], anc[0];
s0[1] = measure anc[0];
reset anc[0];
cx q[3], anc[0];
cx q[4], anc[0];
s1[0] = measure anc[0];
reset anc[0];
cx q[4], anc[0];
cx q[5], anc[0];
s1[1] = measure anc[0];
reset anc[0];
cx q[6], anc[0];
cx q[7], anc[0];
s2[0] = measure anc[0];
reset anc[0];
cx q[7], anc[0];
cx q[8], anc[0];
s2[1] = measure anc[0];

// two X-checks via for-loop CX chains
reset anc[0];
h anc[0];
for uint i in [0:HI1] {
  cx anc[0], q[i];
}
h anc[0];
xs[0] = measure anc[0];
reset anc[0];
h anc[0];
for uint i in [3:8] {
  cx anc[0], q[i];
}
h anc[0];
xs[1] = measure anc[0];

// per-block bit-flip corrections
if (s0 == 1) x q[0];
if (s0 == 3) x q[1];
if (s0 == 2) x q[2];
if (s1 == 1) x q[3];
if (s1 == 3) x q[4];
if (s1 == 2) x q[5];
if (s2 == 1) x q[6];
if (s2 == 3) x q[7];
if (s2 == 2) x q[8];

// phase-flip corrections (any Z within the identified block works)
if (xs == 1) z q[0];
if (xs == 3) z q[3];
if (xs == 2) z q[6];

// decode (inverse encoder)
cx q[0], q[1];
cx q[0], q[2];
cx q[3], q[4];
cx q[3], q[5];
cx q[6], q[7];
cx q[6], q[8];
h q[0];
h q[3];
h q[6];
cx q[0], q[6];
cx q[0], q[3];
out = measure q;
