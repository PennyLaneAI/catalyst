// MEASUREMENT-FREE flag-verified Steane |0>_L preparation.
//
// Structure after Heussen, Locher & Muller, PRX Quantum 5, 010333 (2024) /
// arXiv:2307.13296 (their Fig. 6 construction): after a non-fault-tolerant
// encoding circuit, two Z-parity products are coherently mapped onto
// ancilla qubits, and a Toffoli conditioned on BOTH ancillas applies a
// correction — catching the dangerous weight-2 fault class without any
// measurement. Ancillas are then reset.
//
// Adaptation to our verified encoder (pivots q0,q1,q3): we inject the
// dangerous weight-2 error X0*X1. This error equals X_L * X2 (X_L = X0X1X2
// for this code), so a naive syndrome-lookup decoder would "correct" q2 and
// silently apply a LOGICAL X — exactly the fault class the paper's flag
// verification exists to catch. The two coherent checks:
//   row r1 = Z0 Z2 Z4 Z6 -> anc[0]   (fires on X0, not X1)
//   row r2 = Z1 Z2 Z5 Z6 -> anc[1]   (fires on X1, not X0)
// Both ancillas are deterministically 0 on a clean |0>_L, and both are 1
// for the injected X0X1.
//
// FIXTURE NOTE (like the toric files): the paper applies a single
// correction chosen from ITS encoder's fault classes; here the double-ccx
// undo of the known injected error keeps the readout deterministic while
// exercising the same Toffoli-conditioned coherent-feedback shape.
//
// Deterministic expectation: out = "0000000". No if/while in the output.
OPENQASM 3.0;
include "stdgates.inc";

qubit[7] q;
qubit[2] anc;
bit[7] out;

reset q;
reset anc;

// encode |0>_L (non-FT encoder)
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

// inject the dangerous weight-2 error (== X_L * X2)
x q[0];
x q[1];
barrier q;

// coherent verification: map two Z-rows onto the ancillas
cx q[0], anc[0];
cx q[2], anc[0];
cx q[4], anc[0];
cx q[6], anc[0];
cx q[1], anc[1];
cx q[2], anc[1];
cx q[5], anc[1];
cx q[6], anc[1];

// Toffoli-conditioned coherent correction: fires only when BOTH flags set
ccx anc[0], anc[1], q[0];
ccx anc[0], anc[1], q[1];

// erase the flags
reset anc;

// decode (same CXs, then H on pivots)
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
