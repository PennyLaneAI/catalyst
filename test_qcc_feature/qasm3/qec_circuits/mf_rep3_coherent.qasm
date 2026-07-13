// MEASUREMENT-FREE bit-flip correction on the 3-qubit repetition code.
//
// Scheme after Heussen, Locher & Muller, "Measurement-free fault-tolerant
// quantum error correction in near-term devices", PRX Quantum 5, 010333
// (2024) / arXiv:2307.13296: instead of measuring the syndrome and applying
// classically-conditioned corrections, the syndrome is mapped COHERENTLY
// onto ancilla qubits and the correction is applied by multi-controlled
// gates whose positive/negative control patterns enumerate the syndromes.
// The ancillas are then erased by reset — no mid-circuit measurement and
// no classical feedforward anywhere in the circuit.
//
// Coherent feedback table (a[0] = Z0Z1 parity, a[1] = Z1Z2 parity):
//   pattern a = (1,0) -> flip q0     ctrl @ negctrl @ x
//   pattern a = (1,1) -> flip q1     ccx
//   pattern a = (0,1) -> flip q2     negctrl @ ctrl @ x
// (First operand is the OUTERMOST modifier's control.)
//
// Injected X q[1] -> ancilla state (1,1) -> only the ccx fires.
// Deterministic expectation: out = "111". Translated output must contain
// NO if/while — that is the point of the measurement-free scheme.
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
qubit[2] a;
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

// coherent syndrome mapping onto ancillas
cx q[0], a[0];
cx q[1], a[0];
cx q[1], a[1];
cx q[2], a[1];

// coherent feedback: one multi-controlled X per syndrome pattern
ctrl @ negctrl @ x a[0], a[1], q[0];
ccx a[0], a[1], q[1];
negctrl @ ctrl @ x a[0], a[1], q[2];

// erase syndrome (ancilla reuse in the paper's repeated cycles)
reset a;

out = measure q;
