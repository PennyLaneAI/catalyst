// MEASUREMENT-FREE Steane-code QEC cycle, X-error half — the flagship
// construction of Heussen, Locher & Muller, PRX Quantum 5, 010333 (2024) /
// arXiv:2307.13296 (their Fig. 2), at the paper's qubit budget
// N = 2n + a = 7 + 7 + 3 = 17:
//
//   [q]   logical DATA block (7 qubits, here |0>_L)
//   [aux] logical AUXILIARY block (7 qubits, |0>_L)
//   [s]   3 physical syndrome qubits
//
// Cycle: (1) transversal CNOT data->aux copies any X error onto the
// auxiliary logical block; (2) the auxiliary block's three Z-stabilizer
// parities are mapped coherently onto the syndrome qubits; (3) SEVEN
// coherent feedback C3NOT gates — one per nonzero syndrome, with
// positive/negative control patterns enumerating the 3-bit syndrome values
// (Steane property: syndrome value k names data qubit q[k-1]); (4) the
// auxiliary and syndrome registers are erased by reset for reuse in the
// next half/cycle. No measurement or feedforward anywhere.
//
// This file implements the X-correcting half; the Z-half is its
// Hadamard-dual (aux prepared in |+>_L, transversal CNOT reversed, C3Z
// feedback) and is structurally identical.
//
// Injected X q[2] (Hamming column 3) -> syndrome s = (1,1,0), so only the
// k=3 feedback gate fires and flips q[2] back. Deterministic expectation:
// out = "0000000". No if/while in the translated output.
OPENQASM 3.0;
include "stdgates.inc";

def encode_zero(qubit[7] d) {
  h d[0];
  h d[1];
  h d[3];
  cx d[0], d[2];
  cx d[0], d[4];
  cx d[0], d[6];
  cx d[1], d[2];
  cx d[1], d[5];
  cx d[1], d[6];
  cx d[3], d[4];
  cx d[3], d[5];
  cx d[3], d[6];
}

qubit[7] q;
qubit[7] aux;
qubit[3] s;
bit[7] out;

reset q;
reset aux;
reset s;

encode_zero(q);
encode_zero(aux);

// inject error on the data block
x q[2];
barrier q;

// (1) transversal CNOT: copy X errors from data onto the auxiliary block
cx q[0], aux[0];
cx q[1], aux[1];
cx q[2], aux[2];
cx q[3], aux[3];
cx q[4], aux[4];
cx q[5], aux[5];
cx q[6], aux[6];

// (2) map the auxiliary block's Z-rows onto the syndrome qubits
cx aux[0], s[0];
cx aux[2], s[0];
cx aux[4], s[0];
cx aux[6], s[0];
cx aux[1], s[1];
cx aux[2], s[1];
cx aux[5], s[1];
cx aux[6], s[1];
cx aux[3], s[2];
cx aux[4], s[2];
cx aux[5], s[2];
cx aux[6], s[2];

// (3) seven coherent feedback C3NOTs, syndrome value k -> flip q[k-1]
// (first operand = outermost modifier's control; s[j] positive iff bit j of k)
ctrl @ negctrl @ negctrl @ x s[0], s[1], s[2], q[0];  // k=1
negctrl @ ctrl @ negctrl @ x s[0], s[1], s[2], q[1];  // k=2
ctrl @ ctrl @ negctrl @ x s[0], s[1], s[2], q[2];     // k=3  <- fires
negctrl @ negctrl @ ctrl @ x s[0], s[1], s[2], q[3];  // k=4
ctrl @ negctrl @ ctrl @ x s[0], s[1], s[2], q[4];     // k=5
negctrl @ ctrl @ ctrl @ x s[0], s[1], s[2], q[5];     // k=6
ctrl @ ctrl @ ctrl @ x s[0], s[1], s[2], q[6];        // k=7

// (4) erase auxiliary and syndrome registers for the next half/cycle
reset aux;
reset s;

// decode the data block and read out
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
