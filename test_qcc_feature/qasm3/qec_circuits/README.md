# QEC Dynamic-Circuit Test Suite

Quantum Error Correction circuits written in OpenQASM 3 **dynamic-circuit
syntax** (mid-circuit measurement, `reset`, feedforward `if (reg == k)`,
adaptive `while (reg != 0)`) to exercise the full QCC pipeline:

```
QASM3 → qasm3_frontend.load_qasm3 → Catalyst MLIR → quantum-opt → quantum-translate → QASM3
```

Every circuit encodes a logical state, **injects a known deterministic
error**, extracts the syndrome, applies feedforward correction (or
repeat-until-clean re-preparation), decodes, and reads out — so the final
register values are exactly predictable and asserted by
`test_qasm3_translation_pytest.py::TestQECCircuits`.

## Circuits

| File | Code | [[n,k,d]] | Qubits (data+anc) | Injected error | Dynamic features | Expected registers |
|------|------|-----------|-------------------|----------------|------------------|--------------------|
| `rep3_bitflip.qasm` | bit-flip repetition | 3,1,(1/3)* | 3+2 | X q1 | reset, `if (syn == k)` | syn="11", out="111" |
| `rep3_phaseflip.qasm` | phase-flip repetition | 3,1,(1/3)* | 3+2 | Z q1 | def, let, nested if/else | syn="11", out="000" |
| `rep3_bitflip_while.qasm` | bit-flip, adaptive | 3,1,(1/3)* | 3+2 | X q2 | `while (syn != 0)`, if==k in loop | syn="00", out="111" |
| `shor9_full.qasm` | Shor | 9,1,3 | 9+1 | **Y** q4 | const, for, reset-reuse ×8 | s0="00" s1="11" s2="00" xs="11", out="000000001" |
| `steane7_lookup.qasm` | Steane | 7,1,3 | 7+1 | X q2 **and** Z q5 | def ×2, 7-entry lookups | zsyn="011" xsyn="110", out="0000000" |
| `surface_d2_detect.qasm` | surface (detection) | 4,1,2 | 4+1 | X q0 | `while`, whole-register reset, re-prepare | det="000", out ∈ {"0000","1111"} |
| `surface17_d3_round.qasm` | rotated surface | 9,1,3 | 9+1 | X q4 **and** Z q7 | def ×4, weight-1 lookup decode | zs="0110" xs="1010", out="000000000" |
| `toric_2x2_detect.qasm` | toric (qLDPC) | 8,2,2 | 8+1 | X q4 | plain syntax, consistency bit | xs="0000" zs="0011", out="00000000" |
| `toric_2x2_rounds.qasm` | toric, multi-round | 8,2,2 | 8+1 | X q4 | const, for (3 rounds), def | xs="000" zs="011", out="00000000" |

\* the 3-qubit repetition code has distance 3 against its protected error
type (X or Z) and distance 1 against the other.

## Conventions

- **Endianness**: `bit[n] c;` is little-endian — `c[i]` is bit `i` (weight
  `2^i`) of any `c == k` comparison. Qiskit/Aer count keys print
  `c[n-1]…c[0]` (MSB leftmost); multiple registers appear space-separated
  with the **last-declared register leftmost**.
- **Z-type check** (parity of ⊗Z): ancilla in |0⟩, `cx data, anc`, measure.
- **X-type check** (⊗X eigenvalue): `h anc; cx anc, data...; h anc`, measure.
- **Encoders**: each |0⟩_L is the uniform superposition over the span of the
  code's X-generators. Encoders `h` one exclusive *pivot* qubit per
  X-generator and CX-fan out. Controls and targets are disjoint sets, so all
  encoder CXs commute and the **decoder is the same CX list followed by the
  same `h` gates** — decoding a corrected state gives a deterministic
  bit string.
- A **single reused ancilla** (with `reset` between extractions) measures
  all stabilizers in the larger codes — exercising mid-circuit reset.

## Code details

### Shor [[9,1,3]] (`shor9_full.qasm`)
Blocks {0,1,2},{3,4,5},{6,7,8}. Z-stabilizers: Z0Z1, Z1Z2 (and per-block
analogues); X-stabilizers: X0..X5, X3..X8. The injected **Y** error tests
both decoders at once: its X component gives the block-1 syndrome `s1 == 3
→ x q[4]` (exact); its Z component fires both X-checks, `xs == 3 → z q[3]`
(exact up to the stabilizer Z3Z4).

### Steane [[7,1,3]] (`steane7_lookup.qasm`)
Qubit `i` ↔ Hamming column `i+1`; stabilizer rows (X- and Z-type share
supports): r1={0,2,4,6}, r2={1,2,5,6}, r3={3,4,5,6}. The syndrome value
**is** the flipped qubit's column: `if (syn == k) correct q[k-1]`.

### Rotated d=3 surface code (`surface17_d3_round.qasm`)
3×3 data grid, q[3·row+col]. Z-plaquettes {0,3}, {1,2,4,5}, {3,4,6,7},
{5,8}; X-plaquettes {0,1,3,4}, {4,5,7,8}, {1,2}, {6,7}. Logicals
Z_L=Z0Z1Z2, X_L=X0X3X6. The weight-1 lookup tables are in the file header;
every degenerate syndrome pair (e.g. X q1 vs X q2 → zs==2) is resolved by a
**stabilizer** (X1X2 = Xp3), never a logical — as distance 3 requires.

### 2×2 toric code [[8,2,2]] (`toric_2x2_*.qasm`) — qLDPC
The toric code is the prototypical **quantum LDPC** code: constant-weight-4
sparse checks, k=2. Edges q0..q3 horizontal, q4..q7 vertical; vertex
X-checks A0={0,1,4,6}, A1={0,1,5,7}, A2={2,3,4,6}, A3={2,3,5,7}; plaquette
Z-checks B0={0,2,4,5}, B1={1,3,4,5}, B2={0,2,6,7}, B3={1,3,6,7}. Only 3 of
each 4 are independent (`toric_2x2_detect.qasm` measures the 4th as an
XOR-consistency bit).

**Scope note**: at d=2 the syndrome `zs == 3` is shared by X q4 and X q5,
which differ by the logical X̄b — a real decoder cannot choose, and
production qLDPC decoding (BP+OSD) requires runtime classical computation
(`extern` territory, outside a translation pipeline). The `if (zs == 3)
x q[4];` undo is an explicitly-labeled **test fixture** so the readout is
deterministic; these files test syndrome-extraction circuit shape and
lookup feedforward.

## Measurement-free QEC (coherent feedback)

Based on the scheme of S. Heußen, D. F. Locher and M. Müller,
*"Measurement-Free Fault-Tolerant Quantum Error Correction in Near-Term
Devices"*, PRX Quantum **5**, 010333 (2024) / arXiv:2307.13296: instead of
measuring syndromes and applying classically-conditioned corrections, the
syndrome is mapped **coherently** onto ancilla qubits and corrections are
applied by multi-controlled gates whose positive/negative control patterns
enumerate the syndrome values; ancillas are erased by `reset`. These files
therefore exercise the *complementary* half of the dynamic-circuit
envelope: gate modifiers (`ctrl @` / `negctrl @`), `ccx`, and reset — with
the explicit negative assertion that the translated output contains **no
`if`/`while` at all**.

| File | Construction (paper ref) | Qubits | Injected error | Expected |
|------|--------------------------|--------|----------------|----------|
| `mf_rep3_coherent.qasm` | coherent-feedback principle, 3-qubit warm-up | 3+2 | X q1 | out="111" |
| `mf_steane_prep_flag.qasm` | flag-verified \|0⟩_L prep (Fig. 6 structure) | 7+2 | X q0·X q1 (≡ X_L·X2, the dangerous class) | out="0000000" |
| `mf_steane_cycle_x.qasm` | full MF-QEC cycle, X-half (Fig. 2), N = 2n+a = 17 | 7+7+3 | X q2 | out="0000000" |

Adaptation notes: the circuits use **our verified encoders** (pivot
construction above) rather than the paper's encoding circuits, so the
flag-verification file targets the dangerous weight-2 class of *our*
encoder and undoes the known injected error fixture-style (labeled in the
file, like the toric files). `mf_steane_cycle_x` implements the X-error
half at the paper's 17-qubit budget with all seven coherent-feedback
C3NOTs (`ctrl @`/`negctrl @` chains over the 3 syndrome qubits, syndrome
value k targeting data qubit k−1); the Z-half is the Hadamard-dual and is
not duplicated.

## Running

```bash
cd /home/ubuntu/catalyst
# pytest (structure + grammar + Aer semantics of source AND translated programs):
python -m pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py::TestQECCircuits -v
# CI-style batch through the full pipeline:
python qcc_simulator.py --batch test_qcc_feature/qasm3/qec_circuits/
```

Note: files with `while` loops translate to `!=` register comparisons,
which are valid QASM3 but rejected by qiskit's re-importer — those outputs
are grammar-validated with the reference `openqasm3` parser instead (and
their physics is asserted on the source circuit).
