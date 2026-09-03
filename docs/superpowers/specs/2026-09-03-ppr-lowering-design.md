# PennyLane PPR Lowering

## Goal

Support PennyLane's discrete `PPR` Operator2 in Catalyst's `to_ppr` pass without first
decomposing it through `PauliRot`.

## Representation and lowering

PennyLane captures `PPR` as a generic `qref.operator` because both
`angle_denominator` and `pauli_word` are compilable static arguments. Value-semantics conversion
preserves this as `quantum.operator "PPR"` with both values in `static_data`.

The `to_ppr` conversion will recognize that operator, validate its expected shape, and emit one
native `pbc.ppr` operation. If PennyLane's angle denominator is `k`, the PBC rotation kind is
`2 * k`: PennyLane uses the `PauliRot` angle convention
`exp(-i (pi / k) P / 2)`, while `pbc.ppr` represents `exp(-i pi P / rotation_kind)`.
An adjoint negates the resulting rotation kind.

The conversion will reject controlled PPRs consistently with the other operations accepted by
`to_ppr`. Invalid or missing static data will produce a compilation diagnostic rather than
silently decomposing or mis-lowering the operation.

This design does not introduce dedicated QRef or Quantum dialect PPR operations. The generic
operator already preserves all required compile-time data, and the PBC dialect remains the
canonical discrete-PPR representation after `to_ppr`.

## Decomposition preservation

Add `PPR` to Catalyst's runtime-supported operation set. This keeps device preprocessing from
decomposing the operator to its PennyLane `PauliRot` decomposition before the registered
`to_ppr` pass runs.

No decomposition rule will be added for PPR in Catalyst. Existing graph-decomposition behavior
continues to preserve any operator explicitly included in its target gate set.

## User-facing documentation

Add `qp.PPR` to the `to_ppr` supported-operations list. The dependency pin remains unchanged;
the implementation targets the API from PennyLane PR #10107.

## Tests

1. Add MLIR tests for direct conversion of positive and negative PPR denominators, multi-qubit
   Pauli words, and adjoints.
2. Add frontend coverage proving that `qp.PPR` remains a `quantum.operator "PPR"` without
   `to_ppr`, and becomes one `pbc.ppr` with `to_ppr`.
3. Add preprocessing coverage proving PPR is in Catalyst's supported operation set and therefore
   is not decomposed before lowering.
4. Keep existing PauliRot conversion tests unchanged to guard the separate arbitrary-angle path.

## Non-goals

- Executing a raw PennyLane `PPR` without a PBC-lowering pass.
- Supporting controlled PPRs in `to_ppr`.
- Updating Catalyst's PennyLane dependency pin.
- Adding a new QRef or Quantum dialect operation solely for PPR.
