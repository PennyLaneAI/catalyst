# Resolve Adjoint PPRs Before PPM Lowering

## Goal

Ensure `ppr-to-ppm` never lowers a Pauli product rotation while it is still
nested inside `quantum.adjoint`. The pass must first resolve the adjoint,
including reversing operation order and negating each PPR rotation kind, and
only then decompose PPRs into PPMs.

## Design

Expose the existing adjoint-lowering rewrite-pattern population function from
the Quantum transforms library. At the start of `PPRToPPMPass::runOnOperation`,
apply those patterns greedily to the module. Reuse is important because the
existing implementation already handles SSA remapping, reversed operation
order, nested control-flow requirements, and PPR angle negation.

After adjoint lowering succeeds, retain the current two decomposition phases:

1. Decompose non-Clifford PPRs.
2. Decompose Clifford PPRs.

If adjoint lowering fails, signal pass failure and do not attempt PPR
decomposition.

## Build Integration

Link the PBC transforms library against the Quantum transforms library so
`ppr-to-ppm` can populate the shared adjoint-lowering patterns.

## Testing

Extend the `PPRToPPM.mlir` lit test with an adjoint region containing PPRs.
Check that:

- `quantum.adjoint` is absent after the pass.
- PPRs are processed in reverse order.
- Their rotation kinds are negated before decomposition.
- No PPM remains nested beneath an adjoint operation.

The test is added and observed failing before production code changes.
