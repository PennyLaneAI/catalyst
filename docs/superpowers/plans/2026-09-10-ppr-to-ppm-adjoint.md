# Adjoint PPR-to-PPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ppr-to-ppm` resolve `quantum.adjoint` regions before decomposing their PPR operations into PPM operations.

**Architecture:** Reuse the existing adjoint-lowering rewrite implementation by exposing its pattern-population function through the Quantum transforms pattern API. `PPRToPPMPass` applies those patterns first, then retains its existing non-Clifford and Clifford decomposition phases.

**Tech Stack:** C++20, MLIR rewrite patterns and passes, CMake, LLVM lit/FileCheck.

## Global Constraints

- Reuse the canonical adjoint lowering rather than duplicating PPR-specific SSA reversal logic.
- A PPR inside an adjoint has its operation order reversed and its rotation kind negated before PPR decomposition.
- Any adjoint-lowering failure fails `ppr-to-ppm` before decomposition begins.
- Preserve the existing non-Clifford-then-Clifford decomposition order.

---

### Task 1: Resolve adjoints before PPR decomposition

**Files:**
- Modify: `mlir/test/PBC/PPRToPPM.mlir`
- Modify: `mlir/include/Quantum/Transforms/Patterns.h`
- Modify: `mlir/lib/Quantum/Transforms/AdjointLowering/AdjointLowering.cpp`
- Modify: `mlir/lib/PBC/Transforms/ppr_to_ppm.cpp`
- Modify: `mlir/lib/PBC/Transforms/CMakeLists.txt`

**Interfaces:**
- Consumes: Existing `AdjointSingleOpRewritePattern`, `applyPatternsGreedily`, and PPR decomposition pattern-population functions.
- Produces: `void catalyst::quantum::populateAdjointLoweringPatterns(mlir::RewritePatternSet &patterns)`.

- [ ] **Step 1: Write the failing regression test**

Append an input section to `mlir/test/PBC/PPRToPPM.mlir`. Adjoint lowering must
reverse `Z(4)` followed by `X(-4)` into `X(4)` followed by `Z(-4)`. Clifford
decomposition represents those signs with a negated `["X", "Y"]` PPM followed
by a non-negated `["Z", "Y"]` PPM.

```mlir
// -----

func.func @test_ppr_to_ppm_adjoint(%q0 : !quantum.bit) -> !quantum.bit {
    %0 = quantum.adjoint(%q0) : !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %1 = pbc.ppr ["Z"](4) %arg0 : !quantum.bit
        %2 = pbc.ppr ["X"](-4) %1 : !quantum.bit
        quantum.yield %2 : !quantum.bit
    }
    return %0 : !quantum.bit

    // CHECK-LABEL: @test_ppr_to_ppm_adjoint
    // CHECK-NOT: quantum.adjoint
    // CHECK: pbc.ppm ["X", "Y"](-) %q0
    // CHECK: pbc.ppm ["Z", "Y"] {{.*}}
    // CHECK-NOT: quantum.adjoint
    // CHECK: return
}
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
build/bin/quantum-opt --ppr-to-ppm --split-input-file -verify-diagnostics mlir/test/PBC/PPRToPPM.mlir | build/bin/FileCheck mlir/test/PBC/PPRToPPM.mlir --check-prefix=CHECK
```

Expected: FAIL in `test_ppr_to_ppm_adjoint` because current output retains `quantum.adjoint` and lowers the positive PPR to a negated PPM inside it.

- [ ] **Step 3: Expose the shared adjoint-lowering patterns**

Add this declaration to the `catalyst::quantum` namespace in `mlir/include/Quantum/Transforms/Patterns.h`:

```cpp
void populateAdjointLoweringPatterns(mlir::RewritePatternSet &patterns);
```

In `mlir/lib/Quantum/Transforms/AdjointLowering/AdjointLowering.cpp`, include the public pattern header:

```cpp
#include "Quantum/Transforms/Patterns.h"
```

Define the population function after the anonymous namespace:

```cpp
namespace catalyst {
namespace quantum {

void populateAdjointLoweringPatterns(RewritePatternSet &patterns) {
    patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
```

Replace the pass-local `patterns.add<AdjointSingleOpRewritePattern>(...)` call with:

```cpp
populateAdjointLoweringPatterns(patterns);
```

- [ ] **Step 4: Apply adjoint lowering first in `ppr-to-ppm`**

Add the public Quantum pattern include to `mlir/lib/PBC/Transforms/ppr_to_ppm.cpp`:

```cpp
#include "Quantum/Transforms/Patterns.h"
```

At the start of `runOnOperation`, before constructing non-Clifford patterns, add:

```cpp
RewritePatternSet adjoint_patterns(ctx);
quantum::populateAdjointLoweringPatterns(adjoint_patterns);

if (failed(applyPatternsGreedily(module, std::move(adjoint_patterns)))) {
    return signalPassFailure();
}
```

- [ ] **Step 5: Link the shared implementation**

Add `quantum-transforms` to `LIBS` in `mlir/lib/PBC/Transforms/CMakeLists.txt`:

```cmake
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRPBC
    PBCUtils
    PBCAnalysis
    quantum-transforms
)
```

- [ ] **Step 6: Build the affected tools**

Run:

```bash
cmake --build build --target quantum-opt FileCheck -j2
```

Expected: Build succeeds without compile or link errors.

- [ ] **Step 7: Run the focused test and verify GREEN**

Run:

```bash
build/bin/quantum-opt --ppr-to-ppm --split-input-file -verify-diagnostics mlir/test/PBC/PPRToPPM.mlir | build/bin/FileCheck mlir/test/PBC/PPRToPPM.mlir --check-prefix=CHECK
```

Expected: PASS. The output contains no `quantum.adjoint`; the reversed
`X(-4)` decomposes first as a negated `["X", "Y"]` PPM, and the reversed
`Z(4)` decomposes second as a non-negated `["Z", "Y"]` PPM.

- [ ] **Step 8: Run adjacent adjoint and PBC tests**

Run:

```bash
build/bin/llvm-lit -sv mlir/test/PBC/PPRToPPM.mlir mlir/test/PBC/AdjointTest.mlir mlir/test/Quantum/AdjointTest.mlir
```

Expected: All selected tests pass.

- [ ] **Step 9: Commit the implementation**

```bash
git add mlir/test/PBC/PPRToPPM.mlir \
  mlir/include/Quantum/Transforms/Patterns.h \
  mlir/lib/Quantum/Transforms/AdjointLowering/AdjointLowering.cpp \
  mlir/lib/PBC/Transforms/ppr_to_ppm.cpp \
  mlir/lib/PBC/Transforms/CMakeLists.txt
git commit -m "fix: resolve adjoint PPRs before PPM lowering"
```
