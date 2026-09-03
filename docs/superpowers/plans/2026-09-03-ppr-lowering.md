# PennyLane PPR Lowering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower PennyLane's fixed-angle `PPR` Operator2 directly to Catalyst `pbc.ppr` without an intermediate `PauliRot` decomposition.

**Architecture:** Keep `PPR` in the existing generic `qref.operator`/`quantum.operator` representation until the registered `to-ppr` pass. Add one conversion branch that reads the two static arguments and emits the canonical PBC operation, and mark `PPR` as runtime-supported so PennyLane device preprocessing preserves it.

**Tech Stack:** Python 3, PennyLane Operator2 capture, Catalyst QRef/Quantum/PBC MLIR dialects, C++17 MLIR conversion patterns, LLVM lit/FileCheck, pytest.

## Global Constraints

- PennyLane `PPR(k, P)` uses `exp(-i (pi / k) P / 2)`; Catalyst `pbc.ppr(P)(r)` uses `exp(-i pi P / r)`, so `r = 2 * k`.
- Allowed PennyLane denominators are exactly `-4`, `-2`, `-1`, `1`, `2`, and `4`.
- An adjoint negates the emitted PBC rotation kind.
- Controlled PPR remains unsupported by `to_ppr`.
- Do not add dedicated QRef or Quantum PPR operations.
- Do not update Catalyst's PennyLane dependency pin.

---

### Task 1: Convert generic quantum PPR operators in `to-ppr`

**Files:**
- Modify: `mlir/test/PBC/ToPPRTest.mlir`
- Modify: `mlir/lib/PBC/Transforms/ToPPR.cpp`

**Interfaces:**
- Consumes: `quantum.operator "PPR"()` in qubit mode with `static_data = {angle_denominator = <integer>, pauli_word = "<XYZ...>"}`.
- Produces: one `pbc.ppr` with the same qubits and Pauli word and rotation kind `2 * angle_denominator`, negated when the source operator has `adj`.

- [ ] **Step 1: Add failing direct-lowering tests**

Append split-input tests covering positive, negative, multi-qubit, and adjoint cases:

```mlir
// -----

func.func @test_ppr_operator_to_ppr(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %0:2 = quantum.operator "PPR"() qubits(%q0, %q1)
        static_data = {angle_denominator = 4 : i64, pauli_word = "XY"}
    // CHECK: pbc.ppr ["X", "Y"](8)
    func.return
}

// -----

func.func @test_negative_and_adjoint_ppr_operator(%q0 : !quantum.bit) {
    %0 = quantum.operator "PPR"() qubits(%q0)
        static_data = {angle_denominator = -2 : i64, pauli_word = "Z"}
    %1 = quantum.operator "PPR"() adj qubits(%0)
        static_data = {angle_denominator = 1 : i64, pauli_word = "X"}
    // CHECK: pbc.ppr ["Z"](-4)
    // CHECK: pbc.ppr ["X"](-2)
    func.return
}
```

- [ ] **Step 2: Run the focused MLIR test and verify RED**

Run:

```bash
lit mlir/test/PBC/ToPPRTest.mlir -v
```

Expected: FAIL because `quantum.operator "PPR"` is explicitly illegal and `PBCGateLowering` reports it as unsupported.

- [ ] **Step 3: Implement minimal PPR conversion**

In `mlir/lib/PBC/Transforms/ToPPR.cpp`, add a helper for `OperatorOp` that:

```cpp
LogicalResult convertPPROperator(OperatorOp op, ConversionPatternRewriter &rewriter)
```

The helper must:

1. Read `angle_denominator` as `IntegerAttr` and `pauli_word` as `StringAttr` from `op.getStaticData()`.
2. Require qubit mode, no dynamic parameters, a non-empty Pauli word, one Pauli character per input qubit, only `X`, `Y`, or `Z`, and an allowed denominator.
3. Build an `ArrayAttr` containing one-character string attributes.
4. Compute `int8_t rotationKind = 2 * denominator`, negating it for `op.getAdjoint()`.
5. Create `PPRotationOp` and replace the generic operator results.

Extend `PBCGateLowering::matchAndRewrite` with an `OperatorOp` branch that calls the helper only when `getOpName() == "PPR"`. Update the supported-operation diagnostic to include `PPR`.

- [ ] **Step 4: Add malformed-input diagnostics**

Add split-input tests with `expected-error` checks for:

```mlir
quantum.operator "PPR"() qubits(%q)
    static_data = {pauli_word = "X"}
```

and:

```mlir
quantum.operator "PPR"() qubits(%q)
    static_data = {angle_denominator = 3 : i64, pauli_word = "X"}
```

Expected diagnostics must identify the missing `angle_denominator` and the unsupported denominator respectively.

- [ ] **Step 5: Run the focused MLIR test and verify GREEN**

Run:

```bash
lit mlir/test/PBC/ToPPRTest.mlir -v
```

Expected: PASS with all `quantum.operator "PPR"` instances converted or diagnosed as expected.

- [ ] **Step 6: Commit the conversion**

```bash
git add mlir/lib/PBC/Transforms/ToPPR.cpp mlir/test/PBC/ToPPRTest.mlir
git commit -m "feat: lower PennyLane PPR in to-ppr"
```

---

### Task 2: Preserve PPR through device preprocessing

**Files:**
- Modify: `frontend/test/pytest/test_verification.py`
- Modify: `frontend/catalyst/device/qjit_device.py`

**Interfaces:**
- Consumes: PennyLane target-device capabilities containing an operation named `PPR`.
- Produces: QJIT capabilities that retain `PPR`, allowing PennyLane preprocessing to leave the operation intact for `to-ppr`.

- [ ] **Step 1: Add a failing capability-intersection test**

Use the existing `get_custom_device` helper to construct capabilities containing `PPR`:

```python
def test_ppr_is_supported_by_qjit_capabilities():
    """Test that QJIT preserves a target device's PPR support."""
    dev = get_custom_device(native_gates={"PPR"}, wires=1)
    target_capabilities = get_device_capabilities(dev, shots=None)

    qjit_capabilities = get_qjit_device_capabilities(target_capabilities)

    assert "PPR" in qjit_capabilities.operations
```

Keep the test independent of `qp.PPR` so it runs against the unchanged PennyLane pin.

- [ ] **Step 2: Run the focused pytest and verify RED**

Run the exact new test:

```bash
pytest frontend/test/pytest/test_verification.py::test_ppr_is_supported_by_qjit_capabilities -v
```

Expected: FAIL because `RUNTIME_OPERATIONS` does not contain `PPR`.

- [ ] **Step 3: Add PPR to runtime-supported operations**

Add the string:

```python
"PPR",
```

next to `"PauliRot"` in `RUNTIME_OPERATIONS` in `frontend/catalyst/device/qjit_device.py`.

- [ ] **Step 4: Run the focused pytest and verify GREEN**

Run:

```bash
pytest frontend/test/pytest/test_verification.py::test_ppr_is_supported_by_qjit_capabilities -v
```

Expected: PASS.

- [ ] **Step 5: Commit preprocessing preservation**

```bash
git add frontend/catalyst/device/qjit_device.py frontend/test/pytest/test_verification.py
git commit -m "feat: preserve PPR during device preprocessing"
```

---

### Task 3: Document and integration-test the frontend behavior

**Files:**
- Modify: `frontend/catalyst/passes/builtin_passes.py`
- Modify: `frontend/test/pytest/test_pauli_rot_and_measure.py`

**Interfaces:**
- Consumes: `qp.PPR` when the installed PennyLane version provides PR #10107.
- Produces: frontend evidence that generic Operator2 capture retains PPR and `@to_ppr` emits exactly one `pbc.ppr`.

- [ ] **Step 1: Add version-compatible integration tests**

Add tests guarded with:

```python
@pytest.mark.skipif(not hasattr(qp, "PPR"), reason="PennyLane PPR is not installed")
```

The first compiles a raw `qp.PPR(4, "XY", wires=[0, 1])` to the quantum compilation stage without `to_ppr` and asserts:

```python
assert 'quantum.operator "PPR"' in optimized_ir
assert "pbc.ppr" not in optimized_ir
```

The second applies `@to_ppr` and asserts:

```python
assert 'pbc.ppr ["X", "Y"](8)' in optimized_ir
assert 'quantum.operator "PPR"' not in optimized_ir
assert "quantum.paulirot" not in optimized_ir
```

- [ ] **Step 2: Run the focused integration tests**

Run:

```bash
pytest frontend/test/pytest/test_pauli_rot_and_measure.py -k "ppr_operator" -v
```

Expected with the current pin: SKIPPED with the explicit reason. Expected with PennyLane PR #10107 installed: both tests PASS. The non-skipped MLIR test from Task 1 remains the required proof of conversion behavior.

- [ ] **Step 3: Update `to_ppr` documentation**

Add ``qp.PPR`` beside ``qp.PauliRot`` in the supported operation list in `to_ppr_setup_inputs`.

- [ ] **Step 4: Run focused frontend regression tests**

Run:

```bash
pytest frontend/test/pytest/test_pauli_rot_and_measure.py -v
```

Expected: existing tests PASS and new tests either PASS with PR #10107 or SKIP with the current pin.

- [ ] **Step 5: Commit frontend coverage and documentation**

```bash
git add frontend/catalyst/passes/builtin_passes.py frontend/test/pytest/test_pauli_rot_and_measure.py
git commit -m "test: cover PennyLane PPR frontend lowering"
```

---

### Task 4: Full verification

**Files:**
- Verify only; modify scoped files if a regression reveals a defect.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a regression-tested branch ready for review.

- [ ] **Step 1: Run PBC MLIR tests**

```bash
lit mlir/test/PBC -v
```

Expected: PASS.

- [ ] **Step 2: Run relevant frontend tests**

```bash
pytest frontend/test/pytest/test_pauli_rot_and_measure.py frontend/test/pytest/test_verification.py -v
```

Expected: PASS, except the two explicitly version-gated PPR integration tests may SKIP on the unchanged PennyLane pin.

- [ ] **Step 3: Run formatting and lint checks for changed files**

Use the repository's configured C++ formatter and Python checks on:

```text
mlir/lib/PBC/Transforms/ToPPR.cpp
frontend/catalyst/device/qjit_device.py
frontend/catalyst/passes/builtin_passes.py
frontend/test/pytest/test_verification.py
frontend/test/pytest/test_pauli_rot_and_measure.py
```

Expected: no formatting or lint errors.

- [ ] **Step 4: Inspect the final diff**

```bash
git status --short
git diff origin/main...HEAD --check
```

Expected: only the design, plan, implementation, tests, and documentation are present; `git diff --check` exits successfully.
