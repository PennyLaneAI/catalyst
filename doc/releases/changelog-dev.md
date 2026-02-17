# Release 0.15.0 (development release)

<h3>New features since last release</h3>

* Executing circuits that are compiled with :func:`pennylane.transforms.to_ppr`, 
  :func:`pennylane.transforms.commute_ppr`, :func:`pennylane.transforms.ppr_to_ppm`, 
  :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.reduce_t_depth`,
  and :func:`pennylane.transforms.decompose_arbitrary_ppr` is now possible with the `lightning.qubit` device and
  with program capture enabled (:func:`pennylane.capture.enable`).
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)
  [(#2424)](https://github.com/PennyLaneAI/catalyst/pull/2424)
  [(#2443)](https://github.com/PennyLaneAI/catalyst/pull/2443)
  
  Previously, circuits compiled with these transforms were only inspectable via 
  :func:`pennylane.specs` and :func:`catalyst.draw`. Now, such circuits can be executed:

  ```python
  import pennylane as qml

  @qml.qjit(capture=True)
  @qml.transforms.decompose_arbitrary_ppr
  @qml.transforms.to_ppr
  @qml.qnode(qml.device("lightning.qubit", wires=3))
  def circuit():
      qml.PauliRot(0.123, pauli_word="XXY", wires=[0, 1, 2])
      qml.pauli_measure("XYZ", wires=[0, 1, 2])
      return qml.probs([0, 1])
  ```

  ```
  >>> print(circuit())
  [0.5 0.  0.  0.5]
  ```

* Added `capture` keyword argument to the `@qjit` decorator for per-function control over
  PennyLane's program capture frontend. This allows selective use of the new capture-based
  compilation pathway without affecting the global `qml.capture.enabled()` state. The parameter
  accepts `"global"` (default, defer to global state), `True` (force capture on), or `False`
  (force capture off). This enables safe testing and gradual migration to the capture system.
  [(#2457)](https://github.com/PennyLaneAI/catalyst/pull/2457)

* OQD (Open Quantum Design) end-to-end pipeline is added to Catalyst.
  The pipeline supports compilation to LLVM IR using the `QJIT` constructor with `link=False`, enabling integration with ARTIQ's cross-compilation toolchain. The generated LLVM IR can be used with the internal `compile_to_artiq()` function from the third-party OQD repository to produce ARTIQ binaries.
  [(#2299)](https://github.com/PennyLaneAI/catalyst/pull/2299)

  see `frontend/test/test_oqd/oqd/test_oqd_artiq_llvmir.py` for more details.
  Note: This PR only covers LLVM IR generation; the `compile_to_artiq` function itself is not included.

  For example:

  ```python
  import os
  import numpy as np
  import pennylane as qml

  from catalyst import qjit
  from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline

  OQD_PIPELINES = OQDDevicePipeline(
      os.path.join("calibration_data", "device.toml"),
      os.path.join("calibration_data", "qubit.toml"),
      os.path.join("calibration_data", "gate.toml"),
      os.path.join("device_db", "device_db.json"),
  )

  oqd_dev = OQDDevice(
      backend="default",
      shots=4,
      wires=1
  )
  qml.capture.enable()

  # Compile to LLVM IR only
  @qml.qnode(oqd_dev)
  def circuit():
      x = np.pi / 2
      qml.RX(x, wires=0)
      return qml.counts(wires=0)

  compiled_circuit = QJIT(circuit, CompileOptions(link=False, pipelines=OQD_PIPELINES))

  # Compile to ARTIQ ELF
  artiq_config = {
      "kernel_ld": "/path/to/kernel.ld",
      "llc_path": "/path/to/llc",
      "lld_path": "/path/to/ld.lld",
  }

  output_elf_path = compile_to_artiq(compiled_circuit, artiq_config)
  # Output:
  # LLVM IR file written to: /path/to/circuit.ll
  # [ARTIQ] Generated ELF: /path/to/circuit.elf
  ```

<h3>Improvements 🛠</h3>

* Catalyst with program capture can now be used with the new `qml.templates.Subroutine` class and the associated
  `qml.capture.subroutine` upstreamed from `catalyst.jax_primitives.subroutine`.
  [(#2396)](https://github.com/PennyLaneAI/catalyst/pull/2396)

* The PPR/PPM lowering passes (`lower-pbc-init-ops`, `unroll-conditional-ppr-ppm`) are now run
  as part of the main quantum compilation pipeline. When using `to-ppr` and `ppr-to-ppm` transforms,
  these passes are applied automatically during compilation; we no longer need to stack them
  explicitly.
  [(#2460)](https://github.com/PennyLaneAI/catalyst/pull/2460)

* `null.qubit` resource tracking is now able to track measurements and observables. This output
  is also reflected in `qml.specs`.
  [(#2446)](https://github.com/PennyLaneAI/catalyst/pull/2446)

* The default mcm_method for the finite-shots setting (dynamic one-shot) no longer silently falls
  back to single-branch statistics in most cases. Instead, an error message is raised pointing out
  alternatives, like explicitly selecting single-branch statistics.
  [(#2398)](https://github.com/PennyLaneAI/catalyst/pull/2398)

  Importantly, single-branch statistics only explores one branch of the MCM decision tree, meaning
  program outputs are typically probabilistic and statistics produced by measurement processes are
  conditional on the selected decision tree path.

* Two new verifiers were added to the `quantum.paulirot` operation. They verify that the Pauli word
  length and the number of qubit operands are the same, and that all of the Pauli words are legal.
  [(#2405)](https://github.com/PennyLaneAI/catalyst/pull/2405)

* `qml.vjp`  and `qml.jvp` can now be used with Catalyst and program capture.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)
  [(#2316)](https://github.com/PennyLaneAI/catalyst/pull/2316)

* The `measurements_from_samples` pass no longer results in `nan`s and cryptic error messages when
  `shots` aren't set. Instead, an informative error message is raised.
  [(#2456)](https://github.com/PennyLaneAI/catalyst/pull/2456)

* Graph decomposition with qjit now accepts `num_work_wires`, and lowers and decomposes correctly
  with the `decompose-lowering` pass and with `qp.transforms.decompose`.
  [(#2470)](https://github.com/PennyLaneAI/catalyst/pull/2470)

<h3>Breaking changes 💔</h3>

* `catalyst.jax_primitives.subroutine` has been moved to `qml.capture.subroutine`.
  [(#2396)](https://github.com/PennyLaneAI/catalyst/pull/2396)

* (Compiler integrators only) The versions of StableHLO/LLVM/Enzyme used by Catalyst have been updated.
  [(#2415)](https://github.com/PennyLaneAI/catalyst/pull/2415)
  [(#2416)](https://github.com/PennyLaneAI/catalyst/pull/2416)
  [(#2444)](https://github.com/PennyLaneAI/catalyst/pull/2444)
  [(#2445)](https://github.com/PennyLaneAI/catalyst/pull/2445)
  [(#2478)](https://github.com/PennyLaneAI/catalyst/pull/2478)

  * The StableHLO version has been updated to
  [v1.13.7](https://github.com/openxla/stablehlo/tree/v1.13.7).
  * The LLVM version has been updated to
  [commit 8f26458](https://github.com/llvm/llvm-project/tree/8f264586d7521b0e305ca7bb78825aa3382ffef7).
  * The Enzyme version has been updated to
  [v0.0.238](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.238).

* When an integer argnums is provided to `catalyst.vjp`, a singleton dimension is now squeezed
  out. This brings the behaviour in line with that of `grad` and `jacobian`.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#2407)](https://github.com/PennyLaneAI/catalyst/pull/2407)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix `CATALYST_XDSL_UNIVERSE` to correctly define the available dialects and transforms, allowing
  tools like `xdsl-opt` to work with Catalyst's custom Python dialects.
  [(#2471)](https://github.com/PennyLaneAI/catalyst/pull/2471)

* Fix a bug with the xDSL `ParitySynth` pass that caused failure when the QNode being transformed
  contained operations with regions.
  [(#2408)](https://github.com/PennyLaneAI/catalyst/pull/2408)

* Fix `replace_ir` for certain stages when used with gradients.
  [(#2436)](https://github.com/PennyLaneAI/catalyst/pull/2436)

* Restore the ability to differentiate multiple (expectation value) QNode results with the
  adjoint-differentiation method.
  [(#2428)](https://github.com/PennyLaneAI/catalyst/pull/2428)

* Fixed the angle conversion when lowering `pbc.ppr` and `pbc.ppr.arbitrary` operations to
  `__catalyst__qis__PauliRot` runtime calls. The PPR rotation angle is now correctly multiplied
  by 2 to match the PauliRot convention (`PauliRot(φ) == PPR(φ/2)`).
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)

* Fixed the `catalyst` CLI tool silently listening to stdin when run without an input file, even when given flags like `--list-passes` that should override this behaviour.
  [(2447)](https://github.com/PennyLaneAI/catalyst/pull/2447)

* Fixing incorrect lowering of PPM into CAPI calls when the PPM is in the negative basis.
  [(#2422)](https://github.com/PennyLaneAI/catalyst/pull/2422)

* Fixed the GlobalPhase discrepancies when executing gridsynth in the PPR basis.
  [(#2433)](https://github.com/PennyLaneAI/catalyst/pull/2433)

* Fixed incorrect decomposition of negative PPR (Pauli Product Rotation) operations in the
  `decompose-clifford-ppr` and `decompose-non-clifford-ppr` passes. The rotation sign is now
  correctly flipped when decomposing negative rotation kinds (e.g., `-π/4` from adjoint gates
  like `T†` or `S†`) to PPM (Pauli Product Measurement) operations.
  [(#2454)](https://github.com/PennyLaneAI/catalyst/pull/2454)

* Fixed incorrect global phase when lowering CNOT gates into PPR/PPM operations.
  [(#2459)](https://github.com/PennyLaneAI/catalyst/pull/2459)

<h3>Internal changes ⚙️</h3>

* The QEC (Quantum Error Correction) dialect has been renamed to PBC (Pauli-Based Computation)
  across the entire codebase. This includes the MLIR dialect (`pbc.*` -> `pbc.*`), C++ namespaces
  (`catalyst::pbc` -> `catalyst::pbc`), Python bindings, compiler passes (e.g.,
  `lower-pbc-init-ops` -> `lower-pbc-init-ops`, `convert-pbc-to-llvm` -> `convert-pbc-to-llvm`),
  qubit type (`!quantum.bit<pbc>` -> `!quantum.bit<pbc>`), and all associated file and directory
  names. The rename better reflects the dialect's purpose as a representation for Pauli-Based
  Computation rather than general quantum error correction.
  [(#2482)](https://github.com/PennyLaneAI/catalyst/pull/2482)

* Updated the integration tests for `qp.specs` to get coverage for new features
  [(#2448)](https://github.com/PennyLaneAI/catalyst/pull/2448)

* The xDSL :class:`~catalyst.python_interface.Quantum` dialect has been split into multiple files
  to structure operations and attributes more concretely.
  [(#2434)](https://github.com/PennyLaneAI/catalyst/pull/2434)

* `catalyst.python_interface.xdsl_universe.XDSL_UNIVERSE` has been renamed to `CATALYST_XDSL_UNIVERSE`.
  [(#2435)](https://github.com/PennyLaneAI/catalyst/pull/2435)

* The private helper `_extract_passes` of `qfunc.py` uses `BoundTransform.tape_transform`
  instead of the deprecated `BoundTransform.transform`.
  `jax_tracer.py` and `tracing.py` also updated accordingly.
  [(#2440)](https://github.com/PennyLaneAI/catalyst/pull/2440)

* Autograph is no longer applied to decomposition rules based on whether it's applied to the workflow itself.
  Operator developers now need to manually apply autograph to decomposition rules when needed.
  [(#2421)](https://github.com/PennyLaneAI/catalyst/pull/2421)

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

* Added lowering of `pbc.ppm`, `pbc.ppr`, and `quantum.paulirot` to the runtime CAPI and QuantumDevice C++ API.
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)

* Added LLVM conversion patterns to lower PBC dialect operations to their corresponding runtime
  CAPI calls.
  This includes `pbc.ppr` and `pbc.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `pbc.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of PBC
  operations through the Catalyst runtime.
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)

* A new compiler pass `unroll-conditional-ppr-ppm` for lowering conditional PPR and PPMs
  into normal PPR and PPMs with SCF dialect to support runtime execution.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

* Increased format size for the `--mlir-timing` flag, displaying more decimals for better timing precision.
  [(#2423)](https://github.com/PennyLaneAI/catalyst/pull/2423)

* Added global phase tracking to the `to-ppr` compiler pass. When converting quantum gates to
  Pauli Product Rotations (PPR), the pass now emits `quantum.gphase` operations to preserve
  global phase correctness.
  [(#2419)](https://github.com/PennyLaneAI/catalyst/pull/2419)

* New qubit-type specializations have been added to Catalyst's MLIR type system. These new qubit
  types include `!quantum.bit<logical>`, `!quantum.bit<pbc>` and `!quantum.bit<physical>`. The
  original `!quantum.bit` type continues to be supported and used as the default qubit type.
  [(#2369)](https://github.com/PennyLaneAI/catalyst/pull/2369)

  The corresponding register-type specializations have also been added.
  [(#2431)](https://github.com/PennyLaneAI/catalyst/pull/2431)

* The upstream MLIR `Test` dialect is now available via the `catalyst` command line tool.
  [(#2417)](https://github.com/PennyLaneAI/catalyst/pull/2417)

* Removing some previously-added guardrails that were in place due to a bug in dynamic allocation 
  that is now fixed.
  [(#2427)](https://github.com/PennyLaneAI/catalyst/pull/2427)

* A new compiler pass `lower-qec-init-ops` has been added to lower QEC initialization operations
  to Quantum dialect operations. This pass converts `qec.prepare` to `quantum.custom` and
  `qec.fabricate` to `quantum.alloc_qb` + `quantum.custom`, enabling runtime execution of
  QEC state preparation operations.
  [(#2424)](https://github.com/PennyLaneAI/catalyst/pull/2424)

* A new compiler pass `split-to-single-terms` has been added for QNode functions containing
  Hamiltonian expectation values. It facilitates execution on devices that don't natively support expectation values of sums of observables by splitting them into individual leaf observable expvals.
  [(#2441)](https://github.com/PennyLaneAI/catalyst/pull/2441)

  Consider the following example:

  ```python
  import pennylane as qml
  from catalyst import qjit
  from catalyst.passes import apply_pass

  @qjit
  @apply_pass("split-to-single-terms")
  @qml.qnode(qml.device("lightning.qubit", wires=3))
  def circuit():
      # Hamiltonian H = Z(0) @ X(1) + 2*Y(2)
      return qml.expval(qml.Z(0) @ qml.X(1) + 2 * qml.Y(2))
  ```

  The pass transforms the function by splitting the Hamiltonian into individual observables:

  **Before:**

  ```mlir
  func @circ1(%arg0) -> (tensor<f64>) {qnode} {
      // ... quantum ops ...
      // Z(0) @ X(1)
      %obs0 = quantum.namedobs %qubit0[ PauliZ] : !quantum.obs
      %obs1 = quantum.namedobs %qubit1[ PauliX] : !quantum.obs
      %T0 = quantum.tensor %obs0, %obs1 : !quantum.obs

      // Y(2)
      %obs2 = quantum.namedobs %qubit2[ PauliY] : !quantum.obs
      %H0 = quantum.hamiltonian(%8 : tensor<1xf64>) %obs2 : !quantum.obs

      %H = quantum.hamiltonian(%coeffs_2xf64) %T0, %H0 : !quantum.obs
      %result = quantum.expval %H : f64   // H = c_0 * (Z @ X) + c_1 * Y

      // ... to tensor ...
      %tensor_result = tensor.from_elements %result : tensor<f64>
      return %tensor_result
  }
  ```

  **After:**

  ```mlir
  func @circ1.quantum() -> (tensor<f64>, tensor<f64>) {qnode} {
      // ... quantum ops ...
      %expval0 = quantum.expval %T0 : f64
      %expval1 = quantum.expval %obs2 : f64

      // ... to tensor ...
      %tensor0 = tensor.from_elements %expval0 : tensor<f64>
      %tensor1 = tensor.from_elements %expval1 : tensor<f64>
      return %tensor0, %tensor1
  }
  func @circ1(%arg0) -> (tensor<f64>, tensor<f64>) {
      // ... setup ...
      %call:2 = call @circ1.quantum()

      // Extract coefficients and compute weighted sum
      %result = c0 * %call#0 + c1 * %call#1
      return %result
  }
  ```

<h3>Documentation 📝</h3>

* Updated the Unified Compiler Cookbook to be compatible with the latest versions of PennyLane and Catalyst.
  [(#2406)](https://github.com/PennyLaneAI/catalyst/pull/2406)

* Updated the changelog and builtin_passes.py to link to <https://pennylane.ai/compilation/pauli-based-computation> instead.
  [(#2409)](https://github.com/PennyLaneAI/catalyst/pull/2409)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Ali Asadi,
Joey Carter,
Yushao Chen,
Lillian Frederiksen,
Sengthai Heng,
David Ittah,
Jeffrey Kam,
Mehrdad Malekmohammadi,
River McCubbin,
Mudit Pandey,
Andrija Paurevic,
David D.W. Ren,
Paul Haochen Wang,
Jake Zaia,
Hongsheng Zheng.
