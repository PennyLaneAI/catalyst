# Release 0.14.0 (development release)

<h3>New features since last release</h3>

* Added ``catalyst.switch``, a qjit compatible, index-switch style control flow decorator.
  [(#2171)](https://github.com/PennyLaneAI/catalyst/pull/2171)

* Catalyst can now compile circuits that are directly expressed in terms of Pauli product rotation
  (PPR) and Pauli product measurement (PPM) operations: :class:`~.PauliRot` and
  :func:`~.pauli_measure`, respectively. This support enables research and development
  spurred from `A Game of Surface Codes (arXiv1808.02892) <https://arxiv.org/pdf/1808.02892>`_.
  [(#2145)](https://github.com/PennyLaneAI/catalyst/pull/2145)
  [(#2233)](https://github.com/PennyLaneAI/catalyst/pull/2233)

  :class:`~.PauliRot` and :func:`~.pauli_measure` can be manipulated with Catalyst's existing passes
  for PPR-PPM compilation, which includes :func:`catalyst.passes.to_ppr`,
  :func:`catalyst.passes.commute_ppr`, :func:`catalyst.passes.merge_ppr_ppm`,
  :func:`catalyst.passes.ppr_to_ppm`, :func:`catalyst.passes.reduce_t_depth`, and
  :func:`catalyst.passes.ppm_compilation`. For clear and inspectable results, use ``target="mlir"``
  in the ``qjit`` decorator, ensure that PennyLane's program capture is enabled,
  :func:`pennylane.capture.enable`, and call the Catalyst passes from the PennyLane frontend (e.g.,
  ``qml.transforms.ppr_to_ppm`` instead of from ``catalyst.passes.``).

  ```python
  import pennylane as qml
  from functools import partial
  import jax.numpy as jnp

  qml.capture.enable()

  @qjit(target="mlir")
  @partial(qml.transforms.ppm_compilation, decompose_method="auto-corrected")
  @qml.qnode(qml.device("null.qubit", wires=3))
  def circuit():
      # equivalent to a Hadamard gate
      qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
      qml.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
      qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

      ppm = qml.pauli_measure(pauli_word="XYZ", wires=[0, 1, 2])

      # equivalent to a CNOT gate
      qml.PauliRot(jnp.pi / 2, pauli_word="ZX", wires=[0, 1])
      qml.PauliRot(-jnp.pi / 2, pauli_word="Z", wires=[0])
      qml.PauliRot(-jnp.pi / 2, pauli_word="X", wires=[1])

      # equivalent to a T gate
      qml.PauliRot(jnp.pi / 4, pauli_word="Z", wires=0)

      ppm = qml.pauli_measure(pauli_word="YYZ", wires=[0, 2, 1])

      return
  ```

  ```pycon
  >>> print(qml.specs(circuit, level="all")()['resources'])
  {
    'No transforms': ...,
    'Before MLIR Passes (MLIR-0)': ...,
    'ppm-compilation (MLIR-1)': Resources(
      num_wires=6,
      num_gates=14,
      gate_types=defaultdict(<class 'int'>, {'PPM-w3': 2, 'PPM-w2': 4, 'PPM-w1': 4, 'PPR-pi/2-w1': 4}),
      gate_sizes=defaultdict(<class 'int'>, {3: 2, 2: 4, 1: 8}),
      depth=None,
      shots=Shots(total_shots=None, shot_vector=())
    )
  }
  ```

<h3>Improvements 🛠</h3>

* Add an experimental `outline_state_evolution_pass` xDSL pass to `catalyst.python_interface.transforms`,
  which moves all quantum gate operations to a private callable.
  [(#8367)](https://github.com/PennyLaneAI/pennylane/pull/8367)

* A new experimental `split_non_commuting_pass` compiler pass has been added to
  `catalyst.python_interface.transforms`. This pass splits quantum functions that
  measure observables on the same wires into multiple function executions, where
  each execution measures observables on different wires (using the "wires" grouping
  strategy). The original function is replaced with calls to these generated functions,
  and the results are combined appropriately.
  [(#8531)](https://github.com/PennyLaneAI/pennylane/pull/8531)

* Add the `PCPhaseOp` operation to the xDSL Quantum dialect.
  [(#8621)](https://github.com/PennyLaneAI/pennylane/pull/8621)

* Users can now apply xDSL passes without the need to pass the `pass_plugins` argument to
  the `qjit` decorator.
  [(#8572)](https://github.com/PennyLaneAI/pennylane/pull/8572)
  [(#8573)](https://github.com/PennyLaneAI/pennylane/pull/8573)
  [(#2169)](https://github.com/PennyLaneAI/catalyst/pull/2169)
  [(#2183)](https://github.com/PennyLaneAI/catalyst/pull/2183)

* The :meth:`catalyst.python_interface.transforms.convert_to_mbqc_formalism_pass` now
  supports :class:`~xdsl.dialects.scf.IndexSwitchOp` in IR and ignores regions that have no body.
  [(#8632)](https://github.com/PennyLaneAI/pennylane/pull/8632)

* The `convert_to_mbqc_formalism` compilation pass now outlines the operations to represent a gate
  in the MBQC formalism into subroutines in order to reduce the IR size for large programs.
  [(#8619)](https://github.com/PennyLaneAI/pennylane/pull/8619)

* The :meth:`catalyst.python_interface.Compiler.run` method now accepts a string as input,
  which is parsed and transformed with xDSL.
  [(#8587)](https://github.com/PennyLaneAI/pennylane/pull/8587)

* An `is_xdsl_pass` function has been added to the `catalyst.python_interface.pass_api` module.
  This function checks if a pass name corresponds to an xDSL implemented pass.
  [(#8572)](https://github.com/PennyLaneAI/pennylane/pull/8572)

* A new `catalyst.python_interface.utils` submodule has been added, containing general-purpose utilities for
  working with xDSL. This includes a function that extracts the concrete value of scalar, constant SSA values.
  [(#8514)](https://github.com/PennyLaneAI/pennylane/pull/8514)

* A new ``"changed"`` option has been added to the ``keep_intermediate`` parameter of 
  :func:`~.qjit`. This option saves intermediate IR files after each pass,
  but only when the IR is actually modified by the pass.
  [(#2186)](https://github.com/PennyLaneAI/catalyst/pull/2186)

* Intermediate IR files are now organized into subdirectories for each compilation stage
  when using ``keep_intermediate="changed"`` or ``keep_intermediate="pass"``.
  [(#2186)](https://github.com/PennyLaneAI/catalyst/pull/2186)

* Resource tracking now tracks calls to `SetState` and `SetBasisState`, and can report results
  that include `qml.StatePrep` operations.
  [(#2230)](https://github.com/PennyLaneAI/catalyst/pull/2230)

* `qml.PCPhase` can be compiled and executed with capture enabled.
  [(#2226)](https://github.com/PennyLaneAI/catalyst/pull/2226)

* Resource tracking now supports dynamic qubit allocation
  [(#2203)](https://github.com/PennyLaneAI/catalyst/pull/2203)

* Pass instrumentation can be applied to each pass within the `NamedSequenceOp` transform sequence for a qnode.
  [(#1978)](https://github.com/PennyLaneAI/catalyst/pull/1978)

* The new graph-based decomposition framework has Autograph feature parity with PennyLane
  when capture enabled. When compiling with `qml.qjit(autograph=True)`, the decomposition rules
  returned by the graph-based framework are now correctly compiled using Autograph.
  This ensures compatibility and deeper optimization for dynamically generated rules.
  [(#2161)](https://github.com/PennyLaneAI/catalyst/pull/2161)

* The ``decompose-lowering`` MLIR pass now supports ``qml.MultiRZ``
  with an arbitrary number of wires. This decomposition is performed
  at MLIR when both capture and graph-decomposition are enabled.
  [(#2160)](https://github.com/PennyLaneAI/catalyst/pull/2160)

* A new option ``use_nameloc`` has been added to :func:`~.qjit` that embeds variable names
  from Python into the compiler IR, which can make it easier to read when debugging programs.
  [(#2054)](https://github.com/PennyLaneAI/catalyst/pull/2054)

* Passes registered under `qml.transform` can now take in options when used with
  :func:`~.qjit` with program capture enabled.
  [(#2154)](https://github.com/PennyLaneAI/catalyst/pull/2154)

* Pytree inputs can now be used when program capture is enabled.
  [(#2165)](https://github.com/PennyLaneAI/catalyst/pull/2165)

* `qml.grad` and `qml.jacobian` can now be used with `qjit` when program capture is enabled.
  [(#2078)](https://github.com/PennyLaneAI/catalyst/pull/2078)

* xDSL passes are now automatically detected when using the `qjit` decorator.
  This removes the need to pass the `pass_plugins` argument to the `qjit` decorator.
  [(#2169)](https://github.com/PennyLaneAI/catalyst/pull/2169)
  [(#2183)](https://github.com/PennyLaneAI/catalyst/pull/2183)

* The ``mlir_opt`` property now correctly handles xDSL passes by automatically
  detecting when the Python compiler is being used and routing through it appropriately.
  [(#2190)](https://github.com/PennyLaneAI/catalyst/pull/2190)

* Dynamically allocated wires can now be passed into control flow and subroutines.
  [(#2130)](https://github.com/PennyLaneAI/catalyst/pull/2130)

* The `--adjoint-lowering` pass can now handle PPR operations.
  [(#2227)](https://github.com/PennyLaneAI/catalyst/pull/2227)

* Catalyst now supports Pauli product rotations with arbitrary or dynamic angles in the 
  QEC dialect. This will allow :class:`qml.PauliRot` with arbitrary or dynamic angles, 
  angles not known at compile time, to be lowered to the QEC dialect. This is implemented 
  as a new `qec.ppr.arbitrary` operation, which takes a Pauli product and an arbitrary or
  dynamic angle as input. The arbitrary angles are specified as a double in terms of radian.
  [(#2232)](https://github.com/PennyLaneAI/catalyst/pull/2232)
  [(#2233)](https://github.com/PennyLaneAI/catalyst/pull/2233)

  For example:
  ```mlir
  %const = arith.constant 0.124 : f64
  %1:2 = qec.ppr.arbitrary ["X", "Z"](%const) %q1, %q2 : !quantum.bit, !quantum.bit
  %2:2 = qec.ppr.arbitrary ["X", "Z"](%const) %1#0, %1#1 cond(%c0) : !quantum.bit, !quantum.bit
  ```

<h3>Breaking changes 💔</h3>

* The MLIR pipeline ``enforce-runtime-invariants-pipeline`` has been renamed to
  ``quantum-compilation-pipeline`` and the old ``quantum-compilation-pipeline`` has been renamed to
  ``gradient-lowering-pipeline``. Users who referenced these pipeline names directly would need to 
  update their code to use the new names.
  [(#2186)](https://github.com/PennyLaneAI/catalyst/pull/2186)

  * The ``pipeline``  and "passes" postfixes in the compilation stage names have been changed to ``stage``.
  [(#2230)](https://github.com/PennyLaneAI/catalyst/pull/2230)

* The plxpr transform `pl_map_wires` has been removed along with its test.
  [(#2220)](https://github.com/PennyLaneAI/catalyst/pull/2220)

* The JAX version used by Catalyst has been updated to 0.7.0.
  [(#2131)](https://github.com/PennyLaneAI/catalyst/pull/2131)

* (Compiler integrators only) The versions of LLVM/Enzyme/stablehlo used by Catalyst have been
  updated. Enzyme now targets `v0.0.203` with the build target `EnzymeStatic-22`, and the nanobind
  requirement for the latest LLVM has been updated to version 2.9.
  [(#2122)](https://github.com/PennyLaneAI/catalyst/pull/2122)
  [(#2174)](https://github.com/PennyLaneAI/catalyst/pull/2174)
  [(#2175)](https://github.com/PennyLaneAI/catalyst/pull/2175)
  [(#2181)](https://github.com/PennyLaneAI/catalyst/pull/2181)

  - The LLVM version has been updated to
  [commit 113f01a](https://github.com/llvm/llvm-project/tree/113f01aa82d055410f22a9d03b3468fa68600589).
  - The stablehlo version has been updated to
  [commit 0a4440a](https://github.com/openxla/stablehlo/commit/0a4440a5c8de45c4f9649bf3eb4913bf3f97da0d).
  - The Enzyme version has been updated to
  [v0.0.203](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.203).

* The pass `remove-chained-self-inverse` has been renamed to `cancel-inverses`, to better
  conform with the name of the corresponding transform in PennyLane.
  [(#2201)](https://github.com/PennyLaneAI/catalyst/pull/2201)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* The experimental xDSL :func:`~catalyst.python_interface.transforms.measurements_from_samples_pass`
  pass has been updated to support `shots` defined by an `arith.constant` operation.
  [(#8460)](https://github.com/PennyLaneAI/pennylane/pull/8460)

* The experimental xDSL :func:`~catalyst.python_interface.transforms.diagonalize_measurements`
  pass has been updated to fix a bug that included the wrong SSA value for final qubit insertion
  and deallocation at the end of the circuit. A clear error is now also raised when there are
  observables with overlapping wires.
  [(#8383)](https://github.com/PennyLaneAI/pennylane/pull/8383)

* Fixes a bug in the constructor of the xDSL Quantum dialect's `QubitUnitaryOp` that
  prevented an instance from being constructed.
  [(#8456)](https://github.com/PennyLaneAI/pennylane/pull/8456)

* Fixes an issue where a heap-to-stack allocation conversion pass was causing SIGSEGV issues
  during program execution at runtime.
  [(#2172)](https://github.com/PennyLaneAI/catalyst/pull/2172)

* Fixes the issue with capturing unutilized abstracted adjoint and controlled rules
  by the graph in the new decomposition framework.
  [(#2160)](https://github.com/PennyLaneAI/catalyst/pull/2160)

* Fixes the translation of plxpr control flow for edge cases where the `consts` were being
  reordered.
  [(#2128)](https://github.com/PennyLaneAI/catalyst/pull/2128)
  [(#2133)](https://github.com/PennyLaneAI/catalyst/pull/2133)

* Fixes the translation of `QubitUnitary` and `GlobalPhase` ops
  when they are modified by adjoint or control.
  [(##2158)](https://github.com/PennyLaneAI/catalyst/pull/2158)

* Fixes the translation of a workflow with different transforms applied to different qnodes.
  [(#2167)](https://github.com/PennyLaneAI/catalyst/pull/2167)

* Fix canonicalization of eliminating redundant `quantum.insert` and `quantum.extract` pairs.
  When extracting a qubit immediately after inserting it at the same index, the operations can
  be cancelled out while properly updating remaining uses of the register.
  [(#2162)](https://github.com/PennyLaneAI/catalyst/pull/2162)
  For an example:
  ```mlir
  // Before canonicalization
  %1 = quantum.insert %0[%idx], %qubit1 : !quantum.reg, !quantum.bit
  %2 = quantum.extract %1[%idx] : !quantum.reg -> !quantum.bit
  ...
  %3 = quantum.insert %1[%i0], %qubit2 : !quantum.reg, !quantum.bit
  %4 = quantum.extract %1[%i1] : !quantum.reg -> !quantum.bit
  // ... use %1
  // ... use %4

  // After canonicalization
  // %2 directly uses %qubit1
  // %3 and %4 updated to use %0 instead of %1
  %3 = quantum.insert %0[%i0], %qubit2 : !quantum.reg, !quantum.bit
  %4 = quantum.extract %0[%i1] : !quantum.reg -> !quantum.bit
  // ... use %qubit1
  // ... use %4
  ```

* Fixes :func:`~.passes.commute_ppr` and :func:`~.passes.merge_ppr_ppm` incorrectly
  moving nullary operations. This also improves the compilation time by reducing
  the sort function, by explicitly passing the operations that need to be sorted.
  [(#2200)](https://github.com/PennyLaneAI/catalyst/pull/2200)

* The pass pipeline is correctly registered to the transform named sequence of the
  one-shot qnode when `one-shot` mcm method is used.
  [(#2198)](https://github.com/PennyLaneAI/catalyst/pull/2198)

* Fixed a bug where `qml.StatePrep` and `qml.BasisState` might be pushed after other
  gates, overwriting their effects.
  [(#2239)](https://github.com/PennyLaneAI/catalyst/pull/2239)

<h3>Internal changes ⚙️</h3>

* A new `catalyst.python_interface.inspection.mlir_specs` method has been added to facilitate
  PennyLane's new pass-by-pass specs feature. This function returns information gathered by parsing
  the xDSL generated by a given QJIT object, such as gate counts, measurements, or qubit allocations.
  [(#2238)](https://github.com/PennyLaneAI/catalyst/pull/2238)

* The `catalyst.python_interface.visualization` module has been renamed to
  `catalyst.python_interface.inspection`, and various utility functions within this module
  have been streamlined.
  [(#2237)](https://github.com/PennyLaneAI/catalyst/pull/2237)

* Migrated the `pennylane.compiler.python_compiler` submodule from PennyLane to Catalyst.
  It is now accessible as `catalyst.python_interface`.
  [(#2199)](https://github.com/PennyLaneAI/catalyst/pull/2199)

* Resource tracking now writes out at device destruction time instead of qubit deallocation
  time. The written resources will be the total amount of resources collected throughout the
  lifetime of the execution. For executions that split work between multiple functions,
  e.g. with the `split-non-commuting` pass, this ensures that resource tracking outputs
  the total resources used for all splits.
  [(#2219)](https://github.com/PennyLaneAI/catalyst/pull/2219)

* Replaces the deprecated `shape_dtype_to_ir_type` function with the `RankedTensorType.get` method.
  [(#2159)](https://github.com/PennyLaneAI/catalyst/pull/2159)

* Updates to PennyLane's use of a single transform primitive with a `transform` kwarg.
  [(#2177)](https://github.com/PennyLaneAI/catalyst/pull/2177)

* The pytest tests are now run with `strict=True` by default.
  [(#2180)](https://github.com/PennyLaneAI/catalyst/pull/2180)

* Refactor Catalyst pass registering so that it's no longer necessary to manually add new
  passes at `registerAllCatalystPasses`.
  [(#1984)](https://github.com/PennyLaneAI/catalyst/pull/1984)

* Split `from_plxpr.py` into two files.
  [(#2142)](https://github.com/PennyLaneAI/catalyst/pull/2142)

* Re-work `DataView` to avoid an axis of size 0 possibly triggering a segfault via an underflow
  error, as discovered in
  [this comment](https://github.com/PennyLaneAI/catalyst/pull/1598#issuecomment-2779178046).
  [(#1621)](https://github.com/PennyLaneAI/catalyst/pull/2164)

* Decouple the ion dialect from the quantum dialect to support the new RTIO compilation flow.
  The ion dialect now uses its own `!ion.qubit` type instead of depending on `!quantum.bit`.
  Conversion between qubits of quantum and ion dialects is handled via unrealized conversion casts.
  [(#2163)](https://github.com/PennyLaneAI/catalyst/pull/2163)

  For an example, quantum qubits are converted to ion qubits as follows:
  ```mlir
  %qreg = quantum.alloc(1) : !quantum.reg
  %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit

  // Convert quantum.bit to ion.qubit
  %ion_qubit_0 = builtin.unrealized_conversion_cast %q0 : !quantum.bit to !ion.qubit

  // Use in ion dialect operations
  %pp = ion.parallelprotocol(%ion_qubit_0) : !ion.qubit {
    ^bb0(%arg1: !ion.qubit):
      // ... ion operations ...
  }
  ```

  * Added support for `ppr-to-ppm` as an individual MLIR pass and python binding
  for the qec dialect.
  [(#2189)](https://github.com/PennyLaneAI/catalyst/pull/2189)

  * Added a canonicalization pattern for `qec.ppr` to remove any PPRs consisting only
  of identities.
  [(#2192)](https://github.com/PennyLaneAI/catalyst/pull/2192)

  * Renamed `annotate-function` pass to `annotate-invalid-gradient-functions` and move it to the 
  gradient dialect and the `lower-gradients` compilation stage.
  [(#2241)](https://github.com/PennyLaneAI/catalyst/pull/2241)

  * Added support for PPRs to the :func:`~.passes.merge_rotations` pass to merge PPRs with
  equivalent angles, and cancelling of PPRs with opposite angles, or angles
  that sum to identity. Also supports conditions on PPRs, merging when conditions are 
  identical and not merging otherwise.
  [(#2224)](https://github.com/PennyLaneAI/catalyst/pull/2224)	
  [(#2245)](https://github.com/PennyLaneAI/catalyst/pull/2245)
  [(#2254)](https://github.com/PennyLaneAI/catalyst/pull/2254)

<h3>Documentation 📝</h3>

* Added a "Unified Compiler Cookbook" RST file, along with tutorials, to `catalyst.python_interface.doc`,
  which provides a quickstart guide for getting started with xDSL and its integration with PennyLane and
  Catalyst.
  [(#8571)](https://github.com/PennyLaneAI/pennylane/pull/8571)

* A typo in the code example for :func:`~.passes.ppr_to_ppm` has been corrected.
  [(#2136)](https://github.com/PennyLaneAI/catalyst/pull/2136)

* Fix `catalyst.qjit` and `catalyst.CompileOptions` docs rendering.
  [(#2156)](https://github.com/PennyLaneAI/catalyst/pull/2156)

* Update `MLIR Plugins` documentation stating that plugins require adding passes via
  `--pass-pipeline`.
  [(#2168)](https://github.com/PennyLaneAI/catalyst/pull/2168)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Yushao Chen,
Sengthai Heng,
Jeffrey Kam,
Christina Lee,
Mehrdad Malekmohammadi,
River McCubbin,
Lee J. O'Riordan,
Roberto Turrado,
Paul Haochen Wang,
Jake Zaia,
Hongsheng Zheng.
