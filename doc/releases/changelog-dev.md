# Release 0.10.0-dev (development release)

<h3>New features since last release</h3>

* Catalyst can now load local MLIR plugins from python. Including support for `entry_points`.
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)
  [(#1361)](https://github.com/PennyLaneAI/catalyst/pull/1361)
  [(#1370)](https://github.com/PennyLaneAI/catalyst/pull/1370)

<h3>Improvements 🛠</h3>

* Lightning runtime shot-measurement support for Hermitian observables.
  [(#451)](https://github.com/PennyLaneAI/catalyst/pull/451)

* Replace pybind11 with nanobind for C++/Python bindings in the frontend and in the runtime.
  [(#1173)](https://github.com/PennyLaneAI/catalyst/pull/1173)
  [(#1293)](https://github.com/PennyLaneAI/catalyst/pull/1293)
  [(#1391)](https://github.com/PennyLaneAI/catalyst/pull/1391)

  Nanobind has been developed as a natural successor to the pybind11 library and offers a number of
  [advantages](https://nanobind.readthedocs.io/en/latest/why.html#major-additions), in particular,
  its ability to target Python's [stable ABI interface](https://docs.python.org/3/c-api/stable.html)
  starting with Python 3.12.

* Catalyst now uses the new compiler API to compile quantum code from the python frontend.
  Frontend no longer uses pybind11 to connect to the compiler. Instead, it uses subprocess instead.
  [(#1285)](https://github.com/PennyLaneAI/catalyst/pull/1285)

* Add a MLIR decomposition for the gate set {"T", "S", "Z", "Hadamard", "RZ", "PhaseShift", "CNOT"}
  to the gate set {RX, RY, MS}. It is useful for trapped ion devices. It can be used thanks to
  `quantum-opt --ions-decomposition`.
  [(#1226)](https://github.com/PennyLaneAI/catalyst/pull/1226)

* qml.CosineWindow is now compatible with QJIT.
  [(#1166)](https://github.com/PennyLaneAI/catalyst/pull/1166)

* All PennyLane templates are tested for QJIT compatibility.
  [(#1161)](https://github.com/PennyLaneAI/catalyst/pull/1161)

* Decouple Python from the Runtime by using the Python Global Interpreter Lock (GIL) instead of
  custom mutexes.
  [(#624)](https://github.com/PennyLaneAI/catalyst/pull/624)

  In addition, executables created using :func:`~.debug.compile_executable` no longer require
  linking against Python shared libraries after decoupling Python from the Runtime C-API.
  [(#1305)](https://github.com/PennyLaneAI/catalyst/pull/1305)

* Improves the readability of conditional passes in pipelines
  [(#1194)](https://github.com/PennyLaneAI/catalyst/pull/1194)

* Cleans up the output of compiler instrumentation.
  [(#1343)](https://github.com/PennyLaneAI/catalyst/pull/1343)

* Generate stable ABI wheels for Python 3.12 and up.
  [(#1357)](https://github.com/PennyLaneAI/catalyst/pull/1357)
  [(#1385)](https://github.com/PennyLaneAI/catalyst/pull/1385)

* A new circuit optimization pass, `--disentangle-CNOT`, is available.
  [(#1154)](https://github.com/PennyLaneAI/catalyst/pull/1154)

  The pass disentangles CNOT gates whenever possible, e.g. when the control bit
  is known to be in |0>, the pass removes the CNOT. The pass uses a finite state
  machine to propagate simple one-qubit states, in order to determine
  the input states to the CNOT.

  The algorithm is taken from [Relaxed Peephole Optimization: A Novel Compiler Optimization for Quantum Circuits, by Ji Liu, Luciano Bello, and Huiyang Zhou](https://arxiv.org/abs/2012.07711).

* A new circuit optimization pass, `--disentangle-SWAP`, is available.
  [(#1297)](https://github.com/PennyLaneAI/catalyst/pull/1297)

  The pass disentangles SWAP gates whenever possible by using a finite state
  machine to propagate simple one-qubit states, similar to the `--disentangle-CNOT` pass.

  The algorithm is taken from [Relaxed Peephole Optimization: A Novel Compiler Optimization for Quantum Circuits, by Ji Liu, Luciano Bello, and Huiyang Zhou](https://arxiv.org/abs/2012.07711).

<h3>Breaking changes 💔</h3>

* The `sample` and `counts` measurement primitives now support dynamic shot values across catalyst, although at the PennyLane side, the device shots still is constrained to a static integer literal.

  To support this, `SampleOp` and `CountsOp` in mlir no longer carry the shots attribute, since integer attributes are tied to literal values and must be static.

  `DeviceInitOp` now takes in an optional SSA argument for shots, and the device init runtime CAPI will take in this SSA shots value as an argument and set it as the device shots.
  The sample and counts runtime CAPI functions no longer take in the shots argument and will retrieve shots from the device.

  Correspondingly, the device C++ interface should no longer parse the `DeviceInitOp`'s attributes dictionary for the shots.
  For now we still keep the shots as an attribute so device implementors can have time to migrate, but we will remove shots from the attribute dictionary in the next release.

  [(#1170)](https://github.com/PennyLaneAI/catalyst/pull/1170)
  [(#1310)](https://github.com/PennyLaneAI/catalyst/pull/1310)

* The `toml` module has been migrated to PennyLane with an updated schema for declaring device
  capabilities. Devices with TOML files using `schema = 2` will not be compatible with the latest
  Catalyst. See [Custom Devices](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html)
  for updated instructions on integrating your device with Catalyst and PennyLane
  [(#1275)](https://github.com/PennyLaneAI/catalyst/pull/1275)

* Handling for the legacy operator arithmetic (the `Hamiltonian` and `Tensor` classes in PennyLane)
  is removed.
  [(#1308)](https://github.com/PennyLaneAI/catalyst/pull/1308)

<h3>Bug fixes 🐛</h3>

* Fix bug introduced in 0.8 that breaks nested invocations of `qml.adjoint` and `qml.ctrl`.
  [(#1301)](https://github.com/PennyLaneAI/catalyst/issues/1301)

* Fix a bug in `debug.compile_executable` which would generate incorrect stride information for
  array arguments of the function, in particular when non-64bit datatypes are used.
  [(#1338)](https://github.com/PennyLaneAI/catalyst/pull/1338)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Catalyst no longer depends on or pins the `scipy` package, instead OpenBLAS is sourced directly
  from [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/) or Accelerate is used.
  [(#1322)](https://github.com/PennyLaneAI/catalyst/pull/1322)
  [(#1328)](https://github.com/PennyLaneAI/catalyst/pull/1328)

* The `QuantumExtension` module (previously implemented with pybind11) has been removed. This module
  was not included in the distributed wheels and has been deprecated to align with our adoption of
  Python's stable ABI, which pybind11 does not support.
  [(#1187)](https://github.com/PennyLaneAI/catalyst/pull/1187)

* Remove Lightning Qubit Dynamic plugin from Catalyst.
  [(#1227)](https://github.com/PennyLaneAI/catalyst/pull/1227)
  [(#1307)](https://github.com/PennyLaneAI/catalyst/pull/1307)
  [(#1312)](https://github.com/PennyLaneAI/catalyst/pull/1312)

* `catalyst-cli` and `quantum-opt` are compiled with `default` visibility, which allows for MLIR plugins to work.
  [(#1287)](https://github.com/PennyLaneAI/catalyst/pull/1287)

* Sink patching of autograph's allowlist.
  [(#1332)](https://github.com/PennyLaneAI/catalyst/pull/1332)
  [(#1337)](https://github.com/PennyLaneAI/catalyst/pull/1337)

* Each qnode now has its own transformation schedule.
  Instead of relying on the name of the qnode, each qnode now has a transformation module,
  which denotes the transformation schedule, embedded in its MLIR representation.
  [(#1323)](https://github.com/PennyLaneAI/catalyst/pull/1323)

* The `apply_registered_pass_p` primitive is removed. The API for scheduling passes
  to run using the transform dialect has been refactored. In particular,
  passes are appended to a tuple as they are being registered and they will
  be run in order. If there are no local passes, the global `pass_pipeline` is
  scheduled. Furthermore, this commit also reworks the caching mechanism for
  primitives, which is important as qnodes and functions are primitives and
  now that we can apply passes to them, they are distinct based on which
  passes have been scheduled to run on them.
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)

* Replace Python C-API calls with Stable ABI calls.
  [(#1354)](https://github.com/PennyLaneAI/catalyst/pull/1354)

* A framework for loading and interacting with databases containing hardware information and
  calibration data for Open Quantum Design (OQD) trapped-ion quantum devices has been added.
  [(#1348)](https://github.com/PennyLaneAI/catalyst/pull/1348)

  A new module, `catalyst.utils.toml_utils`, was also added to assist in loading information from
  these databases, which are stored as text files in TOML format. In particular, this module
  contains a new function, :func:`~.utils.toml_utils.safe_eval`, to safely evaluate mathematical
  expressions:

  ```python
  >>> from catalyst.utils.toml_utils import safe_eval
  >>> safe_eval("2 * math.pi * 1e9")
  6283185307.179586
  ```

* A default backend for OQD trapped-ion quantum devices has been added.
  [(#1355)](https://github.com/PennyLaneAI/catalyst/pull/1355)

* `expval` and `var` operations no longer keep the static shots attribute, as a step towards supporting dynamic shots across catalyst.
  [(#1360)](https://github.com/PennyLaneAI/catalyst/pull/1360)

* A new `ion` dialect has been added for Catalyst programs targeting OQD trapped-ion quantum devices.
  [(#1260)](https://github.com/PennyLaneAI/catalyst/pull/1260)
  [(#1372)](https://github.com/PennyLaneAI/catalyst/pull/1372)

  The `ion` dialect defines the set of physical properties of the device, such as the ion species
  and their atomic energy levels, as well as the operations to manipulate the qubits in the
  trapped-ion system, such as laser pulse durations, polarizations, detuning frequencies, etc.

  A new pass, `--quantum-to-ion`, has also been added to convert logical gate-based circuits in the
  Catalyst `quantum` dialect to laser pulse operations in the `ion` dialect. This pass accepts
  logical quantum gates from the set {RX, RY, Mølmer–Sørensen (MS)}. Doing so enables the insertion
  of physical device parameters into the IR, which will be necessary when lowering to OQD's backend
  calls. The physical parameters are read in from [TOML](https://toml.io/en/) files during the
  `--quantum-to-ion` conversion. The TOML files are assumed to exist by the pass (the paths to the
  TOML file locations are taken in as pass options), with the intention that they are generated
  immediately before compilation during hardware-calibration runs.

* IR is now extended to support literal values as opposed to SSA Values for static parameters of
  quantum gates by adding a new gate called StaticCustomOp with lowering to regular customOp.
  [(#1387)](https://github.com/PennyLaneAI/catalyst/pull/1387)

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an
  empty pass that prints hello world. The code of the tutorial is at
  [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

* Updated catalyst-cli documentation to reflect the removal of func-name option for trasnformation passes.
  [(#1368)](https://github.com/PennyLaneAI/catalyst/pull/1368)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Joey Carter,
David Ittah,
Erick Ochoa Lopez,
Mehrdad Malekmohammadi,
William Maxwell
Romain Moyard,
Shuli Shu,
Ritu Thombre,
Raul Torres,
Paul Haochen Wang.
