# Release 0.10.0 (current release)

<h3>New features since last release</h3>

* Catalyst can now load and apply local MLIR plugins from the PennyLane frontend.
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)
  [(#1361)](https://github.com/PennyLaneAI/catalyst/pull/1361)
  [(#1370)](https://github.com/PennyLaneAI/catalyst/pull/1370)

  Custom compilation passes and dialects in MLIR can be specified for use in Catalyst via a shared 
  object (`*.so` or `*.dylib` on MacOS) that implements the pass. Details on creating your own 
  plugin can be found in our 
  [compiler plugin documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/plugins.html).
  At a high level, there are three ways to utilize a plugin once it's properly specified:

  * :func:`~.passes.apply_pass` can be used on QNodes when there is a  
    [Python entry point](https://packaging.python.org/en/latest/specifications/entry-points/) 
    defined for the plugin.

    ```python
    @catalyst.passes.apply_pass(pass_name)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        return qml.state()

    @qml.qjit
    def module():
        return qnode()
    ```

  * :func:`~.passes.apply_pass_plugin` can be used on QNodes when there is not an entry point 
    defined for the plugin.

    ```python
    @catalyst.passes.apply_pass_plugin(path_to_plugin, pass_name)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        return qml.state()

    @qml.qjit
    def module():
        return qnode()
    ```

  * Specifying multiple compilation pass plugins or dialect plugins directly in :func:`~.qjit` via 
    the `pass_plugins` and `dialect_plugins` keyword arguments, which must be lists of plugin paths.

    ```python
    from pathlib import Path

    plugin = Path("shared_object_file.so")

    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
      qml.Hadamard(wires=0)
      return qml.state()

    @qml.qjit(pass_plugins=[plugin], dialect_plugins=[plugin])
    def module():
      return catalyst.passes.apply_pass(qnode, "pass_name")()
    ```

  For more information on usage, 
  visit our [compiler plugin documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/plugins.html).

<h3>Improvements 🛠</h3>

* The lightning runtime now supports finite shots with measuring expectation values of `qml.Hermitian`.
  [(#451)](https://github.com/PennyLaneAI/catalyst/pull/451)

* Pybind11 has been replaced with nanobind for C++/Python bindings in the frontend and in the runtime.
  [(#1173)](https://github.com/PennyLaneAI/catalyst/pull/1173)
  [(#1293)](https://github.com/PennyLaneAI/catalyst/pull/1293)
  [(#1391)](https://github.com/PennyLaneAI/catalyst/pull/1391)

  Nanobind has been developed as a natural successor to the pybind11 library and offers a number of
  [advantages](https://nanobind.readthedocs.io/en/latest/why.html#major-additions) like its ability 
  to target Python's [stable ABI interface](https://docs.python.org/3/c-api/stable.html) starting 
  with Python 3.12.

* Catalyst now uses the new compiler API (`catalyst-cli`) to compile quantum code from the Python 
  frontend instead of using pybind11 as an interface between the compiler and the frontend. 
  [(#1285)](https://github.com/PennyLaneAI/catalyst/pull/1285)

* Gates in the gate set `{T, S, Z, Hadamard, RZ, PhaseShift, CNOT}` now have MLIR decompositions to 
  the gate set `{RX, RY, MS}`, which are useful for trapped ion devices.
  [(#1226)](https://github.com/PennyLaneAI/catalyst/pull/1226)

* `qml.CosineWindow` is now compatible with QJIT.
  [(#1166)](https://github.com/PennyLaneAI/catalyst/pull/1166)

* All PennyLane templates are tested for QJIT compatibility.
  [(#1161)](https://github.com/PennyLaneAI/catalyst/pull/1161)

* Python is now decoupled from the Runtime by using the Python Global Interpreter Lock (GIL) instead 
  of custom mutexes.
  [(#624)](https://github.com/PennyLaneAI/catalyst/pull/624)

  In addition, executables created using :func:`~.debug.compile_executable` no longer require
  linking against Python shared libraries after decoupling Python from the Runtime C-API.
  [(#1305)](https://github.com/PennyLaneAI/catalyst/pull/1305)

* The readability of conditional passes in `catalyst.pipelines` has been improved.
  [(#1194)](https://github.com/PennyLaneAI/catalyst/pull/1194)

* The output of compiler instrumentation has been cleaned up by only printing stats after a `pipeline`. 
  It is still possible to get the more detailed output with `qjit(verbose=True)`.
  [(#1343)](https://github.com/PennyLaneAI/catalyst/pull/1343)

* Stable ABI wheels for Python 3.12 and up are now generated.
  [(#1357)](https://github.com/PennyLaneAI/catalyst/pull/1357)
  [(#1385)](https://github.com/PennyLaneAI/catalyst/pull/1385)

* Two new circuit optimization passes, `--disentangle-CNOT` and `--disentangle-SWAP`, are available.
  [(#1154)](https://github.com/PennyLaneAI/catalyst/pull/1154)

  The CNOT pass disentangles CNOT gates whenever possible, e.g., when the control bit is known to be 
  in the `|0>` state, the pass removes the CNOT. The pass uses a finite state machine to propagate 
  simple one-qubit states, in order to determine the input states to the CNOT.
  
  Similarly, the SWAP pass disentangles SWAP gates whenever possible by using a finite state machine 
  to propagate simple one-qubit states, similar to the `--disentangle-CNOT` pass.

  Both passes are implemented in accordance with the algorithm from 
  J. Liu, L. Bello, and H. Zhou, _Relaxed Peephole Optimization: A Novel Compiler Optimization for Quantum Circuits_, 2020, [arXiv:2012.07711](https://arxiv.org/abs/2012.07711) [quant-ph].

* Allow specifying a branch to switch to when setting up a dev environment from the wheels.
  [(#1406)](https://github.com/PennyLaneAI/catalyst/pull/1406)

<h3>Breaking changes 💔</h3>

* The `sample` and `counts` measurement primitives now support dynamic shot values across Catalyst, 
  although, on the PennyLane side, the device's shots is still constrained to a static integer 
  literal.
  [(#1310)](https://github.com/PennyLaneAI/catalyst/pull/1310)

  To support this, `SampleOp` and `CountsOp` in MLIR no longer carry the shots attribute, since 
  integer attributes are tied to literal values and must be static.

  `DeviceInitOp` now takes in an optional SSA argument for shots, and the device init runtime CAPI 
  will take in this SSA shots value as an argument and set it as the device shots. The sample and 
  counts runtime CAPI functions no longer take in the shots argument and will retrieve shots from 
  the device.

  Correspondingly, the device C++ interface should no longer parse the `DeviceInitOp`'s attributes 
  dictionary for the shots. For now, we still keep the shots as an attribute so device implementors 
  can have time to migrate, but we will remove shots from the attribute dictionary in the next 
  release (`v0.11`)

* The `toml` module has been migrated to PennyLane with an updated schema for declaring device
  capabilities. Devices with TOML files using `schema = 2` will not be compatible with the latest
  Catalyst. See the [Custom Devices documentation page](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html)
  for updated instructions on integrating your device with Catalyst and PennyLane.
  [(#1275)](https://github.com/PennyLaneAI/catalyst/pull/1275)

* Handling for the legacy operator arithmetic (the `Hamiltonian` and `Tensor` classes in PennyLane)
  has been removed.
  [(#1308)](https://github.com/PennyLaneAI/catalyst/pull/1308)

<h3>Bug fixes 🐛</h3>

* Fixed a bug introduced in 0.8 that breaks nested invocations of `qml.adjoint` and `qml.ctrl` (e.g., 
  `qml.adjoint(qml.adjoint(qml.H(0)))`).
  [(#1301)](https://github.com/PennyLaneAI/catalyst/issues/1301)

* Fixed a bug in :func:`~.debug.compile_executable` that would generate incorrect stride information for
  array arguments of the function, in particular when non-64bit datatypes are used.
  [(#1338)](https://github.com/PennyLaneAI/catalyst/pull/1338)

<h3>Internal changes ⚙️</h3>

* Catalyst no longer depends on or pins the `scipy` package. Instead, OpenBLAS is sourced directly
  from [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/) or 
  [Accelerate](https://developer.apple.com/accelerate/) is used.
  [(#1322)](https://github.com/PennyLaneAI/catalyst/pull/1322)
  [(#1328)](https://github.com/PennyLaneAI/catalyst/pull/1328)

* The `QuantumExtension` module—previously implemented with pybind11—has been removed. This module
  was not included in the distributed wheels and has been deprecated to align with our adoption of
  Python's stable ABI, which pybind11 does not support.
  [(#1187)](https://github.com/PennyLaneAI/catalyst/pull/1187)

* Code for using `lightning.qubit` with Catalyst has been moved from the Catalyst repository to 
  the [Lightning repository](https://github.com/PennyLaneAI/pennylane-lightning) so that Catalyst
  wheels will build faster.
  [(#1227)](https://github.com/PennyLaneAI/catalyst/pull/1227)
  [(#1307)](https://github.com/PennyLaneAI/catalyst/pull/1307)
  [(#1312)](https://github.com/PennyLaneAI/catalyst/pull/1312)

* `catalyst-cli` and `quantum-opt` are now compiled with `default` visibility, which allows for MLIR 
  plugins to work.
  [(#1287)](https://github.com/PennyLaneAI/catalyst/pull/1287)

* The patching mechanism of autograph's `allowlist` has been streamlined to only be used in places 
  where it's required.
  [(#1332)](https://github.com/PennyLaneAI/catalyst/pull/1332)
  [(#1337)](https://github.com/PennyLaneAI/catalyst/pull/1337)

* Each qnode now has its own transformation schedule. Instead of relying on the name of the qnode, 
  each qnode now has a transformation module, which denotes the transformation schedule, embedded in 
  its MLIR representation.
  [(#1323)](https://github.com/PennyLaneAI/catalyst/pull/1323)

* The `apply_registered_pass_p` primitive has been removed and the API for scheduling passes to run 
  using the transform dialect has been refactored. In particular, passes are appended to a tuple as 
  they are being registered and they will be run in order. If there are no local passes, the global 
  `pass_pipeline` is scheduled. Furthermore, this commit also reworks the caching mechanism for 
  primitives, which is important as qnodes and functions are primitives and now that we can apply 
  passes to them, they are distinct based on which passes have been scheduled to run on them.
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)

* Python C-API calls have been replaced with Stable ABI calls.
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
  [(#1403)](https://github.com/PennyLaneAI/catalyst/pull/1355)

  Support for OQD devices is still under development, therefore the OQD modules are currently not
  included in the distributed wheels.

* As a step towards supporting dynamic shots across catalyst, `expval` and `var` operations no 
  longer keep the static shots attribute.
  [(#1360)](https://github.com/PennyLaneAI/catalyst/pull/1360)

* A new `ion` dialect has been added for Catalyst programs targeting OQD trapped-ion quantum devices.
  [(#1260)](https://github.com/PennyLaneAI/catalyst/pull/1260)
  [(#1372)](https://github.com/PennyLaneAI/catalyst/pull/1372)

  The `ion` dialect defines the set of physical properties of the device, such as the ion species
  and their atomic energy levels, as well as the operations to manipulate the qubits in the
  trapped-ion system, such as laser pulse durations, polarizations, detuning frequencies, etc.

  A new pass, `--quantum-to-ion`, has also been added to convert logical gate-based circuits in the
  Catalyst `quantum` dialect to laser pulse operations in the `ion` dialect. This pass accepts
  logical quantum gates from the set `{RX, RY, MS}`, where `MS` is the Mølmer–Sørensen gate. Doing 
  so enables the insertion of physical device parameters into the IR, which will be necessary when 
  lowering to OQD's backend calls. The physical parameters are read in from 
  [TOML](https://toml.io/en/) files during the `--quantum-to-ion` conversion. The TOML files are 
  assumed to exist by the pass (the paths to the TOML file locations are taken in as pass options), 
  with the intention that they are generated immediately before compilation during 
  hardware-calibration runs.

* The Catalyst IR has been extended to support literal values as opposed to SSA Values for static 
  parameters of quantum gates by adding a new gate called `StaticCustomOp` with lowering to regular 
  `CustomOp`.
  [(#1387)](https://github.com/PennyLaneAI/catalyst/pull/1387)
  [(#1396)](https://github.com/PennyLaneAI/catalyst/pull/1396)

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an
  empty pass that prints `hello world`. The code for the tutorial is located in
  [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

* The `catalyst-cli` documentation has been updated to reflect the removal of the `func-name` option 
  for transformation passes.
  [(#1368)](https://github.com/PennyLaneAI/catalyst/pull/1368)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Joey Carter,
David Ittah,
Erick Ochoa Lopez,
Mehrdad Malekmohammadi,
William Maxwell,
Romain Moyard,
Shuli Shu,
Ritu Thombre,
Raul Torres,
Paul Haochen Wang.
