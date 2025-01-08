# Release 0.10.0 (current release)

<h3>New features since last release</h3>

* Catalyst can now load and apply local MLIR plugins from the PennyLane frontend.
  [(#1287)](https://github.com/PennyLaneAI/catalyst/pull/1287)
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)
  [(#1361)](https://github.com/PennyLaneAI/catalyst/pull/1361)
  [(#1370)](https://github.com/PennyLaneAI/catalyst/pull/1370)

  Custom compilation passes and dialects in MLIR can be specified for use in Catalyst via a shared
  object (`*.so` or `*.dylib` on macOS) that implements the pass. Details on creating your own
  plugin can be found in our
  [compiler plugin documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/plugins.html).
  At a high level, there are three ways to use a plugin once it's properly specified:

  - :func:`~.passes.apply_pass` can be used on QNodes when there is a
    [Python entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
    defined for the plugin. In that case, the plugin and pass should both be specified and separated
    by a period.

    ```python
    @catalyst.passes.apply_pass("plugin_name.pass_name")
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        return qml.state()

    @qml.qjit
    def module():
        return qnode()
    ```

  - :func:`~.passes.apply_pass_plugin` can be used on QNodes when the plugin did define an entry
    point. In that case the full filesystem path must be specified in addition to the pass name.

    ```python
    from pathlib import Path

    @catalyst.passes.apply_pass_plugin(Path("path_to_plugin"), "pass_name")
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        return qml.state()

    @qml.qjit
    def module():
        return qnode()
    ```

  - Alternatively, one or more dialect and pass plugins can be specified in advance in the
    :func:`~.qjit` decorator, via the `pass_plugins` and `dialect_plugins` keyword arguments. The
    :func:`~.passes.apply_pass` function can then be used without specifying the plugin.

    ```python
    from pathlib import Path

    plugin = Path("shared_object_file.so")

    @catalyst.passes.apply_pass("pass_name")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
      qml.Hadamard(wires=0)
      return qml.state()

    @qml.qjit(pass_plugins=[plugin], dialect_plugins=[plugin])
    def module():
      return qnode()
    ```

  For more information on usage, visit our
  [compiler plugin documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/plugins.html).

<h3>Improvements 🛠</h3>

* The Catalyst CLI, a command line interface for debugging and dissecting different stages of
  compilation, is now available under the `catalyst` command after installing Catalyst with pip.
  Even though the tool was first introduced in `v0.9`, it was not yet included in binary
  distributions of Catalyst (wheels). The full usage instructions are available in the
  [Catalyst CLI documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/catalyst-cli/catalyst-cli.html).
  [(#1285)](https://github.com/PennyLaneAI/catalyst/pull/1285)
  [(#1368)](https://github.com/PennyLaneAI/catalyst/pull/1368)
  [(#1405)](https://github.com/PennyLaneAI/catalyst/pull/1405)

* Lightning devices now support finite-shot expectation values of `qml.Hermitian` when used with
  Catalyst.
  [(#451)](https://github.com/PennyLaneAI/catalyst/pull/451)

* The PennyLane state preparation template `qml.CosineWindow` is now compatible with Catalyst.
  [(#1166)](https://github.com/PennyLaneAI/catalyst/pull/1166)

* A development distribution of Python with dynamic linking support (`libpython.so`) is no longer
  needed in order to use :func:`~.debug.compile_executable` to generate standalone executables of
  compiled programs.
  [(#1305)](https://github.com/PennyLaneAI/catalyst/pull/1305)

* In Catalyst `v0.9` the output of the compiler instrumentation (:func:`~.debug.instrumentation`)
  had inadvertently been made more verbose by printing timing information for each run of each pass.
  This change has been reverted. Instead, the :func:`~.qjit` option `verbose=True` will now instruct
  the instrumentation to produce this more detailed output.
  [(#1343)](https://github.com/PennyLaneAI/catalyst/pull/1343)

* Two additional circuit optimizations have been added to Catalyst: `disentangle-CNOT` and
  `disentangle-SWAP`. The optimizations are available via the :mod:`~.passes` module.
  [(#1154)](https://github.com/PennyLaneAI/catalyst/pull/1154)

  The optimizations use a finite state machine to propagate limited qubit state information through
  the circuit to turn CNOT and SWAP gates into cheaper instructions. The pass is based on the work
  by J. Liu, L. Bello, and H. Zhou, _Relaxed Peephole Optimization: A Novel Compiler Optimization
  for Quantum Circuits_, 2020, [arXiv:2012.07711](https://arxiv.org/abs/2012.07711).

<h3>Breaking changes 💔</h3>

* The minimum supported PennyLane version has been updated to `v0.40`; backwards compatibility in
  either direction is not maintained.
  [(#1308)](https://github.com/PennyLaneAI/catalyst/pull/1308)

* (Device Developers Only) The way the `shots` parameter is initialized in C++ device backends is
  changing.
  [(#1310)](https://github.com/PennyLaneAI/catalyst/pull/1310)

  The previous method of including the shot number in the `kwargs` argument of the device
  constructor is deprecated and will be removed in the next release (`v0.11`). Instead, the shots
  value will be specified exclusively via the existing `SetDeviceShots` function called at the
  beginning of a quantum execution. Device developers are encouraged to update their device
  implementations between this and the next release while both methods are supported.

  Similarly, the `Sample` and `Counts` functions (and their `Partial*` equivalents) will no longer
  provide a `shots` argument, since they are redundant. The signature of these functions will update
  in the next release.

* (Device Developers Only) The `toml`-based device schemas have been integrated with PennyLane and
  updated to a new version `schema = 3`.
  [(#1275)](https://github.com/PennyLaneAI/catalyst/pull/1275)

  Devices with existing TOML `schema = 2` will not be compatible with the current release of
  Catalyst until updated. A summary of the most importation changes is listed here:
  - `operators.gates.native` renamed to `operators.gates`
  - `operators.gates.decomp` and `operators.gates.matrix` are removed and no longer necessary
  - `condition` property is renamed to `conditions`
  - Entries in the `measurement_processes` section now expect the full PennyLane class name as
    opposed to the deprecated `mp.return_type` shorthand (e.g. `ExpectationMP` instead of `Expval`).
  - The `mid_circuit_measurements` field has been replaced with `supported_mcm_methods`, which
    expects a list of mcm methods that the device is able to work with (or empty if unsupported).
  - A new field has been added, `overlapping_observables`, which indicates whether a device supports
    multiple measurements during one execution on overlapping wires.
  - The `options` section has been removed. Instead, the Python device class should define a
    `device_kwargs` field holding the name and values of C++ device constructor kwargs.

  See the [Custom Devices page](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/custom_devices.html)
  for the most up-to-date information on integrating your device with Catalyst and PennyLane.

<h3>Bug fixes 🐛</h3>

* Fixed a bug introduced in Catalyst `v0.8` that breaks nested invocations of `qml.adjoint` and
  `qml.ctrl` (e.g. `qml.adjoint(qml.adjoint(qml.H(0)))`).
  [(#1301)](https://github.com/PennyLaneAI/catalyst/issues/1301)

* Fixed a bug in :func:`~.debug.compile_executable` when using non-64bit arrays as input to the
  compiled function, due to incorrectly computed stride information.
  [(#1338)](https://github.com/PennyLaneAI/catalyst/pull/1338)

<h3>Internal changes ⚙️</h3>

* Starting with Python 3.12, Catalyst's binary distributions (wheels) will now follow Python's
  [Stable ABI](https://docs.python.org/3/c-api/stable.html), eliminating the need for a separate
  wheel per minor Python version. To enable this, the following changes have made:

  - Stable ABI wheels are now generated for Python 3.12 and up.
    [(#1357)](https://github.com/PennyLaneAI/catalyst/pull/1357)
    [(#1385)](https://github.com/PennyLaneAI/catalyst/pull/1385)

  - Pybind11 has been replaced with nanobind for C++/Python bindings across all components.
    [(#1173)](https://github.com/PennyLaneAI/catalyst/pull/1173)
    [(#1293)](https://github.com/PennyLaneAI/catalyst/pull/1293)
    [(#1391)](https://github.com/PennyLaneAI/catalyst/pull/1391)
    [(#624)](https://github.com/PennyLaneAI/catalyst/pull/624)

    Nanobind has been developed as a natural successor to the pybind11 library and offers a number
    of [advantages](https://nanobind.readthedocs.io/en/latest/why.html#major-additions) like its
    ability to target Python's Stable ABI.
    starting with Python 3.12.

  - Python C-API calls have been replaced with functions from Python's Limited API.
    [(#1354)](https://github.com/PennyLaneAI/catalyst/pull/1354)

  - The `QuantumExtension` module for MLIR Python bindings, which relies on pybind11, has been
    removed. The module was never included in the distributed wheels and could not be converted to
    nanobind easily due to its dependency on upstream MLIR code. Pybind11 does not support the
    Python Stable ABI.
    [(#1187)](https://github.com/PennyLaneAI/catalyst/pull/1187)

* Catalyst no longer depends on or pins the `scipy` package. Instead, OpenBLAS is sourced directly
  from [scipy-openblas32](https://pypi.org/project/scipy-openblas32/) or
  [Accelerate](https://developer.apple.com/accelerate/) is used.
  [(#1322)](https://github.com/PennyLaneAI/catalyst/pull/1322)
  [(#1328)](https://github.com/PennyLaneAI/catalyst/pull/1328)

* The Catalyst plugin for the `lightning.qubit` device has been migrated from the Catalyst repo to
  the [Lightning repository](https://github.com/PennyLaneAI/pennylane-lightning). This reduces
  the size of Catalyst's binary distributions and the build time of the project, by avoiding
  re-compilation of the lightning source code.
  [(#1227)](https://github.com/PennyLaneAI/catalyst/pull/1227)
  [(#1307)](https://github.com/PennyLaneAI/catalyst/pull/1307)
  [(#1312)](https://github.com/PennyLaneAI/catalyst/pull/1312)

* The AutoGraph exception mechanism (`allowlist` parameter) has been streamlined to only be used in
  places where it's required.
  [(#1332)](https://github.com/PennyLaneAI/catalyst/pull/1332)
  [(#1337)](https://github.com/PennyLaneAI/catalyst/pull/1337)

* Each QNode now has its own transformation schedule. Instead of relying on the name of the QNode,
  each QNode now has a transformation module, which denotes the transformation schedule, embedded in
  its MLIR representation.
  [(#1323)](https://github.com/PennyLaneAI/catalyst/pull/1323)

* The `apply_registered_pass_p` primitive has been removed and the API for scheduling passes to run
  using the transform dialect has been refactored. In particular, passes are appended to a tuple as
  they are being registered and they will be run in order. If there are no local passes, the global
  `pass_pipeline` is scheduled. Furthermore, this commit also reworks the caching mechanism for
  primitives, which is important as qnodes and functions are primitives and now that we can apply
  passes to them, they are distinct based on which passes have been scheduled to run on them.
  [(#1317)](https://github.com/PennyLaneAI/catalyst/pull/1317)

* The Catalyst infrastructure has been upgraded to support a dynamic `shots` parameter for quantum
  execution. Previously, this value had to be a static compile-time constant, and could not be
  changed once the program was compiled. Upcoming UI changes will make the feature accessible to
  users.
  [(#1360)](https://github.com/PennyLaneAI/catalyst/pull/1360)

* Several changes for experimental support of trapped-ion OQD devices have been made, including:

  - An experimental `ion` dialect has been added for Catalyst programs targeting OQD trapped-ion
    quantum devices.
    [(#1260)](https://github.com/PennyLaneAI/catalyst/pull/1260)
    [(#1372)](https://github.com/PennyLaneAI/catalyst/pull/1372)

    The `ion` dialect defines the set of physical properties of the device, such as the ion species
    and their atomic energy levels, as well as the operations to manipulate the qubits in the
    trapped-ion system, such as laser pulse durations, polarizations, detuning frequencies, etc.

    A new pass, `--quantum-to-ion`, has also been added to convert logical gate-based circuits in
    the Catalyst `quantum` dialect to laser pulse operations in the `ion` dialect. This pass accepts
    logical quantum gates from the set `{RX, RY, MS}`, where `MS` is the Mølmer–Sørensen gate. Doing
    so enables the insertion of physical device parameters into the IR, which will be necessary when
    lowering to OQD's backend calls. The physical parameters, which are typically obtained from
    hardware-calibration runs, are read in from [TOML](https://toml.io/en/) files during the
    `--quantum-to-ion` conversion. The TOML filepaths are taken in as pass options.

  - A plugin and device backend for OQD trapped-ion quantum devices has been added.
    [(#1355)](https://github.com/PennyLaneAI/catalyst/pull/1355)
    [(#1403)](https://github.com/PennyLaneAI/catalyst/pull/1403)

  - An MLIR transformation has been added to decompose `{T, S, Z, Hadamard, RZ, PhaseShift, CNOT}`
    gates into the set `{RX, RY, MS}`.
    [(#1226)](https://github.com/PennyLaneAI/catalyst/pull/1226)

  Support for OQD devices is still under development, therefore OQD modules are currently not
  included in binary distributions (wheels) of Catalyst.

* The Catalyst IR has been extended to support literal values as opposed to SSA Values for static
  parameters of quantum gates by adding a new gate called `StaticCustomOp`, with eventual lowering
  to the regular `CustomOp` operation.
  [(#1387)](https://github.com/PennyLaneAI/catalyst/pull/1387)

* Code readability in the `catalyst.pipelines` module has been improved, in particular for pipelines
  with conditionally included passes.
  [(#1194)](https://github.com/PennyLaneAI/catalyst/pull/1194)

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an
  empty pass that prints `hello world`. The code for the tutorial is located in
  [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

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
