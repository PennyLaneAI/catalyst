# Release 0.13.0 (current release)

<h3>New features since last release</h3>

* A new experimental decomposition system is introduced in Catalyst enabling the
  PennyLane's graph-based decomposition and MLIR-based lowering of decomposition rules.
  This feature is integrated with PennyLane program capture and graph-based decomposition
  including support for custom decomposition rules and operators via ``qml.transforms.decompose``.
  Similar to PennyLane's behaviour, this experimental feature will fall back to the old system
  whenever the graph cannot find decomposition rules for all unsupported operators in the program,
  and a ``UserWarning`` is raised.
  [(#2001)](https://github.com/PennyLaneAI/catalyst/pull/2001)
  [(#2029)](https://github.com/PennyLaneAI/catalyst/pull/2029)
  [(#2068)](https://github.com/PennyLaneAI/catalyst/pull/2068)
  [(#2091)](https://github.com/PennyLaneAI/catalyst/pull/2091)
  [(#2099)](https://github.com/PennyLaneAI/catalyst/pull/2099)


* Catalyst now supports dynamic wire allocation with ``qml.allocate()`` and
  ``qml.deallocate()`` when program capture is enabled.
  [(#2002)](https://github.com/PennyLaneAI/catalyst/pull/2002)
  [(#2075)](https://github.com/PennyLaneAI/catalyst/pull/2075)

  Two new functions, ``qml.allocate()`` and ``qml.deallocate()``, [have been added to
  PennyLane](https://docs.pennylane.ai/en/stable/development/release_notes.html#release-0-43-0)
  to support dynamic wire allocation. With Catalyst, these features can be accessed on
   ``lightning.qubit``, ``lightning.kokkos``, and ``lightning.gpu``.

  Dynamic wire allocation refers to the allocation of wires in the middle of a circuit, as opposed to the static allocation during device initialization. For example:

  ```python
  qml.capture.enable()

  @qjit
  @qml.qnode(qml.device("lightning.qubit", wires=3))  # 3 initial qubits
  def circuit():
      qml.X(0)                        # |10>

      with qml.allocate(1) as q:      # |10> and |0>, 1 dynamically allocated qubit
          qml.X(q[0])                 # |10> and |1>
          qml.CNOT(wires=[q[0], 1])   # |11> and |1>

      return qml.probs(wires=[0, 1, 2])
  ```

  ```pycon
  >>>  print(circuit())
  [0. 0. 0. 1.]
  ```

  In the above program, 2 qubits are allocated during device initialization, and 1
  additional qubit is allocated inside the circuit with ``qml.allocate(1)``.

  For more information on what ``qml.allocate`` and ``qml.deallocate`` do, please consult the
  [PennyLane v0.43 release notes](https://docs.pennylane.ai/en/stable/development/release_notes.html#release-0-43-0).

  However, there are some notable differences between the behaviour of these features
  with ``qjit`` versus without. For details, please see the relevant sections in the
  [Catalyst sharp bits page](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/sharp_bits.html#functionality-differences-from-pennylane).

* A new quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
  rotations (PPRs) in circuits is now available. This compilation pass works by commuting
  non-Clifford PPRs (often just referred to as ``T`` gates) in adjacent
  layers and merging compatible ones. More details can be found in Figure 6 of
  [A Game of Surface Codes](https://arXiv:1808.02892v3).
  [(#1975)](https://github.com/PennyLaneAI/catalyst/pull/1975)
  [(#2048)](https://github.com/PennyLaneAI/catalyst/pull/2048)
  [(#2085)](https://github.com/PennyLaneAI/catalyst/pull/2085)

  The impact of the ``reduce_t_depth`` pass can be measured using :func:`~.passes.ppm_specs`
  to compare the circuit depth before and after applying the pass. The ``ppm_specs`` function
  provides detailed statistics including ``depth_pi8_ppr`` (non-Clifford PPR depth) and
  ``pi8_ppr`` (number of non-Clifford PPRs), allowing users to quantify the optimization
  achieved by the pass.

  Consider the following circuit:

  ```python
  import pennylane as qml
  from catalyst import qjit, measure
  from catalyst.passes import to_ppr, commute_ppr, reduce_t_depth, merge_ppr_ppm

  pips = [("pipe", ["enforce-runtime-invariants-pipeline"])]

  no_reduce_T = {
      "to_ppr": {},
      "commute_ppr": {},
      "merge_ppr_ppm": {},
  }

  reduce_T = {
      "to_ppr": {},
      "commute_ppr": {},
      "merge_ppr_ppm": {},
      "t_layer_reduction": {}
  }

  for pipeline in [reduce_T, no_reduce_T]:

      @qjit(pipelines=pips, target="mlir", circuit_transform_pipeline=pipeline)
      @qml.qnode(qml.device("null.qubit", wires=3))
      def circuit():
          n = 3
          for i in range(n):
              qml.H(wires=i)
              qml.S(wires=i)
              qml.CNOT(wires=[i, (i + 1) % n])
              qml.T(wires=i)
              qml.H(wires=i)
              qml.T(wires=i)

          return [measure(wires=i) for i in range(n)]

      print(ppm_specs(circuit))
  ```

  ```pycon
  {'circuit_0': {'depth_pi8_ppr': 3, 'depth_ppm': 1, 'logical_qubits': 3, 'max_weight_pi8': 3, 'num_of_ppm': 3, 'pi8_ppr': 6}}
  {'circuit_0': {'depth_pi8_ppr': 4, 'depth_ppm': 1, 'logical_qubits': 3, 'max_weight_pi8': 3, 'num_of_ppm': 3, 'pi8_ppr': 6}}
  ```

  After performing the :func:`~.passes.to_ppr`, :func:`~.passes.commute_ppr`, and :func:`~.passes.merge_ppr_ppm`,
  passes, the circuit contains a depth of four of non-Clifford PPRs (`depth_pi8_ppr`). Subsequently applying the
  :func:`~.passes.t_layer_reduction` pass will move PPRs around via commutation, resulting in a circuit with a
  smaller PPR depth of three.

* Catalyst now provides native support for `SingleExcitation`, `DoubleExcitation`,
  and `PCPhase` on compatible devices like Lightning simulators.
  This enhancement avoids unnecessary gate decomposition,
  leading to reduced compilation time and improved overall performance.
  [(#1980)](https://github.com/PennyLaneAI/catalyst/pull/1980)
  [(#1987)](https://github.com/PennyLaneAI/catalyst/pull/1987)

  For example, the code below is captured with `PCPhase` avoiding the
  decomposition to many `MultiControlledX` and `PhaseShift` gates:

  ```python
  dev = qml.device("lightning.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.ctrl(qml.PCPhase(0.5, dim=1, wires=[0, 2]), control=[1])
      return qml.probs()
  ```

* Catalyst now supports returning classical and MCM values with the dynamic one-shot MCM method.
  [(#2004)](https://github.com/PennyLaneAI/catalyst/pull/2004)
  [(#2090)](https://github.com/PennyLaneAI/catalyst/pull/2090)

  For example, the code below will generate 10 values, with an equal probability of 42 and 43
  appearing.

  ```python
  import pennylane as qml
  from catalyst import qjit, measure

  @qjit(autograph=True)
  @qml.set_shots(10)
  @qml.qnode(qml.device("lightning.qubit", wires=1), mcm_method="one-shot")
  def circuit():
      qml.Hadamard(wires=0)
      m = measure(0)
      if m:
          return 42, m
      else:
          return 43, m
  ```

  ```pycon
  >>>  print(circuit())
  (Array([42, 43, 42, 42, 43, 42, 42, 43, 42, 42], dtype=int64),
   Array([ True, False,  True,  True, False,  True,  True, False,  True,
           True], dtype=bool))
  ```

* The default mid-circuit measurement method in catalyst has been changed from `"single-branch-statistics"` to `"one-shot"`.
  [[#2017]](https://github.com/PennyLaneAI/catalyst/pull/2017)
  [[#2019]](https://github.com/PennyLaneAI/catalyst/pull/2019)

<h3>Improvements 🛠</h3>

* `QubitUnitary` is no longer favoured in the decomposition of controlled operators when the
  operator is not natively supported by the device, and the device supports `QubitUnitary`. Instead,
  conversion to `QubitUnitary` only happens if the operator does not define another decomposition.
  The previous behaviour was the cause of performance issues when dealing with large controlled
  operators, as their matrix representation could be embedded as dense constant data into the
  program. The performance difference can span multiple orders of magnitude depending on regime.
  [(#2100)](https://github.com/PennyLaneAI/catalyst/pull/2100)

* Catalyst conditional operators, such as :func:`~.cond` or :func:`pennylane.cond`, now allow the
  target / branch functions to use arguments in their call signature. Previously, one had to supply
  all values via closure, but this is now done automatically under the hood.
  [(#2096)](https://github.com/PennyLaneAI/catalyst/pull/2096)

* Significantly improved resource tracking with `null.qubit`.
  The new tracking has better integration with PennyLane (e.g. for passing the filename to write out), cleaner documentation, and its own wrapper class.
  It also now tracks circuit depth, as well as gate counts by number of wires.
  [(#2033)](https://github.com/PennyLaneAI/catalyst/pull/2033)
  [(#2055)](https://github.com/PennyLaneAI/catalyst/pull/2055)

* Improve the pass `--ppm-specs` to count the depth of PPRs and PPMs in the circuit.
  [(#2014)](https://github.com/PennyLaneAI/catalyst/pull/2014)

* A new pass `--partition-layers` has been added to group PPR/PPM operations into `qec.layer`
  operations based on qubit interactive and commutativity, enabling circuit analysis and
  potentially to support parallel execution.
  [(#1951)](https://github.com/PennyLaneAI/catalyst/pull/1951)

* Added a new JAX primitive to capture and compile the decomposition rule
  definitions to MLIR. `decomposition_rule` is the decorator integrated
  with this primitive for development purposes.
  [(#1820)](https://github.com/PennyLaneAI/catalyst/pull/1820)

* Renamed `QregManager` to `QubitHandler` and extended the class to manage
  converting PLxPR wire indices into Catalyst JAXPR qubits.
  This is especially useful for lowering subroutines that take
  in qubits as arguments, for example decomposition rules.
  [(#1820)](https://github.com/PennyLaneAI/catalyst/pull/1820)

* Fix resource tracking unit test polluting the environment with output files
  [(#1861)](https://github.com/PennyLaneAI/catalyst/pull/1861)

* Adjoint differentiation is used by default when executing on lightning devices, significantly reduces gradient computation time.
  [(#1961)](https://github.com/PennyLaneAI/catalyst/pull/1961)

* Added `detensorizefunctionboundary` pass to remove scalar tensors across function boundaries and enabled `symbol-dce` pass to remove dead functions, reducing the number of instructions for compilation.
  [(#1904)](https://github.com/PennyLaneAI/catalyst/pull/1904)

* Workflows `for_loop`, `while_loop` and `cond` now error out if `qml.capture` is enabled.
  [(#1945)](https://github.com/PennyLaneAI/catalyst/pull/1945)

* Displays Catalyst version in `quantum-opt --version` output.
  [(#1922)](https://github.com/PennyLaneAI/catalyst/pull/1922)

* Snakecased keyword arguments to :func:`catalyst.passes.apply_pass()` are
  now correctly parsed to kebab-case pass options.
  [(#1954)](https://github.com/PennyLaneAI/catalyst/pull/1954).

  For example:

  ```python
  @qjit(target="mlir")
  @catalyst.passes.apply_pass("some-pass", "an-option", maxValue=1, multi_word_option=1)
  @qml.qnode(qml.device("null.qubit", wires=1))
  def example():
      return qml.state()
  ```

  which looks like the following line in the MLIR:

  ```pycon
  %0 = transform.apply_registered_pass "some-pass" with options = {"an-option" = true, "maxValue" = 1 : i64, "multi-word-option" = 1 : i64}
  ```

* Added checks to raise an error when the input qubits to the multi-qubit gates in the runtime CAPI are not all distinct.
  [(#2006)](https://github.com/PennyLaneAI/catalyst/pull/2006).

* Commuting Clifford Pauli Product Rotation (PPR) operations, past non-Clifford PPRs, now supports P(π/2) Cliffords in addition to P(π/4)
  [(#1966)](https://github.com/PennyLaneAI/catalyst/pull/1966)

* A new jax primitive `qdealloc_qb_p` is available for single qubit deallocations.
  [(#2005)](https://github.com/PennyLaneAI/catalyst/pull/2005)

* Changed the attribute of `number_original_arg` in `CustomCallOp` from dense array to integer.
  [(#2022)](https://github.com/PennyLaneAI/catalyst/pull/2022)

* Renaming `get_ppm_specs` to `ppm_specs` and the corresponding results' properties.
  [(#2031)](https://github.com/PennyLaneAI/catalyst/pull/2031)

* A new decomposition rule for non-Clifford PPRs into two PPMs based on the Active Volume paper.
  [(#2043)](https://github.com/PennyLaneAI/catalyst/pull/2043)

* Added support to avoid Y-basis measurements in `pauli-corrected` PPR decomposition.
  [(#2047)](https://github.com/PennyLaneAI/catalyst/pull/2047)

* Update conversion targets and refactor `to_ppr` and `ppm_compilation` passes and handle gate `I`.
  [(#2058)](https://github.com/PennyLaneAI/catalyst/pull/2058)

* Using `keep_intermediate='pass'` option now prints the whole module scope of program to the
  intermediate files instead of just the pass scope.
  [(#2051)](https://github.com/PennyLaneAI/catalyst/pull/2051)

<h3>Breaking changes 💔</h3>

* (Device implementers only) The `ReleaseAllQubits` device interface function
  has been replaced with `ReleaseQubits`.
  [(#1996)](https://github.com/PennyLaneAI/catalyst/pull/1996)

  Instead of releasing all currently active qubits, the new interface
  function `ReleaseQubits` explicitly takes in an array of qubit IDs to be
  released.

  For devices without dynamic allocation support it is expected that this
  function only succeed if the ID array contains the same values as those
  produced by the initial `AllocateQubits` call, otherwise the device is
  encouraged to raise an error.

* The `shots` property has been removed from `OQDDevice`. The number of shots for a qnode execution is now set directly on the qnode via `qml.set_shots`,
  either used as decorator `@qml.set_shots(num_shots)` or directly on the qnode `qml.set_shots(qnode, shots=num_shots)`.
  [(#1988)](https://github.com/PennyLaneAI/catalyst/pull/1988)

* The JAX version used by Catalyst is updated to 0.6.2.
  [(#1897)](https://github.com/PennyLaneAI/catalyst/pull/1897)

* The version of LLVM and Enzyme used by Catalyst has been updated.
  The mlir-hlo dependency has been replaced with stablehlo.
  [(#1916)](https://github.com/PennyLaneAI/catalyst/pull/1916)
  [(#1921)](https://github.com/PennyLaneAI/catalyst/pull/1921)

  The LLVM version has been updated to
  [commit f8cb798](https://github.com/llvm/llvm-project/tree/f8cb7987c64dcffb72414a40560055cb717dbf74).
  The stablehlo version has been updated to
  [commit 69d6dae](https://github.com/openxla/stablehlo/commit/69d6dae46e1c7de36e6e6973654754f05353cba5).
  The Enzyme version has been updated to
  [v0.0.186](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.186).

<h3>Deprecations 👋</h3>

* Deprecated usages of ``Device.shots`` along with setting ``device(..., shots=...)``.
  Heavily adjusted frontend pipelines within qfunc, tracer, verification and QJITDevice to account for this change.
  Please use ``qml.set_shots(shots=...)`` or set shots at the QNode level (i.e., ``qml.QNode(..., shots=...)``).
  [(#1952)](https://github.com/PennyLaneAI/catalyst/pull/1952)

<h3>Bug fixes 🐛</h3>

* Fixes an issue with program capture and static argnums on the qnode. The lowering to MLIR is no longer cached
  if there are static argnums.
  [(#2053)](https://github.com/PennyLaneAI/catalyst/pull/2053)

* Fix type promotion on conditional branches, where the return values from `cond` should be the promoted one.
  [(#1977)](https://github.com/PennyLaneAI/catalyst/pull/1977)

* Fix wrong handling of partitioned shots in the decomposition pass of `measurements_from_samples`.
  [(#1981)](https://github.com/PennyLaneAI/catalyst/pull/1981)

* Fix errors in AutoGraph transformed functions when `qml.prod` is used together with other operator
  transforms (e.g. `qml.adjoint`). Additionally, fix AutoGraph conversion when `qml.prod` are
  applied outside the `@qjit` scope.
  [(#1910)](https://github.com/PennyLaneAI/catalyst/pull/1910)
  [(#2083)](https://github.com/PennyLaneAI/catalyst/pull/2083)

* A bug in the `NullQubit::ReleaseQubit()` method that prevented the deallocation of individual
  qubits on the `"null.qubit"` device has been fixed.
  [(#1926)](https://github.com/PennyLaneAI/catalyst/pull/1926)

* Stacked Python decorators for built-in Catalyst passes are now applied in the correct order when
  program capture is enabled.
  [(#2027)](https://github.com/PennyLaneAI/catalyst/pull/2027)

* Fix usage of OQC device, including:
  * fix object file system extension on macOS
  * fix wrong type signature of `Counts` API function
  * various updates to OQC internal APIs
  [(#2032)](https://github.com/PennyLaneAI/catalyst/pull/2032)
  [(#2089)](https://github.com/PennyLaneAI/catalyst/pull/2089)

* Fixed the Clifford PPR decomposition rule where using the Y measurement should take the inverse.
  [(#2043)](https://github.com/PennyLaneAI/catalyst/pull/2043)

* `static_argnums` is now correctly passed to internally transformed kernel functions,
for example the one-shot mid circuit measurement transform.
  [(#2056)](https://github.com/PennyLaneAI/catalyst/pull/2056)

* Fixed a bug where applying a quantum transform after a QNode could produce incorrect results or 
  errors in certain cases. This resolves issues related to transforms operating on QNodes with 
  classical outputs and improves compatibility with measurement transforms.
  [(#2081)](https://github.com/PennyLaneAI/catalyst/pull/2081)

<h3>Internal changes ⚙️</h3>

* Updates use of `qml.transforms.dynamic_one_shot.parse_native_mid_circuit_measurements` to improved signature.
  [(#1953)](https://github.com/PennyLaneAI/catalyst/pull/1953)

* When capture is enabled, `qjit(autograph=True)` will use capture autograph instead of catalyst autograph.
  [(#1960)](https://github.com/PennyLaneAI/catalyst/pull/1960)

* `from_plxpr` can now handle dynamic shots and overridden device shots.
  [(#1983)](https://github.com/PennyLaneAI/catalyst/pull/1983/)

* `from_plxpr` can now translate `counts`.
  [(#2041)](https://github.com/PennyLaneAI/catalyst/pull/2041)

* QJitDevice helper `extract_backend_info` removed its redundant `capabilities` argument.
  [(#1956)](https://github.com/PennyLaneAI/catalyst/pull/1956)

* Raise warning when subroutines are used without capture enabled.
  [(#1930)](https://github.com/PennyLaneAI/catalyst/pull/1930)

* Update imports for noise transforms from `pennylane.transforms` to `pennylane.noise`.
  [(#1918)](https://github.com/PennyLaneAI/catalyst/pull/1918)
  [(#2020)](https://github.com/PennyLaneAI/catalyst/pull/2020)

* Improve error message for quantum subroutines when used outside a quantum context.
  [(#1932)](https://github.com/PennyLaneAI/catalyst/pull/1932)

* `from_plxpr` now supports adjoint and ctrl operations and transforms, operator
  arithmetic observables, `Hermitian` observables, `for_loop` outside qnodes, `cond` outside qnodes,
  `while_loop` outside QNode's, and `cond` with elif branches.
  [(#1844)](https://github.com/PennyLaneAI/catalyst/pull/1844)
  [(#1850)](https://github.com/PennyLaneAI/catalyst/pull/1850)
  [(#1903)](https://github.com/PennyLaneAI/catalyst/pull/1903)
  [(#1896)](https://github.com/PennyLaneAI/catalyst/pull/1896)
  [(#1889)](https://github.com/PennyLaneAI/catalyst/pull/1889)
  [(#1973)](https://github.com/PennyLaneAI/catalyst/pull/1973)

* The `qec.layer` and `qec.yield` operations have been added to the QEC dialect to represent a group
  of QEC operations. The main use case is to analyze the depth of a circuit.
  Also, this is a preliminary step towards supporting parallel execution of QEC layers.
  [(#1917)](https://github.com/PennyLaneAI/catalyst/pull/1917)

* Conversion patterns for the single-qubit `quantum.alloc_qb` and `quantum.dealloc_qb` operations
  have been added for lowering to the LLVM dialect. These conversion patterns allow for execution of
  programs containing these operations.
  [(#1920)](https://github.com/PennyLaneAI/catalyst/pull/1920)

* The default compilation pipeline is now available as `catalyst.pipelines.default_pipeline()`. The
  function `catalyst.pipelines.get_stages()` has also been removed as it was not used and duplicated
  the `CompileOptions.get_stages()` method.
  [(#1941)](https://github.com/PennyLaneAI/catalyst/pull/1941)

* Utility functions for modifying an existing compilation pipeline have been added to the
  `catalyst.pipelines` module.
  [(#1941)](https://github.com/PennyLaneAI/catalyst/pull/1941)

  These functions provide a simple interface to insert passes and stages into a compilation
  pipeline. The available functions are `insert_pass_after`, `insert_pass_before`,
  `insert_stage_after`, and `insert_stage_before`. For example,

  ```pycon
  >>> from catalyst.pipelines import insert_pass_after
  >>> pipeline = ["pass1", "pass2"]
  >>> insert_pass_after(pipeline, "new_pass", ref_pass="pass1")
  >>> pipeline
  ['pass1', 'new_pass', 'pass2']
  ```

* A new built-in compilation pipeline for experimental MBQC workloads has been added, available as
  `catalyst.ftqc.mbqc_pipeline()`.
  [(#1942)](https://github.com/PennyLaneAI/catalyst/pull/1942)

  The output of this function can be used directly as input to the `pipelines` argument of
  :func:`~.qjit`, for example,

  ```python
  from catalyst.ftqc import mbqc_pipeline

  @qjit(pipelines=mbqc_pipeline())
  @qml.qnode(dev)
  def workload():
      ...
  ```

* The `mbqc.graph_state_prep` operation has been added to the MBQC dialect. This operation prepares
  a graph state with arbitrary qubit connectivity, specified by an input adjacency-matrix operand,
  for use in MBQC workloads.
  [(#1965)](https://github.com/PennyLaneAI/catalyst/pull/1965)

* `catalyst.accelerate`, `catalyst.debug.callback`, and `catalyst.pure_callback`, `catalyst.debug.print`, and `catalyst.debug.print_memref` now work when capture is enabled.
  [(#1902)](https://github.com/PennyLaneAI/catalyst/pull/1902)

* The merge rotation pass in Catalyst (:func:`~.passes.merge_rotations`) now also considers
  `qml.Rot` and `qml.CRot`.
  [(#1955)](https://github.com/PennyLaneAI/catalyst/pull/1955)

* Catalyst now supports *array-backed registers*, meaning that `quantum.insert` operations can be
  configured to allow for the insertion of a qubit into an arbitrary position within a register.
  [(#2000)](https://github.com/PennyLaneAI/catalyst/pull/2000)

  This feature is disabled by default. To enable it, configure the pass pipeline to set the
  `use-array-backed-registers` option of the `convert-quantum-to-llvm` pass to `true`. For example,

  ```console
  catalyst --tool=opt --pass-pipeline="builtin.module(convert-quantum-to-llvm{use-array-backed-registers=true})" <input file>
  ```

* Fix auxiliary qubit deallocation in `decompose-non-clifford-ppr` pass
  in the `clifford-corrected` method.
  [(#2039)](https://github.com/PennyLaneAI/catalyst/pull/2039)

* The `NoMemoryEffect` trait has been removed from the `quantum.alloc` operation.
  [(#2044)](https://github.com/PennyLaneAI/catalyst/pull/2044)

* Enhance `ppm_specs` function to prevent duplicate pass addition
  [(#2049)](https://github.com/PennyLaneAI/catalyst/pull/2049)

* Added ``--ppr-to-mbqc`` to lower ``qec.ppr``/``qec.ppm`` into an MBQC-style quantum circuit.
  [(#2057)](https://github.com/PennyLaneAI/catalyst/pull/2057)

  This pass is part of a bottom-of-stack MBQC execution pathway, with a thin shim between the
  PPR/PPM layer and MBQC to enable end-to-end compilation on a mocked backend.  Also, in an MBQC gate
  set, one of the gate `RotXZX` cannot yet be executed on available backends.

  ```python
  import pennylane as qml
  from catalyst import qjit, measure
  from catalyst.passes import ppr_to_mbqc, to_ppr

  pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

  @qjit(target="mlir", pipelines=pipeline)
  @ppr_to_mbqc
  @to_ppr
  @qml.qnode(qml.device("lightning.qubit", wires=2))
  def circuit():
      qml.CNOT(wires=[0, 1])
      qml.T(0)
      return measure(0)

  print(circuit.mlir_opt)
  ```

<h3>Documentation 📝</h3>

* Typos were fixed and supplemental information was added to the
  docstrings for ``ppm_compilaion``, ``to_ppr``, ``commute_ppr``,
  ``ppr_to_ppm``, ``merge_ppr_ppm``, and ``ppm_specs``.
  [(#2050)](https://github.com/PennyLaneAI/catalyst/pull/2050)

* The Catalyst Command Line Interface documentation incorrectly stated that the `catalyst`
  executable is available in the `catalyst/bin/` directory relative to the environment's
  installation directory when installed via pip. The documentation has been updated to point to the
  correct location, which is the `bin/` directory relative to the environment's installation
  directory.
  [(#2030)](https://github.com/PennyLaneAI/catalyst/pull/2030)

* Fixing a few typos in the Catalyst documentation.
  [(#2046)](https://github.com/PennyLaneAI/catalyst/pull/2046)

* Updated `Examples` links to point to relevant, up-to-date demos and removed outdated entries.
  [(#2042)](https://github.com/PennyLaneAI/catalyst/pull/2042)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Joey Carter,
Yushao Chen,
Isaac De Vlugt,
Sengthai Heng,
David Ittah,
Jeffrey Kam,
Christina Lee,
Joseph Lee,
Andrija Paurevic,
Justin Pickering,
Ritu Thombre,
Roberto Turrado,
Paul Haochen Wang,
Jake Zaia,
Hongsheng Zheng.
