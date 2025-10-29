# Release 0.14.0 (development release)

<h3>New features since last release</h3>

* Catalyst now supports Pennylane frontend operations `qml.PauliRot` and `qml.pauli_measure` 
  directly. This allows uses to write circuits in Pennylane consisting of PPR/PPMs, and 
  use the existing compilation passes in Catalyst specifically for PPR/PPMs, namely `to_ppr`, 
  `commute_ppr`, `merge_ppr_ppm`, `ppr_to_ppm`, and `ppm_compilation`.

  This is supported with both program capture enabled and disabled. However, there are several
  caveats. When program capture is disabled, we will not be able to use conditionals (i.e, 
  `qml.cond`) on the measurement result of `qml.pauli_measure`.

  To support this feature, we introduced a dummy device `catalyst.ftqc`. Users of this new 
  feature would need to set the `qnode` to use this dummy device. For example, a user can 
  now write the following circuit and have it compatible with the existing Catalyst PPR/PPM
  passes.

  ```python
  dev = qml.device("catalyst.ftqc", wires=1)
  pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]
  
  @qjit(pipelines=pipeline, target="mlir")
  @ppm_compilation
  @qml.qnode(device=dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.PauliRot(np.pi / 2, "X", wires=0)
      qml.PauliRot(np.pi / 4, "Y", wires=0)
      qml.T(wires=0)
      qml.pauli_measure("X", wires=0)
  ```
  [(#2145)](https://github.com/PennyLaneAI/catalyst/pull/2145)

<h3>Improvements 🛠</h3>

* A new option ``use_nameloc`` has been added to :func:`~.qjit` that embeds variable names
  from Python into the compiler IR, which can make it easier to read when debugging programs.
  [(#2054)](https://github.com/PennyLaneAI/catalyst/pull/2054)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixes the translation of plxpr control flow for edge cases where the `consts` were being
  reordered.
  [(#2128)](https://github.com/PennyLaneAI/catalyst/pull/2128)
  [(#2133)](https://github.com/PennyLaneAI/catalyst/pull/2133)

<h3>Internal changes ⚙️</h3>

* Refactor Catalyst pass registering so that it's no longer necessary to manually add new
  passes at `registerAllCatalystPasses`.
  [(#1984)](https://github.com/PennyLaneAI/catalyst/pull/1984)

* Split `from_plxpr.py` into two files.
  [(#2142)](https://github.com/PennyLaneAI/catalyst/pull/2142)

<h3>Documentation 📝</h3>

* A typo in the code example for :func:`~.passes.ppr_to_ppm` has been corrected.
  [(#2136)](https://github.com/PennyLaneAI/catalyst/pull/2136)

* Fix `catalyst.qjit` and `catalyst.CompileOptions` docs rendering.
  [(#2156)](https://github.com/PennyLaneAI/catalyst/pull/2156)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Jeffrey Kam,
Christina Lee,
Roberto Turrado,
Paul Haochen Wang.
