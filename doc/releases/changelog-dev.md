# Release 0.14.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

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


<h3>Internal changes ⚙️</h3>

* Refactor Catalyst pass registering so that it's no longer necessary to manually add new
  passes at `registerAllCatalystPasses`.
  [(#1984)](https://github.com/PennyLaneAI/catalyst/pull/1984)

* Split `from_plxpr.py` into two files.
  [(#2142)](https://github.com/PennyLaneAI/catalyst/pull/2142)	

* Re-work `DataView` to avoid an axis of size 0 possibly triggering a segfault via an underflow
  error, as discovered in 
  [this comment](https://github.com/PennyLaneAI/catalyst/pull/1598#issuecomment-2779178046).
  [(#1621)](https://github.com/PennyLaneAI/catalyst/pull/2164)

<h3>Documentation 📝</h3>

* A typo in the code example for :func:`~.passes.ppr_to_ppm` has been corrected.
  [(#2136)](https://github.com/PennyLaneAI/catalyst/pull/2136)

* Fix `catalyst.qjit` and `catalyst.CompileOptions` docs rendering.
  [(#2156)](https://github.com/PennyLaneAI/catalyst/pull/2156)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Christina Lee,
River McCubbin,
Roberto Turrado,
Paul Haochen Wang,
Hongsheng Zheng.
