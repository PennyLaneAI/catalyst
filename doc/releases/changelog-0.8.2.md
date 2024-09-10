# Release 0.8.2

<h3>New features</h3>

* Shot-vector support for Catalyst: Introduces support for shot-vectors in Catalyst, currently available for
  `qml.sample` measurements in the `lightning.qubit` device. Shot-vectors now allow elements of the form
  `((20, 5),)`, which is equivalent to `(20,)*5` or `(20, 20, 20, 20, 20)`. Furthermore, multiple `qml.sample`
  calls can now be returned from the same program, and can be structured using Python containers. For example,
  a program can return a dictionary like `return {"first": qml.sample(), "second": qml.sample()}`.
  [(#1051)](https://github.com/PennyLaneAI/catalyst/pull/1051)

<h3>Improvements</h3>

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires 3 steps: `gradient-preprocessing`, 
  `gradient-bufferize`, and `gradient-postprocessing`. `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`. 
  [(#1139)](https://github.com/PennyLaneAI/catalyst/pull/1139)

<h3>Internal changes</h3>

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when 
  running gradient like functions.
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

* Functions with multiple tapes are now split with a new mlir pass `--split-multiple-tapes`, with one tape per function. 
  The reset routine that makes a maeasurement between tapes and inserts a X gate if measured one is no longer used.
  [(#1017)](https://github.com/PennyLaneAI/catalyst/pull/1017)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Paul Haochen Wang,
Erick Ochoa Lopez,
Raul Torres,
Tzung-Han Juang.
