# Release 0.8.2

<h3>Improvements</h3>

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires 3 steps: `gradient-preprocessing`, 
  `gradient-bufferize`, and `gradient-postprocessing`. `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`. 
  [(#1139)](https://github.com/PennyLaneAI/catalyst/pull/1139)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Erick Ochoa Lopez,
Raul Torres,
Tzung-Han Juang.
