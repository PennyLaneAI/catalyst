# Release 0.9.0-dev

<h3>New features</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Those functions calling the `gather_p` primitive (like `jax.scipy.linalg.expm`)
  can now be used in multiple qjits in a single program.
  [(#1096)](https://github.com/PennyLaneAI/catalyst/pull/1096)

<h3>Internal changes</h3>

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when 
  running gradient like functions.
  
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Romain Moyard