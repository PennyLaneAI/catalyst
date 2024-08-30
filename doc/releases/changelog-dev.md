# Release 0.9.0-dev

<h3>New features</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Circuits with preprocessing functions outside qnodes can now be differentiated.
  [(#332)](https://github.com/PennyLaneAI/catalyst/pull/332)

  ```python
  @qml.qnode(qml.device("lightning.qubit", wires=1))
  def f(y):
      qml.RX(y, wires=0)
      return qml.expval(qml.PauliZ(0))

  @catalyst.qjit
  def g(x):
      return catalyst.grad(lambda y: f(jnp.cos(y)) ** 2)(x)
  ```

  ```pycon
  >>> g(0.4)
  0.3751720385067584
  ```

<h3>Internal changes</h3>

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when 
  running gradient like functions.
  
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Romain Moyard