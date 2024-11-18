# Release 0.3.1

<h3>New features</h3>

* The experimental AutoGraph feature, now supports Python `for` loops, allowing native Python loops
  to be captured and compiled with Catalyst.
  [(#258)](https://github.com/PennyLaneAI/catalyst/pull/258)

  ```python
  dev = qml.device("lightning.qubit", wires=n)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(n):
      for i in range(n):
          qml.Hadamard(wires=i)

      return qml.expval(qml.PauliZ(0))
  ```

  This feature extends the existing AutoGraph support for Python `if` statements introduced in v0.3.
  Note that TensorFlow must be installed for AutoGraph support.

* The quantum control operation can now be used in conjunction with Catalyst control flow, such as
  loops and conditionals, via the new `catalyst.ctrl` function.
  [(#282)](https://github.com/PennyLaneAI/catalyst/pull/282)

  Similar in behaviour to the `qml.ctrl` control modifier from PennyLane, `catalyst.ctrl` can
  additionally wrap around quantum functions which contain control flow, such as the Catalyst
  `cond`, `for_loop`, and `while_loop` primitives.

  ```python
  @qjit
  @qml.qnode(qml.device("lightning.qubit", wires=4))
  def circuit(x):

      @for_loop(0, 3, 1)
      def repeat_rx(i):
          qml.RX(x / 2, wires=i)

      catalyst.ctrl(repeat_rx, control=3)()

      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(0.2)
  array(1.)
  ```

* Catalyst now supports JAX's `array.at[index]` notation for array element assignment and updating.
  [(#273)](https://github.com/PennyLaneAI/catalyst/pull/273)

  ```python
  @qjit
  def add_multiply(l: jax.core.ShapedArray((3,), dtype=float), idx: int):
      res = l.at[idx].multiply(3)
      res2 = l.at[idx].add(2)
      return res + res2

  res = add_multiply(jnp.array([0, 1, 2]), 2)
  ```

  ```pycon
  >>> res
  [0, 2, 10]
  ```

  For more details on available methods, see the
  [JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html).

<h3>Improvements</h3>

* The Lightning backend device has been updated to work with the new PL-Lightning monorepo.
  [(#259)](https://github.com/PennyLaneAI/catalyst/pull/259)
  [(#277)](https://github.com/PennyLaneAI/catalyst/pull/277)

* A new compiler driver has been implemented in C++. This improves compile-time performance by
  avoiding *round-tripping*, which is when the entire program being compiled is dumped to
  a textual form and re-parsed by another tool.

  This is also a requirement for providing custom metadata at the LLVM level, which is
  necessary for better integration with tools like Enzyme. Finally, this makes it more natural
  to improve error messages originating from C++ when compared to the prior subprocess-based
  approach.
  [(#216)](https://github.com/PennyLaneAI/catalyst/pull/216)

* Support the `braket.devices.Devices` enum class and `s3_destination_folder`
  device options for AWS Braket remote devices.
  [(#278)](https://github.com/PennyLaneAI/catalyst/pull/278)

* Improvements have been made to the build process, including avoiding unnecessary processes such
  as removing `opt` and downloading the wheel.
  [(#298)](https://github.com/PennyLaneAI/catalyst/pull/298)

* Remove a linker warning about duplicate `rpath`s when Catalyst wheels are installed on macOS.
  [(#314)](https://github.com/PennyLaneAI/catalyst/pull/314)

<h3>Bug fixes</h3>

* Fix incompatibilities with GCC on Linux introduced in v0.3.0 when compiling user programs.
  Due to these, Catalyst v0.3.0 only works when clang is installed in the user environment.

  - Resolve an issue with an empty linker flag, causing `ld` to error.
    [(#276)](https://github.com/PennyLaneAI/catalyst/pull/276)

  - Resolve an issue with undefined symbols provided the Catalyst runtime.
    [(#316)](https://github.com/PennyLaneAI/catalyst/pull/316)

* Remove undocumented package dependency on the zlib/zstd compression library.
  [(#308)](https://github.com/PennyLaneAI/catalyst/pull/308)

* Fix filesystem issue when compiling multiple functions with the same name and
  `keep_intermediate=True`.
  [(#306)](https://github.com/PennyLaneAI/catalyst/pull/306)

* Add support for applying the `adjoint` operation to `QubitUnitary` gates.
  `QubitUnitary` was not able to be `adjoint`ed when the variable holding the unitary matrix might
  change. This can happen, for instance, inside of a for loop.
  To solve this issue, the unitary matrix gets stored in the array list via push and pops.
  The unitary matrix is later reconstructed from the array list and `QubitUnitary` can be executed
  in the `adjoint`ed context.
  [(#304)](https://github.com/PennyLaneAI/catalyst/pull/304)
  [(#310)](https://github.com/PennyLaneAI/catalyst/pull/310)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Erick Ochoa Lopez,
Jacob Mai Peng,
Sergei Mironov,
Romain Moyard.