# Release 0.4.1

<h3>Improvements</h3>

* Catalyst wheels are now packaged with OpenMP and ZStd, which avoids installing additional
  requirements separately in order to use pre-packaged Catalyst binaries.
  [(#457)](https://github.com/PennyLaneAI/catalyst/pull/457)
  [(#478)](https://github.com/PennyLaneAI/catalyst/pull/478)

  Note that OpenMP support for the `lightning.kokkos` backend has been disabled on macOS x86_64, due
  to memory issues in the computation of Lightning's adjoint-jacobian in the presence of multiple
  OMP threads.

<h3>Bug fixes</h3>

* Resolve an infinite recursion in the decomposition of the `Controlled`
  operator whenever computing a Unitary matrix for the operator fails.
  [(#468)](https://github.com/PennyLaneAI/catalyst/pull/468)

* Resolve a failure to generate gradient code for specific input circuits.
  [(#439)](https://github.com/PennyLaneAI/catalyst/pull/439)

  In this case, `jnp.mod`
  was used to compute wire values in a for loop, which prevented the gradient
  architecture from fully separating quantum and classical code. The following
  program is now supported:
  ```py
  @qjit
  @grad
  @qml.qnode(dev)
  def f(x):
      def cnot_loop(j):
          qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

      for_loop(0, 4, 1)(cnot_loop)()

      return qml.expval(qml.PauliZ(0))
  ```

* Resolve unpredictable behaviour when importing libraries that share Catalyst's LLVM dependency
  (e.g. TensorFlow). In some cases, both packages exporting the same symbols from their shared
  libraries can lead to process crashes and other unpredictable behaviour, since the wrong functions
  can be called if both libraries are loaded in the current process.
  The fix involves building shared libraries with hidden (macOS) or protected (linux) symbol
  visibility by default, exporting only what is necessary.
  [(#465)](https://github.com/PennyLaneAI/catalyst/pull/465)

* Resolve a failure to find the SciPy OpenBLAS library when running Catalyst,
  due to a different SciPy version being used to build Catalyst than to run it.
  [(#471)](https://github.com/PennyLaneAI/catalyst/pull/471)

* Resolve a memory leak in the runtime stemming from  missing calls to device destructors
  at the end of programs.
  [(#446)](https://github.com/PennyLaneAI/catalyst/pull/446)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah.
