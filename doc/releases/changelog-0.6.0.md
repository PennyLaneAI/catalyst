# Release 0.6.0

<h3>New features</h3>

* Catalyst now supports externally hosted callbacks with parameters and return values
  within qjit-compiled code. This provides the ability to insert native Python code
  into any qjit-compiled function, allowing for the capability to include subroutines
  that do not yet support qjit-compilation and enhancing the debugging experience.
  [(#540)](https://github.com/PennyLaneAI/catalyst/pull/540)
  [(#596)](https://github.com/PennyLaneAI/catalyst/pull/596)
  [(#610)](https://github.com/PennyLaneAI/catalyst/pull/610)
  [(#650)](https://github.com/PennyLaneAI/catalyst/pull/650)
  [(#649)](https://github.com/PennyLaneAI/catalyst/pull/649)
  [(#661)](https://github.com/PennyLaneAI/catalyst/pull/661)
  [(#686)](https://github.com/PennyLaneAI/catalyst/pull/686)
  [(#689)](https://github.com/PennyLaneAI/catalyst/pull/689)

  The following two callback functions are available:

  - `catalyst.pure_callback` supports callbacks of **pure** functions. That is, functions
    with no [side-effects](https://runestone.academy/ns/books/published/fopp/Functions/SideEffects.html) that accept parameters and return values. However, the return
    type and shape of the function must be known in advance, and is provided as a type signature.

    ```python
    @pure_callback
    def callback_fn(x) -> float:
        # here we call non-JAX compatible code, such
        # as standard NumPy
        return np.sin(x)

    @qjit
    def fn(x):
        return jnp.cos(callback_fn(x ** 2))
    ```
    ```pycon
    >>> fn(0.654)
    array(0.9151995)
    ```

  - `catalyst.debug.callback` supports callbacks of functions with **no** return values. This makes it
    an easy entry point for debugging, for example via printing or logging at runtime.

    ```python
    @catalyst.debug.callback
    def callback_fn(y):
        print("Value of y =", y)

    @qjit
    def fn(x):
        y = jnp.sin(x)
        callback_fn(y)
        return y ** 2
    ```
    ```pycon
    >>> fn(0.54)
    Value of y = 0.5141359916531132
    array(0.26433582)
    >>> fn(1.52)
    Value of y = 0.998710143975583
    array(0.99742195)
    ```

  Note that callbacks do not currently support differentiation, and cannot be used inside
  functions that `catalyst.grad` is applied to.

* More flexible runtime printing through support for format strings.
  [(#621)](https://github.com/PennyLaneAI/catalyst/pull/621)

  The `catalyst.debug.print` function has been updated to support Python-like format
  strings:

  ```python
  @qjit
  def cir(a, b, c):
      debug.print("{c} {b} {a}", a=a, b=b, c=c)
  ```

  ```pycon
  >>> cir(1, 2, 3)
  3 2 1
  ```

  Note that previous functionality of the print function to print out memory reference information
  of variables has been moved to `catalyst.debug.print_memref`.

* Catalyst now supports QNodes that execute on [Oxford Quantum Circuits (OQC)](https://www.oqc.tech/)
  superconducting hardware, via [OQC Cloud](https://docs.oqc.app).
  [(#578)](https://github.com/PennyLaneAI/catalyst/pull/578)
  [(#579)](https://github.com/PennyLaneAI/catalyst/pull/579)
  [(#691)](https://github.com/PennyLaneAI/catalyst/pull/691)

  To use OQC Cloud with Catalyst, simply ensure your credentials are set as environment variables,
  and load the `oqc.cloud` device to be used within your qjit-compiled workflows.

  ```python
  import os
  os.environ["OQC_EMAIL"] = "your_email"
  os.environ["OQC_PASSWORD"] = "your_password"
  os.environ["OQC_URL"] = "oqc_url"

  dev = qml.device("oqc.cloud", backend="lucy", shots=2012, wires=2)

  @qjit
  @qml.qnode(dev)
  def circuit(a: float):
      qml.Hadamard(0)
      qml.CNOT(wires=[0, 1])
      qml.RX(wires=0)
      return qml.counts(wires=[0, 1])

  print(circuit(0.2))
  ```

* Catalyst now ships with an instrumentation feature allowing to explore what steps are run during
  compilation and execution, and for how long.
  [(#528)](https://github.com/PennyLaneAI/catalyst/pull/528)
  [(#597)](https://github.com/PennyLaneAI/catalyst/pull/597)

  Instrumentation can be enabled from the frontend with the `catalyst.debug.instrumentation`
  context manager:

  ```pycon
  >>> @qjit
  ... def expensive_function(a, b):
  ...     return a + b
  >>> with debug.instrumentation("session_name", detailed=False):
  ...     expensive_function(1, 2)
  [DIAGNOSTICS] Running capture                   walltime: 3.299 ms      cputime: 3.294 ms       programsize: 0 lines
  [DIAGNOSTICS] Running generate_ir               walltime: 4.228 ms      cputime: 4.225 ms       programsize: 14 lines
  [DIAGNOSTICS] Running compile                   walltime: 57.182 ms     cputime: 12.109 ms      programsize: 121 lines
  [DIAGNOSTICS] Running run                       walltime: 1.075 ms      cputime: 1.072 ms
  ```

  The results will be appended to the provided file if the `filename` attribute is set, and printed
  to the console otherwise. The flag `detailed` determines whether individual steps in the compiler
  and runtime are instrumented, or whether only high-level steps like "program capture" and
  "compilation" are reported.

  Measurements currently include wall time, CPU time, and (intermediate) program size.

<h3>Improvements</h3>

* AutoGraph now supports return statements inside conditionals in qjit-compiled
  functions.
  [(#583)](https://github.com/PennyLaneAI/catalyst/pull/583)

  For example, the following pattern is now supported, as long as
  all return values have the same type:

  ```python
  @qjit(autograph=True)
  def fn(x):
      if x > 0:
          return jnp.sin(x)
      return jnp.cos(x)
  ```

  ```pycon
  >>> fn(0.1)
  array(0.09983342)
  >>> fn(-0.1)
  array(0.99500417)
  ```

  This support extends to quantum circuits:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(x: float):
    qml.RX(x, wires=0)

    m = catalyst.measure(0)

    if not m:
        return m, qml.expval(qml.PauliZ(0))

    qml.RX(x ** 2, wires=0)

    return m, qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> f(1.4)
  (array(False), array(1.))
  >>> f(1.4)
  (array(True), array(0.37945176))
  ```

  Note that returning results with different types or shapes within the same function, such as
  different observables or differently shaped arrays, is not possible.

* Errors are now raised at compile time if the gradient of an unsupported function
  is requested.
  [(#204)](https://github.com/PennyLaneAI/catalyst/pull/204)

  At the moment, `CompileError` exceptions will be raised if at compile time it is found that code
  reachable from the gradient operation contains either a mid-circuit measurement, a callback, or a
  JAX-style custom call (which happens through the mitigation operation as well as certain JAX operations).

* Catalyst now supports devices built from the
  [new PennyLane device API](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.Device.html).
  [(#565)](https://github.com/PennyLaneAI/catalyst/pull/565)
  [(#598)](https://github.com/PennyLaneAI/catalyst/pull/598)
  [(#599)](https://github.com/PennyLaneAI/catalyst/pull/599)
  [(#636)](https://github.com/PennyLaneAI/catalyst/pull/636)
  [(#638)](https://github.com/PennyLaneAI/catalyst/pull/638)
  [(#664)](https://github.com/PennyLaneAI/catalyst/pull/664)
  [(#687)](https://github.com/PennyLaneAI/catalyst/pull/687)

  When using the new device API, Catalyst will discard the preprocessing from the original device,
  replacing it with Catalyst-specific preprocessing based on the TOML file provided by the device.
  Catalyst also requires that provided devices specify their wires upfront.

* A new compiler optimization that removes redundant chains of self inverse operations has been
  added. This is done within a new MLIR pass called `remove-chained-self-inverse`. Currently we
  only match redundant Hadamard operations, but the list of supported operations can be expanded.
  [(#630)](https://github.com/PennyLaneAI/catalyst/pull/630)

* The `catalyst.measure` operation is now more lenient in the accepted type for the `wires` parameter.
  In addition to a scalar, a 1D array is also accepted as long as it only contains one element.
  [(#623)](https://github.com/PennyLaneAI/catalyst/pull/623)

  For example, the following is now supported:

  ```python
  catalyst.measure(wires=jnp.array([0]))
  ```

* The compilation & execution of `@qjit` compiled functions can now be aborted using an interrupt
  signal (SIGINT). This includes using `CTRL-C` from a command line and the `Interrupt` button in
  a Jupyter Notebook.
  [(#642)](https://github.com/PennyLaneAI/catalyst/pull/642)

* The Catalyst Amazon Braket support has been updated to work with the latest version of the
  Amazon Braket PennyLane plugin (v1.25.0) and Amazon Braket Python SDK (v1.73.3)
  [(#620)](https://github.com/PennyLaneAI/catalyst/pull/620)
  [(#672)](https://github.com/PennyLaneAI/catalyst/pull/672)
  [(#673)](https://github.com/PennyLaneAI/catalyst/pull/673)

  Note that with this update, all declared qubits in a submitted program will always be measured, even if specific qubits were never used.

* An updated quantum device specification format, TOML schema v2, is now supported by Catalyst. This
  allows device authors to specify properties such as native quantum control
  support, gate invertibility, and differentiability on a per-operation level.
  [(#554)](https://github.com/PennyLaneAI/catalyst/pull/554)

  For more details on the new TOML schema, please refer to the
  [custom devices documentation](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/custom_devices.html).

* An exception is now raised when OpenBLAS cannot be found by Catalyst during compilation.
  [(#643)](https://github.com/PennyLaneAI/catalyst/pull/643)

<h3>Breaking changes</h3>

* `qml.sample` and `qml.counts` now produce integer arrays for the sample array and basis state
  array when used without observables.
  [(#648)](https://github.com/PennyLaneAI/catalyst/pull/648)

* The endianness of counts in Catalyst now matches the convention of PennyLane.
  [(#601)](https://github.com/PennyLaneAI/catalyst/pull/601)

* `catalyst.debug.print` no longer supports the `memref` keyword argument.
  Please use `catalyst.debug.print_memref` instead.
  [(#621)](https://github.com/PennyLaneAI/catalyst/pull/621)

<h3>Bug fixes</h3>

* The QNode argument `diff_method=None` is now supported for QNodes within a qjit-compiled function.
  [(#658)](https://github.com/PennyLaneAI/catalyst/pull/658)

* A bug has been fixed where the C++ compiler driver was incorrectly being triggered twice.
  [(#594)](https://github.com/PennyLaneAI/catalyst/pull/594)

* Programs with `jnp.reshape` no longer fail.
  [(#592)](https://github.com/PennyLaneAI/catalyst/pull/592)

* A bug in the quantum adjoint routine in the compiler has been fixed, which didn't take into
  account control wires on operations in all instances.
  [(#591)](https://github.com/PennyLaneAI/catalyst/pull/591)

* A bug in the test suite causing stochastic autograph test failures has been fixed.
  [(#652)](https://github.com/PennyLaneAI/catalyst/pull/652)

* Running Catalyst tests should no longer raise `ResourceWarning` from the use of `tempfile.TemporaryDirectory`.
  [(#676)](https://github.com/PennyLaneAI/catalyst/pull/676)

* Raises an exception if the user has an incompatible CUDA Quantum version installed.
  [(#707)](https://github.com/PennyLaneAI/catalyst/pull/707)

<h3>Internal changes</h3>

* The deprecated `@qfunc` decorator, in use mainly by the LIT test suite, has been removed.
  [(#679)](https://github.com/PennyLaneAI/catalyst/pull/679)

* Catalyst now publishes a revision string under `catalyst.__revision__`, in addition
  to the existing `catalyst.__version__` string.
  The revision contains the Git commit hash of the repository at the time of packaging,
  or for editable installations the active commit hash at the time of package import.
  [(#560)](https://github.com/PennyLaneAI/catalyst/pull/560)

* The Python interpreter is now a shared resource across the runtime.
  [(#615)](https://github.com/PennyLaneAI/catalyst/pull/615)

  This change allows any part of the runtime to start executing Python code through pybind.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Romain Moyard,
Sergei Mironov,
Erick Ochoa Lopez,
Lee James O'Riordan,
Muzammiluddin Syed.
