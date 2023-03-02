Benchmark
=========

This folder contains a set of benchmarking procedures comparing Catalyst with
PennyLane in various configurations.


Setup
-----

1. Git-clone the Catalyst repository locally and follow the regular installation
   procedure.
2. Install the additional benchmark dependencies with Pip:

   ``` shell
   $ pip install --user -r benchmark/requirements.txt
   ```

3. Add the `benchmark` folder to the `PYTHONPATH`:

   ``` shell
   # From the Catalyst folder
   $ export PYTHONPATH=$(pwd)/benchmark:$PYTHONPATH
   ```

4. Optionally, run the self-check:

   ``` shell
   $ python3 -m catalyst_benchmark.main selfcheck
   ```

Running
-------

* `./batchrun.py` is the top-level script supporting `compile` (time) and
  `runtime` measurements for each of the four categories described in the
  [guidelines](https://www.notion.so/xanaduai/Benchmarking-Strategy-07492f9104724f1984d09315e2e5ff0e):

  ```shell
  $ python3 batchrun.py --measure runtime --category regular
  ```

  The script runs series of measurements and saves results to the
  hardcoded `./_benchmark/` folder as JSON files. The `--dry-run` flag is
  supported.

* `catalyst_benchmark.main` runs a number of measurements of a single value in a
  fixed configuration as specified by its command line arguemnts. The full
  command has the following structure:

  ``` shell
  python3 -m catalyst_benchmark.main run \
      -p PROBLEM -m MEASURE -i IMPLEMENTATION \
      <problem-specific-options>
  ```

  Passing `?` to `-p`,`-m` or `-i` prints the list of supported parameters.

  The `IMPLEMENTATION` field has the following format:
  `(pennylane[+jax]|catalyst)/(device_name)`

  Examples:

  ``` shell
  $ python3 -m catalyst_benchmark.main run -p grover -m compile -i pennylane+jax/default.qubit.jax
  $ python3 -m catalyst_benchmark.main run -p vqe -m runtime -i catalyst/lightning.qubit
  ```

Extending
---------

1. In order to add a new problem, one typically needs to provide its `Catalyst`
   and `Pennylane` implementations as separate files. Each implementation should
   export the following global entities:

   * A descendant of the `Problem` class with the `trial_params` method:

     ``` python
     from .types import Problem

     class MyProblem(Problem):
         def __init__(self, dev, params, **qnode_kwargs):
             super().__init__(dev, **qnode_kwargs)
             # Access `self.nqubits` to scale the problem
             # Access `params` to scale the problem
             # Initialize the problem-specific state

         def trial_params(self, i:int)->Any:
             """ Return problem-specific parameters for the trial `i`. The
             result type is typically a `pnp.array` or `jnp.array`. """
             # Initialize the problem parameters
             return pnp.array(....) # or `jnp.array(...)`
     ```

   * The main algorithm function matching the following signature:

     `def algorithm(p:Problem, params:NumpyArray) -> NumpyArray`

     where `NumpyArray` is tupically a `np.array`, `pnp.array` or `jnp.array`,
     depending on the implementation. The requirements for the algorithm
     function are:

     - Algorithm function is allowed to initialize arbitrary number of
       `qml.node` with the device `p.dev` and parameters `p.qnode_kwargs`
       whenever requred.
     - Algorithm function is expected to be compilable with `jax.jit` or `qjit`
       as applicable.
     - Algorithm function is expected to return a _deterministic_ numeric
       result. The result should be the same across all the implementations.


2. Modify each of the
   `measure_{compile,runtime}_{catalyst,pennylane,pennylanejax}` functions
   defined in `main.py` (currently there are six functions).
   The `main.py` module generally uses the following pattern for performing the
   measurements:

   ```python
   from .my_problem import MyProblem, algorithm

   p = MyProblem(qml.device(...), ...)

   def problem(params):
       return algorithm(p, params)

   compiled_problem = compile(problem)

   for i in range(ntrials):
       params = p.trial_params(i)

       b = time()
       compiled_problem(params)
       e = time()
   ```

3. Modify the `selfcheck` function, compare the numeric results of
   the implementations.


Roadmap
-------

Implementation checklist:

* [x] Implement the problem runner script `catalyst_bechmark/main.py`
  - [x] Core problem running and measurement procedures
  - [x] Collect and record basic system information (cpu, ram, etc.)
  - [x] Collect the effective number of quantum gates (circuit depth)
* [ ] Implement the top-level wrapper script `batchrun.py`
  - [x] Data collection and serialization routines
  - [ ] Visualization
    + [x] Generic visualization procedure
    + [ ] Circuit depth visualization

Guidelines coverage:
*The marks below represent the implementation status. The data records require verification and are subject to change.*

* [x] Regulat circuits (the Grover-like task with the default number of layers and varying number of qubits)
  - [x] Catalyst
  - [x] PennyLane+jax / lightning.qubit
  - [x] PennyLane+jax / default.qubit.jax
  - [x] PennyLane / default.qubit
  - [x] PennyLane / lightning.qubit
* [x] Deep circuits (the Grover-like task with varying number of layers and fixed (==7) number of qubits)
  - [x] Catalyst
  - [x] PennyLane+jax / lightning.qubit
  - [x] PennyLane+jax / default.qubit.jax
  - [x] PennyLane / default.qubit
  - [x] PennyLane / lightning.qubit
* [ ] Hybrid workflows
* [ ] Variational algorithms (the VQE-like task with varying number of qubits)
  - [x] Catalyst
  - [ ] PennyLane+jax / lightning.qubit
  - [ ] PennyLane+jax / default.qubit.jax
  - [x] PennyLane / default.qubit
  - [ ] PennyLane / lightning.qubit
  - [ ] Vary gradient methods

Known issues:

* [ ]  PennyLane+jax/default.qubit.jax implementation of Grover-like task: compilation and running times are not clearly distinguishable
* [ ] PennyLane+jax implementation of VQE problem does not currently work
* [ ] Altair visualization: plot legend is clipped, see https://github.com/vega/vl-convert/issues/30 The best workaround I am aware of  is to provide shorter aliases for implementations.

