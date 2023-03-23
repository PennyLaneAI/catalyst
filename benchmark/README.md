Benchmark
=========

This folder contains a set of benchmarking procedures comparing Catalyst with
PennyLane in various configurations.


Setup
-----

1. Git-clone the Catalyst repository locally and follow the regular installation
   procedure.
2. Install the additional dependencies:

   * Python dependencies:
     ``` shell
     $ pip install --user -r benchmark/requirements.txt
     ```

   * LaTeX dependencies: `texlive` package needs to be installed using the system package manager.

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

### Full cycle measurements

1. Run the `./batchrun.py`.
   ``` sh
   $ python3 batchrun.py --tag=today --timeout-1run=4000
   ```
   - The data is collected incrementally.
   - Data collection and post-processing operations could be separated using the `-a|--action`
     argument. The `collect` and `plot` operations are supported.
   - If the script is not running on the same machine that was used to collect the data, call the
     `plot` action with the `-H|--force-sysinfo-hash` arguemnt specifying the measurement system's
     sysinfo hash.

2. Render the report by copying the `./tex/report_template.tex`, adjusting the measurement tag
   and/or the system information hash, adding comments and finally running the `sh/mkpdflatex.sh`
   rendering script.
   ``` sh
   $ cp ./tex/report_template.tex ./tex/report.tex
   $ $EDITOR ./tex/report.tex # ... edit `\SYSHASH`, `\TAG`, etc.
   $ ./sh/mkpdflatex.sh ./tex/report.tex
   ```
   Grab the `./tex/report.pdf`.

### Running a single measurement

* `./benchmark.py` measures a specific time value in a
  fixed configuration as specified by its command line arguemnts. The full
  command has the following structure:

  ``` sh
  $ python3 benchmark.py run \
      -p PROBLEM -m MEASURE -i IMPLEMENTATION \
      <problem-specific-options>
  ```

Passing `?` to `-p`,`-m` or `-i` prints the list of supported parameters.

The `IMPLEMENTATION` field has the following format:
`(pennylane[+jax]|catalyst)/(device_name)`

Examples:

``` shell
$ python3 benchmark.py run -p grover -m compile -i pennylane+jax/default.qubit.jax
$ python3 benchmark.py run -p chemvqe -m runtime -i catalyst/lightning.qubit
```

Extending
---------

1. In order to add a new problem, one typically needs to provide its `Catalyst` and `Pennylane`
   implementations as separate files in the `test_cases` subfolder. Each implementation should
   export the following global entities:

   * A subclass of the `Problem` class with the `trial_params` method:

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
   * The `qcompile` function matching the following signature:

     `def qcompile(p:Problem, params:NumpyArray) -> None`

     The function is expected to compile the quantum parts of the problem.

   * The `workflow` function matching the following signature:

     `def workflow(p:Problem, params:NumpyArray) -> NumpyArray`

   * Optionally, the `depth` function, calculating the depth of the problem circuits in quantum
     gates.

   Here, `NumpyArray` is typically a `np.array`, `pnp.array` or `jnp.array`,
   depending on the implementation.


2. Modify each of the
   `measurements.measure_{compile,runtime}_{catalyst,pennylane,pennylanejax}` functions
   defined in `main.py` (currently there are six functions).
   The `main.py` module generally uses the following pattern for performing the
   measurements:

   ```python
   import framework
   from .test_cases.my_problem import Problem, qcompile, workflow

   p = Problem(qml.device(...), ...)

   def _main(params):
       qcompile(p, params)
       return algorithm(p, params)

   compiled_main = framework.compile(_main)

   for i in range(ntrials):
       params = p.trial_params(i)

       b = time()
       compiled_main(params)
       e = time()
   ```

3. Modify the `measurements.selfcheck` function, compare the numeric results across the
   the implementations.

