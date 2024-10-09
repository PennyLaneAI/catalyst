
Debugging Tips
##############

Run-time Dependencies
=====================

There are some environment variables that will affect the execution of Catalyst.
For example, Catalyst relies on the ``PATH`` environment variable to be set, just like most other UNIX programs.
Most of the time, Catalyst should work without special configuration of the environment, however here are some errors and warnings you might see and how to fix them.

* ``Compiler clang failed during the execution of command ...``: During the linking process, Catalyst uses some well-known compilers that may be installed in your computer.
  If for whatever reason the compiler fails to execute, this warning will be issued.
  Please note that more attempts to link will be issued with different compilers.
  You may be able to fix this warning by re-installing or uninstalling the faulty compiler.
* ``Unable to link... All available compiler options exhausted ...``: Even though Catalyst has a list of compilers that may be available in your system, it is possible that no compiler successfully linked the program to the libraries shipped with Catalyst.
  This should be a rare case, but if you ever find yourself seeing this error, you should check that one of the following compilers or interfaces are available in your ``PATH``: ``clang``, ``gcc``, ``c99``, ``c89``, ``cc``.
  Correctly installing any of these should be sufficient.
  Alternatively, you can also specify a preferred compiler to use during linking using the environment variable ``CATALYST_CC``.
  For example ``CATALYST_CC=clang-14`` specifies that ``clang-14`` will be the preferred compiler to be used during the linking stage.
* ``User defined compiler is not in path``: Following the example above, you may have installed ``clang-14`` in a different directory that is not currently on the ``PATH``.
  Please make sure that the executable specified in ``CATALYST_CC`` is available in the ``PATH`` and is flag compatible with ``gcc`` or ``clang``.

Running without Python runtime
==============================

Sometimes, it might be useful to run the JIT compiled function without Python, for example in case of debugging.
If the program succeeds without Python, then it is likely that the error is found in the interface between Python and the JIT compiled function, or within Catalyst's Python internals.
As a quick sanity check you might want to run the JIT compiled function without Python.
Below is an example of how to obtain a C program that can be linked against the generated function:

.. code-block:: python

    @qjit
    def identity(x):
        return x

    print(debug.get_cmain(identity, 1.0))

Using the :func:`~.debug.get_cmain` function, the following string is returned to the user:

.. code-block:: C

    #include <complex.h>
    #include <stddef.h>
    #include <stdint.h>

    typedef int64_t int64;
    typedef double float64;
    typedef float float32;
    typedef double complex complex128;
    typedef float complex complex64;


    struct memref_float64x0_t
    {
        float64* allocated;
        float64* aligned;
        size_t offset;

    };
    struct result_t {
        struct memref_float64x0_t f_0;
    };


    extern void setup(int, char**);
    extern void _catalyst_ciface_jit_identity(struct result_t*, struct memref_float64x0_t*);
    extern void teardown();

    int
    main(int argc, char** argv)
    {

        struct result_t result_val;
        float64 buff_0 = 1.0;
        struct memref_float64x0_t arg_0 = { &buff_0, &buff_0, 0 };


        setup(1, &argv[0]);
        _catalyst_ciface_jit_identity(&result_val, &arg_0);
        teardown();
    }

The user can now compile and link this program and run it without Python.


Verbose Mode
============

Catalyst uses a number of command line tools under the hood.
In order to see how these tools are used, one can use the verbose mode.
The verbose mode prints out the tools and flags used by Catalyst.


In order to enable verbose mode, the user must use the keyword argument ``verbose`` and set it to ``True`` for the ``@qjit`` wrapper.
For example:

.. code-block:: python

    @qjit(verbose=True)
    def circuit():
        ...

Will print out something close to the following:

.. code-block:: bash

        [RUNNING] mlir-hlo-opt --allow-unregistered-dialect --canonicalize --chlo-legalize-to-hlo --stablehlo-legalize-to-hlo --mhlo-legalize-control-flow --hlo-legalize-to-linalg --mhlo-legalize-to-std --convert-to-signless --canonicalize /tmp/tmpwsoh3acq/circuit.mlir -o /tmp/tmpwsoh3acq/circuit.nohlo.mlir
        [RUNNING] quantum-opt --lower-gradients --convert-arraylist-to-memref /tmp/tmpwsoh3acq/circuit.nohlo.mlir -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.mlir
        [RUNNING] quantum-opt --inline --gradient-bufferize --scf-bufferize --convert-tensor-to-linalg --convert-elementwise-to-linalg --arith-bufferize --empty-tensor-to-alloc-tensor --bufferization-bufferize --tensor-bufferize --linalg-bufferize --tensor-bufferize --quantum-bufferize --func-bufferize --finalizing-bufferize --buffer-loop-hoisting --convert-bufferization-to-memref --canonicalize --cp-global-memref /tmp/tmpwsoh3acq/circuit.nohlo.opt.mlir -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.mlir
        [RUNNING] quantum-opt --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata --lower-affine --arith-expand --convert-complex-to-standard --convert-complex-to-llvm --convert-math-to-llvm --convert-math-to-libm --convert-arith-to-llvm --finalize-memref-to-llvm=use-generic-functions --convert-index-to-llvm --convert-gradient-to-llvm --convert-quantum-to-llvm --emit-catalyst-py-interface --canonicalize --reconcile-unrealized-casts /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.mlir -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.mlir
        [RUNNING] mlir-translate --mlir-to-llvmir /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.mlir -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.ll
        [RUNNING] llc --filetype=obj --relocation-model=pic /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.ll -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.o
        [RUNNING] clang -shared -rdynamic -Wl,-no-as-needed -Wl,-rpath,runtime/build/lib/capi:runtime/build/lib/backend:mlir/llvm-project/build/lib -Lmlir/llvm-project/build/lib -Lruntime/build/lib/capi -Lruntime/build/lib/backend -lrt_backend -lrt_capi -lpthread -lmlir_c_runner_utils /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.o -o /tmp/tmpwsoh3acq/circuit.nohlo.opt.buff.llvm.so


Pass Pipelines
==============

The compilation steps which take MLIR as an input and lower it to binary are broken into MLIR pass
pipelines.  The ``pipelines`` argument of the ``qjit`` function may be used to alter the steps used
for compilation. The default set of pipelines is defined via the ``catalyst.compiler.get_stages``
list.

One could customize what compilation passes are executed. A good use case of this would be if you
are debugging Catalyst itself or you want to enable or disable passes within a specific pipeline.
It is recommended to copy the default pipelines and edit them to suit your goals and afterwards
passing them to the ``@qjit`` decorator. E.g. if you want to disable inlining

.. code-block:: python

    my_pipelines = [
        ...
        (
            "MyBufferizationPass",
            [
                "one-shot-bufferize{dialect-filter=memref}",
                # "inline",
                "gradient-bufferize",
                ...
            ],
        ),
        ...
        ]

     @qjit(pipelines=my_pipelines, keep_intermediate=True)
     @qml.qnode(dev)
     def circuit():
        ...


Here, each item represents a pipeline. Each pipeline has a name and a list of MLIR passes
to perform. Most of the standard passes are described in the
`MLIR passes documentation <https://mlir.llvm.org/docs/Passes/>`_. Quantum MLIR passes are
implemented in Catalyst and can be found in the sources.

All pipelines are executed in sequence, the output MLIR of each non-empty pipeline is stored in
memory and becomes available via the :func:`~.debug.get_compilation_stage` function in the ``debug`` module.
It is necessary however, to have compiled with the option ``keep_intermediate=True`` to use
:func:`~.debug.get_compilation_stage`.

Printing the IR generated by Pass Pipelines
===========================================

We won't get into too much detail here, but sometimes it is useful to look at the output of a
specific pass pipeline.
To do so, simply use the :func:`~.debug.get_compilation_stage` function and print the return value out.
For example, if one wishes to inspect the output of the ``BufferizationPass`` pipeline, simply run
the following command.

.. code-block:: python

    print(get_compilation_stage(circuit, "BufferizationPass")

Profiling and instrumentation
=============================

Catalyst features built-in instrumentation capabilities for the compiler, which can
be used to explore which steps are run by the compiler (and certain runtime functions),
and for how long.

The instrumentation can be enabled from the frontend with the :func:`~.debug.instrumentation`
context manager:

>>> @qjit
... def expensive_function(a, b):
...     return a + b
>>> with debug.instrumentation("session_name", detailed=False):
...     expensive_function(1, 2)
[DIAGNOSTICS] Running capture                   walltime: 3.299 ms      cputime: 3.294 ms       programsize: 0 lines
[DIAGNOSTICS] Running generate_ir               walltime: 4.228 ms      cputime: 4.225 ms       programsize: 14 lines
[DIAGNOSTICS] Running compile                   walltime: 57.182 ms     cputime: 12.109 ms      programsize: 121 lines
[DIAGNOSTICS] Running run                       walltime: 1.075 ms      cputime: 1.072 ms

Measurements currently include wall time, CPU time, and (intermediate) program size;
please refer to the docstring for more details.

Compilation Steps
=================

The compilation process of a QJITed quantum function moves through various stages of the compilation pipeline including:

- **Quantum Tape**: the quantum record of hybrid quantum programs in a single ``qml.QNode``
- **JAXPR**: the graph data structure maintained by `JAX <https://github.com/google/jax>`_ for the classical & quantum parts of the compiled program
- **MLIR**: a novel compiler framework and intermediate representation
- **HLO (XLA) + Quantum Dialect**: Lowering to `HLO <https://github.com/tensorflow/mlir-hlo>`_ is the first stage inside MLIR after leaving JAXPR.
- **Builtin + Quantum Dialects**: HLO is then converted to a variety of classical dialects in MLIR.
- **Bufferized MLIR**: All tensors are `converted <https://mlir.llvm.org/docs/Bufferization>`_ to memory buffer allocations at this step.
- **LLVM Dialect**: Lowering the code to the `LLVM Dialect <https://mlir.llvm.org/docs/Dialects/LLVM/>`_ in MLIR simplifies the translation to LLVMIR by providing a one-to-one mapping.
- **QIR (LLVMIR)**: a `specification <https://learn.microsoft.com/en-us/azure/quantum/concepts-qir>`_ for quantum programs in LLVMIR

To ensure that you have access to all the stages, the ``keep_intermediate=True`` flag must be specified in the ``qjit`` decorator.
In the following example, we also compile ahead-of-time so that there is no requirements to pass actual parameters:

.. code-block:: python

    @qjit(keep_intermediate=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(x: float, y: float):
        theta = jnp.sin(x) + y
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0,1])
        return qml.state()

    print(circuit.jaxpr)

Out:

.. code-block:: python

    { lambda ; a:f64[] b:f64[]. let
        c:c128[4] = func[
        call_jaxpr={ lambda ; d:f64[] e:f64[]. let
            f:AbstractQreg() = qalloc 2
            g:f64[] = sin d
            h:f64[] = add g e
            i:AbstractQbit() = qextract f 0
            j:AbstractQbit() = qinst[op=RY qubits_len=1 runtime=lightning] i h
            k:AbstractQbit() = qextract f 1
            l:AbstractQbit() m:AbstractQbit() = qinst[
                op=CNOT
                qubits_len=2
                runtime=lightning
            ] j k
            _:AbstractObs(num_qubits=2,primitive=compbasis) = compbasis l m
            n:c128[4] = state l m
            = qdealloc f
            in (n,) }
        fn=<QNode: wires=2, device='lightning.qubit', interface='autograd', diff_method='best'>
        ] a b
    in (c,) }

The next stage is the JAXPR equivalent in MLIR, expressed using the MHLO dialect for classical
computation and the Quantum dialect for quantum computation. Note that the MHLO dialect is a
representation of HLO in MLIR, where HLO is the input IR to the accelerated linear algebra (XLA)
compiler used by TensorFlow.

.. code-block:: python

    print(circuit.mlir)

Lowering out of the MHLO dialect leaves us with the classical computation represented by generic
dialects such as ``arith``, ``math``, or ``linalg``. This allows us to later generate machine code
via standard LLVM-MLIR tooling.

.. code-block:: python

    print(get_compilation_stage(circuit, "HLOLoweringPass"))

The quantum compilation pipeline expands high-level quantum instructions like adjoint, and applies quantum differentiation methods and optimization techniques.

.. code-block:: python

    print(get_compilation_stage(circuit, "QuantumCompilationPass"))

An important step in getting to machine code from a high-level representation is allocating memory
for all the tensor/array objects in the program.

.. code-block:: python

    print(get_compilation_stage(circuit, "BufferizationPass"))

The LLVM dialect can be considered the "exit point" from MLIR when using LLVM for low-level compilation:

.. code-block:: python

    print(get_compilation_stage(circuit, "MLIRToLLVMDialect"))

And finally some LLVMIR that is inspired by QIR.

.. code-block:: python

    print(circuit.qir)


The LLVMIR code is compiled to an object file using the LLVM static compiler and linked to the
runtime libraries. The generated shared object is stored by the caching mechanism in Catalyst
for future calls.

Recompiling a Function
======================

Catalyst offers a way to extract IRs from pipeline stages and feed modified IRs back for recompilation.
To enable this feature, ``qjit`` decorated function must be compiled with the option ``keep_intermediate=True``.

The following example creates a square function decorated with ``@qjit(keep_intermediate=True)``.
The function must be compiled first so that the IR from each pipeline stage can be accessed.

>>> @qjit(keep_intermediate=True)
... def f(x):
...     return x**2
>>> f(2.0)
4.0

After compilation, we can use :func:`~.debug.get_compilation_stage`  in the ``debug`` module to get the IR
from the given compiler stage. In this example, we request for the IR after ``HLOLoweringPass``:

>>> from catalyst.debug import get_compilation_stage
>>> old_ir = get_compilation_stage(f, "HLOLoweringPass")

The output IR is:

.. code-block:: mlir

    module @f {
        func.func public @jit_f(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
            %0 = tensor.empty() : tensor<f64>
            %1 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%arg0, %arg0 : tensor<f64>, tensor<f64>) outs(%0 : tensor<f64>) {
            ^bb0(%in: f64, %in_0: f64, %out: f64):
                %2 = arith.mulf %in, %in_0 : f64
                linalg.yield %2 : f64
            } -> tensor<f64>
            return %1 : tensor<f64>
        }
        func.func @setup() {
            quantum.init
            return
        }
        func.func @teardown() {
            quantum.finalize
            return
        }
    }

Here we modify ``%2 = arith.mulf %in, %in_0 : f64`` to turn the square function into a cubic one.

>>> new_ir = old_ir.replace(
...     "%2 = arith.mulf %in, %in_0 : f64\n",
...     "%t = arith.mulf %in, %in_0 : f64\n    %2 = arith.mulf %t, %in_0 : f64\n"
... )

After that, we can use :func:`~.debug.replace_ir` to make the compiler use the modified
IR for recompilation.
The recompilation starts after the given checkpoint stage:

>>> from catalyst.debug import replace_ir
>>> replace_ir(f, "HLOLoweringPass", new_ir)
>>> f(2.0)
8.0

C Executable Generation
=======================

Catalyst provides :func:`~.debug.compile_executable` to generate a C executable with a given function and the
corresponding arguments.

.. code-block:: python

    import subprocess
    from catalyst import qjit
    from catalyst.debug import compile_executable, print_memref

    @qjit
    def f(x):
        y = x * x
        print_memref(y)
        return y

>>> binary = compile_executable(f, 5)
>>> print(binary)
/path/to/executable

.. code-block:: shell

    $ /path/to/executable
    MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
    25
