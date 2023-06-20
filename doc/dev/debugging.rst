
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

Sometimes, it might be useful to run the JIT compiled function without ``python``, for example in case of debugging.
If the program succeeds without ``python``, then it is likely that the error is found in the interface between ``python`` and the JIT compiled function, or within Catalyst's ``python`` internals.
As a quick sanity check you might want to run the JIT compiled function without Python.
Below is an example of how to obtain a C program that can be linked against the generated function:

.. code-block:: python

    @qjit
    def identity(x):
        return x

    print(identity.get_cmain(1.0))

Using the ``QJIT.get_cmain`` function, the following string is returned to the user:

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

The user can now compile and link this program and run it without ``python``.

Pass Pipelines
==============

A lot of the compilation steps are broken into pass pipelines.
``PassPipeline`` is a class that specifies which binary and which flags are used for compilation.
Users can implement their own ``PassPipeline`` by inheriting from this class and implementing the relevant methods/attributes.
Catalyst's compilation strategy can then adjusted by inserting new passes in then default pipeline or forming a completely new pipeline.

We won't get into too much detail here, but sometimes it is useful to look at the output of a specific ``PassPipeline``.
To do so, simply use the ``get_output_of`` method available in ``QJIT``.
For example, if one wishes to inspect the output of the ``BufferizationPass``, simply run the following command.

.. code-block:: python

    circuit.get_output_of("BufferizationPass")


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

    circuit.get_output_of("MHLOPass")

An important step in getting to machine code from a high-level representation is allocating memory
for all the tensor/array objects in the program.

.. code-block:: python

    circuit.get_output_of("BufferizationPass")

The LLVM dialect can be considered the "exit point" from MLIR when using LLVM for low-level compilation:

.. code-block:: python

    circuit.get_output_of("MLIRToLLVMDialect")

And finally some real LLVMIR adhering to the QIR specification:

.. code-block:: python

    print(circuit.qir)


The LLVMIR code is compiled to an object file using the LLVM static compiler and linked to the
runtime libraries. The generated shared object is stored by the caching mechanism in Catalyst
for future calls.

