
Debugging Tips
##############

Compilation Steps
=================

The compilation process of a QJITed quantum function moves through various stages of the compilation pipeline including:

- **Quantum Tape**: The quantum record of variational quantum programs in a single ``qml.QNode``
- **JAXPR**: The graph data structure maintained by `JAX <https://github.com/google/jax>`_ of the classical part for transformations
- **MLIR**: A novel compiler framework and intermediate representation
- **HLO (XLA) + Quantum Dialect**: The first stage inside MLIR after leaving JAXPR
- **Builtin + Quantum Dialects**: HLO has been converted to a variety of classical dialects in MLIR
- **Bufferized MLIR**: All tensors have been converted to memory buffer allocations at this step
- **LLVM Dialect**: Lower the code to LLVM Dialect to target LLVMIR by providing a one-to-one mapping in MLIR
- **QIR**: A specification for quantum programs in LLVMIR

To ensure that you have access to all the stages, the ``keep_intermediate=True`` flag must be specified in ``qjit``.
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

    circuit.print_stage("nohlo")

An important step in getting to machine code from a high-level representation is allocating memory
for all the tensor/array objects in the program.

.. code-block:: python

    circuit.print_stage("buff")

The LLVM dialect can be considered the "exit point" from MLIR when using LLVM for low-level compilation:

.. code-block:: python

    circuit.print_stage("llvm")

And finally some real LLVMIR adhering to the QIR specification:

.. code-block:: python

    circuit.print_stage("ll")

The LLVMIR code is compiled to an object file using the LLVM static compiler and linked to the
runtime libraries. The generated shared object is stored by the caching mechanism in Catalyst
for future calls.

