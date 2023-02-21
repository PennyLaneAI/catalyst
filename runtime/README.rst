.. runtime-start-inclusion-marker-do-not-remove

Catalyst Runtime
################

The Catalyst Runtime is a C++ QIR runtime that enables the execution of Catalyst-compiled quantum programs, and is currently
backed by state-vector simulators `PennyLane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_
and `Pennylane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_.
A complete list of the quantum instruction set supported by the runtime can be found in
`RuntimeCAPI.h <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/include/RuntimeCAPI.h>`_.


The directory is structured as follows:

- `include <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/include>`_ : This contains the public header files including the runtime driver interface (for backend devices) and the runtime C-API (for QIR programs).
- `extensions <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/extensions>`_ : A collection of extensions for backend simulators to fit into the `QIR programming model <https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/4_Quantum_Runtime.md#qubits>`_. The `StateVectorDynamicCPU <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/extensions/StateVectorDynamicCPU.hpp>`_ class extends the state-vector class of `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ providing dynamic allocation and deallocation of qubits.
- `lib <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib>`_ : The core modules of the runtime are structured into ``lib/capi`` and ``lib/backend``. `lib/capi <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/capi>`_  contains the source of the bridge between QIR instructions in LLVM-IR and C++ backends. `lib/backend <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/backend>`_ contains implementations of the ``QuantumDevice`` API for backend simulators.
- `tests <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/tests>`_ : A collection of C++ tests for modules and methods in the runtime.

Requirements
============

To build the runtime from source, it is required to have an up to date version of a C/C++ compiler such as gcc, clang, or MSVC and the static library of
``stdlib`` from `qir-runner <https://github.com/qir-alliance/qir-runner/tree/main/stdlib>`_.

The ``make runtime`` command uses ``clang`` and ``clang++`` as the default C and C++ compilers. You can use ``c_compiler`` and ``cpp_compiler`` make
options to change the default behavior.

.. code-block:: console

    make runtime c_compiler=$(which gcc) cpp_compiler=$(which g++)

The runtime leverages the ``stdlib`` Rust package for the QIR standard runtime instructions.
To build this package from source, the `Rust <https://www.rust-lang.org/tools/install>`_ toolchain installed via ``rustup`` is also required.

Installation
============

By default, the runtime leverages `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ as the backend simulator.
You can use the CMake flag ``-DENABLE_LIGHTNING_KOKKOS`` to build the runtime with `Pennylane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_
in the serial mode or run:

.. code-block:: console

    make runtime kokkos=ON

Lightning-Kokkos provides support for other Kokkos backends including OpenMP, HIP and CUDA.
Please refer to `the installation guideline <https://github.com/PennyLaneAI/pennylane-lightning-kokkos#installation>`_ for the requirements.

The runtime uses the QIR standard library for `basic QIR instructions <https://github.com/qir-alliance/qir-runner/blob/main/stdlib/include/qir_stdlib.h>`_.
Before building ``stdlib``, the ``llvm-tools-preview`` Rustup component needs to be installed:

.. code-block:: console

  rustup component add llvm-tools-preview


To build the static library of ``stdlib``:

.. code-block:: console

    make qir

And use CMake flags ``-DQIR_STDLIB_LIB`` and ``-DQIR_STDLIB_INCLUDES`` to respectively locate ``libqir_stdlib.a`` and ``qir_stdlib.h``, or run:

.. code-block:: console

    make runtime qir_stdlib_dir=$(pwd)/qir-stdlib/target/release qir_stdlib_include_dir=$(pwd)/qir-stdlib/target/release/build/include

To check the runtime test suite:

.. code-block:: console

    make test

You can also build and test the runtime (and ``qir-stdlib``) from the top level directory via ``make runtime`` and ``make test-runtime``.

.. runtime-end-inclusion-marker-do-not-remove
