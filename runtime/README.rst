.. runtime-start-inclusion-marker-do-not-remove

Catalyst Runtime
################

The Catalyst Runtime is a C++ QIR runtime that enables the execution of Catalyst-compiled
quantum programs, and is currently backed by state-vector simulators
`PennyLane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_
and `Pennylane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_.

The runtime employs the `QuantumDevice <https://docs.pennylane.ai/projects/catalyst/en/stable/api/structCatalyst_1_1Runtime_1_1QuantumDevice.html#exhale-struct-structcatalyst-1-1runtime-1-1quantumdevice>`_
public interface to support an extensible list of backend devices. This interface comprises two collections of abstract methods:

- The Qubit management, device shot noise, and quantum tape recording methods are utilized for the implementation of Quantum Runtime (QR) instructions.

- The quantum operations, observables, measurements, and gradient methods are used to implement Quantum Instruction Set (QIS) instructions.

A complete list of instructions supported by the runtime can be found in
`RuntimeCAPI.h <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/include/RuntimeCAPI.h>`_.

Contents
========

The directory is structured as follows:

- `include <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/include>`_:
    This contains the public header files of the runtime including the ``QuantumDevice`` API
    for backend quantum devices and the runtime CAPI.

- `extensions <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/extensions>`_:
    A collection of extensions for backend simulators to fit into the
    `QIR programming model <https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/4_Quantum_Runtime.md#qubits>`_.
    The `StateVectorDynamicCPU <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/extensions/StateVectorDynamicCPU.hpp>`_
    class extends the state-vector class of `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ providing
    dynamic allocation and deallocation of qubits.

- `lib <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib>`_:
    The core modules of the runtime are structured into ``lib/capi`` and ``lib/backend``.
    `lib/capi <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/capi>`_  implements the bridge between
    QIR instructions in LLVM-IR and C++ device backends. `lib/backend <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/backend>`_
    contains implementations of the ``QuantumDevice`` API for backend simulators.

- `tests <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/tests>`_:
    A collection of C++ tests for modules and methods in the runtime.

Backend Devices
===============

New device backends for the runtime can be realized by implementing the quantum device interface.
The following table shows the available devices along with supported features:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - **Features**
     - **PennyLane-Lightning**
     - **PennyLane-Lightning-Kokkos**
   * - Qubit Management
     - Dynamic allocation/deallocation
     - Static allocation/deallocation
   * - Gate Operations
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/src/gates/GateOperation.hpp>`_
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/src/gates/GateOperation.hpp>`_
   * - Quantum Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
   * - Expectation Value
     - All observables; Finite-shots not supported
     - All observables; Finite-shots not supported
   * - Variance
     - Only for ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, and ``Hadamard``; Finite-shots not supported
     - Not supported
   * - Probability
     - Only for the computational basis on the supplied qubits; Finite-shots not supported
     - Only for the computational basis on the supplied qubits; Finite-shots not supported
   * - Sampling
     - Only for the computational basis on the supplied qubits
     - Only for the computational basis on the supplied qubits
   * - Mid-Circuit Measurement
     - Only for the computational basis on the supplied qubit
     - Only for the computational basis on the supplied qubit
   * - Gradient
     - The Adjoint-Jacobian method for expectation values on all observables
     - The Adjoint-Jacobian method for expectation values on all observables

Requirements
============

To build the runtime from source, it is required to have an up to date version of a C/C++ compiler such as gcc or clang
and the static library of ``stdlib`` from `qir-runner <https://github.com/qir-alliance/qir-runner/tree/main/stdlib>`_.

The runtime leverages the ``stdlib`` Rust package for the QIR standard runtime instructions. To build this package from source,
the `Rust <https://www.rust-lang.org/tools/install>`_ toolchain installed via ``rustup`` is also required.

Installation
============

By default, the runtime leverages `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ as the backend simulator.
You can use the CMake flag ``-DENABLE_LIGHTNING_KOKKOS`` to build the runtime with `Pennylane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_
in the serial mode or run:

.. code-block:: console

    ENABLE_KOKKOS=ON make runtime

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

    QIR_STDLIB_DIR=$(pwd)/qir-stdlib/target/release QIR_STDLIB_INCLUDES_DIR=$(pwd)/qir-stdlib/target/release/build/include make runtime

To check the runtime test suite:

.. code-block:: console

    make test

You can also build and test the runtime (and ``qir-stdlib``) from the top level directory via ``make runtime`` and ``make test-runtime``.

.. runtime-end-inclusion-marker-do-not-remove
