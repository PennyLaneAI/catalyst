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
   :widths: 25 25 25 25
   :header-rows: 0

   * - **Features**
     - **PennyLane-Lightning**
     - **PennyLane-Lightning-Kokkos**
     - **Amazon-Braket-OpenQasm**
   * - Qubit Management
     - Dynamic allocation/deallocation
     - Static allocation/deallocation
     - Static allocation/deallocation
   * - Gate Operations
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/src/gates/GateOperation.hpp>`_
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/src/gates/GateOperation.hpp>`_
     - `Braket operations <https://github.com/PennyLaneAI/catalyst/blob/e812afbadbd777209862d5c76f394e3f0c43ffb6/runtime/lib/backend/openqasm/OpenQasmBuilder.hpp#L49>`_
   * - Quantum Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, and Tensor Product of Observables
   * - Expectation Value
     - All observables; Finite-shots not supported
     - All observables; Finite-shots not supported
     - All observables; Finite-shots supported
   * - Variance
     - All observables; Finite-shots not supported
     - All observables; Finite-shots not supported
     - All observables; Finite-shots supported
   * - Probability
     - Only for the computational basis on the supplied qubits; Finite-shots not supported
     - Only for the computational basis on the supplied qubits; Finite-shots not supported
     - The computational basis on all active qubits; Finite-shots supported
   * - Sampling
     - Only for the computational basis on the supplied qubits
     - Only for the computational basis on the supplied qubits
     - The computational basis on all active qubits; Finite-shots supported
   * - Mid-Circuit Measurement
     - Only for the computational basis on the supplied qubit
     - Only for the computational basis on the supplied qubit
     - Not supported
   * - Gradient
     - The Adjoint-Jacobian method for expectation values on all observables
     - The Adjoint-Jacobian method for expectation values on all observables
     - Not supported

Requirements
============

To build the runtime from source, it is required to have an up to date version of a C/C++ compiler such as gcc or clang
with support for the C++20 standard library and the static library of ``stdlib`` from `qir-runner <https://github.com/qir-alliance/qir-runner/tree/main/stdlib>`_.

The runtime leverages the ``stdlib`` Rust package for the QIR standard runtime instructions. To build this package from source,
the `Rust <https://www.rust-lang.org/tools/install>`_ toolchain installed via ``rustup`` is also required.

Installation
============

By default, the runtime leverages `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ as the backend simulator.
You can build the runtime with multiple devices from the list of Backend Devices.
You can use ``ENABLE_LIGHTNING_KOKKOS=ON`` to build the runtime with `Pennylane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_:

.. code-block:: console

    ENABLE_LIGHTNING_KOKKOS=ON make runtime

Lightning-Kokkos provides support for other Kokkos backends including OpenMP, HIP and CUDA.
Please refer to `the installation guideline <https://github.com/PennyLaneAI/pennylane-lightning-kokkos#installation>`_ for the requirements.
You can further use the ``CMAKE_ARGS`` flag to issue any additional compiler arguments or override the preset ones in the make commands.
To build the runtime with Lightning-Kokkos and the ``Kokkos::OpenMP`` backend execution space:

.. code-block:: console

    ENABLE_LIGHTNING_KOKKOS=ON CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" make runtime

You can also use ``ENABLE_OPENQASM=ON`` to build the runtime with `Amazon-Braket-OpenQasm <https://aws.amazon.com/braket/>`_:

.. code-block:: console

    ENABLE_OPENQASM=ON make runtime

This device currently offers generators for the `OpenQasm3 <https://openqasm.com/versions/3.0/index.html>`_ specification and
`Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-supported-features.html>`_ assembly extension.
Moreover, the generated assembly can be executed on Amazon Braket devices leveraging `amazon-braket-sdk-python <https://github.com/aws/amazon-braket-sdk-python>`_.

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
