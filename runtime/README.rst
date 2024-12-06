.. runtime-start-inclusion-marker-do-not-remove

Catalyst Quantum Runtime
########################

The Catalyst Runtime is a C++ QIR runtime that enables the execution of Catalyst-compiled
quantum programs, and is currently backed by `PennyLane-Lightning <https://docs.pennylane.ai/projects/lightning/en/stable>`_
state-vector simulators, and `Amazon Braket <https://amazon-braket-pennylane-plugin-python.readthedocs.io>`__
devices. Additional hardware support, including QPUs, to come.

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

- `lib <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib>`_:
    The core modules of the runtime are structured into ``lib/capi`` and ``lib/backend``.
    `lib/capi <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/capi>`_  implements the semantics for
    QIR instructions lowered to our custom runtime. `lib/backend <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/backend>`_
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
     - **PennyLane-Lightning-Qubit**
     - **PennyLane-Lightning-Kokkos** and **PennyLane-Lightning-GPU**
     - **Amazon-Braket-OpenQasm**
   * - Qubit Management
     - Dynamic allocation/deallocation
     - Static allocation/deallocation
     - Static allocation/deallocation
   * - Gate Operations
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/core/src/gates/GateOperation.hpp>`_
     - `Lightning operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/core/src/gates/GateOperation.hpp>`_ without controlled gates support
     - `Braket operations <https://github.com/PennyLaneAI/catalyst/blob/e812afbadbd777209862d5c76f394e3f0c43ffb6/runtime/lib/backend/openqasm/OpenQasmBuilder.hpp#L49>`_
   * - Quantum Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
     - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, and Tensor Product of Observables
   * - Expectation Value
     - All observables; Finite-shots supported
     - All observables; Finite-shots supported
     - All observables; Finite-shots supported
   * - Variance
     - All observables; Finite-shots supported
     - All observables; Finite-shots supported
     - All observables; Finite-shots supported
   * - Probability
     - Only for the computational basis on the supplied qubits; Finite-shots supported
     - Only for the computational basis on the supplied qubits; Finite-shots supported
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
with support for the C++20 standard library.

Installation
============

By default, the runtime builds all supported backend devices.
You can build the runtime with custom devices from the list of Backend Devices.

You can use ``ENABLE_OPENQASM=OFF`` to disable building the runtime with `Amazon-Braket-OpenQasm <https://aws.amazon.com/braket/>`_:

.. code-block:: console

    make runtime ENABLE_OPENQASM=OFF

This device currently offers generators for the `OpenQasm3 <https://openqasm.com/versions/3.0/index.html>`_ specification and
`Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-supported-features.html>`__ assembly extension.
Moreover, the generated assembly can be executed on Amazon Braket devices leveraging `amazon-braket-sdk-python <https://github.com/aws/amazon-braket-sdk-python>`_.

To check the runtime test suite from the root directory:

.. code-block:: console

    make test-runtime

.. runtime-end-inclusion-marker-do-not-remove
