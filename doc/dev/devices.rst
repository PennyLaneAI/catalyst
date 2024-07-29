Supported devices
=================

Not all PennyLane devices currently work with Catalyst, and for those that do, their supported
feature set may not necessarily match supported features when used without :func:`~.qjit`.

Supported backend devices include:

.. list-table::
  :widths: 20 80

  * - ``lightning.qubit``

    - A fast state-vector qubit simulator written with a C++ backend. See the
      `Lightning documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/lightning_qubit/lightning_qubit.toml>`__.

  * - ``lightning.kokkos``

    - A fast state-vector qubit simulator utilizing the Kokkos library for CPU and GPU accelerated
      circuit simulation. See the
      `Lightning documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/lightning_kokkos/lightning_kokkos.toml>`__.

  * - ``braket.aws.qubit``

    - Interact with quantum computing hardware devices and simulators through Amazon Braket. To use
      this device with Catalyst, make sure to install the
      `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.
      See the
      `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/runtime/lib/backend/openqasm/braket_aws_qubit.toml>`__
      for supported operations.

  * - ``braket.local.qubit``

    - Run gate-based circuits on the Braket SDK's local simulator. To use
      this device with Catalyst, make sure to install the
      `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.
      See the
      `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/runtime/lib/backend/openqasm/braket_local_qubit.toml>`__
      for more details.

  * - ``qrack.simulator``

    - `Qrack <https://github.com/unitaryfund/qrack>`__ is a GPU-accelerated quantum computer
      simulator with many novel optimizations including hybrid stabilizer simulation. To use this
      device with Catalyst, make sure to install the
      `PennyLane-Qrack plugin <https://pennylane-qrack.readthedocs.io/en/latest/>`__, and check out
      the `QJIT compilation with Qrack and Catalyst tutorial <https://pennylane.ai/qml/demos/qrack/>`__.

      See the `Catalyst configuration file <https://github.com/unitaryfund/pennylane-qrack/blob/master/pennylane_qrack/QrackDeviceConfig.toml>`__
      for more details.

  * - ``oqc.cloud``

    - Execute on `Oxford Quantum Circuits (OQC) <https://www.oqc.tech/>`__ superconducting hardware,
      via `OQC Cloud <https://docs.oqc.app>`__. To use OQC Cloud with Catalyst, simply `install the
      client <https://docs.oqc.app/installation.html>`__, ensure your credentials are set as
      environment variables, and use the ``backend`` argument to specify the OQC backend to use:

      .. code-block:: python

          import os
          os.environ["OQC_EMAIL"] = "your_email"
          os.environ["OQC_PASSWORD"] = "your_password"
          os.environ["OQC_URL"] = "oqc_url"

          dev = qml.device("oqc.cloud", backend="lucy", shots=2012, wires=2)

      See the `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/third_party/oqc/src/oqc.toml>`__
      for more details.



Device features
---------------

The ``lightning,qubit``, ``lightning.kokkos``, ``braket.aws.qubit``, ``braket.local.qubit``,
and ``oqc.cloud`` devices are currently provided by the Catalyst package. For these
built-in devices, the following table shows supported features devices (for external
qjit-compatible devices, please consult the corresponding device documentation):

.. list-table::
  :widths: 16 21 21 21 21
  :header-rows: 0

  * - **Features**
    - ``lightning.qubit``
    - ``lightning.kokkos``
    - ``braket.aws.qubit``
    - ``oqc.cloud``
  * - Qubit Management
    - Dynamic allocation/deallocation
    - Static allocation/deallocation
    - Static allocation/deallocation
    - Static allocation/deallocation
  * - Gate Operations
    - `Lightning Qubit operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/lightning_qubit/lightning_qubit.toml>`__
    - `Lightning Kokkos operations <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/lightning_kokkos/lightning_kokkos.toml>`__
    - `Braket operations <https://github.com/PennyLaneAI/catalyst/blob/main/runtime/lib/backend/openqasm/braket_aws_qubit.toml>`__
    - `OQC operations <https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/third_party/oqc/src/oqc.toml>`__
  * - Quantum Observables
    - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
    - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, ``Hamiltonian``, and Tensor Product of Observables
    - ``Identity``, ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard``, ``Hermitian``, and Tensor Product of Observables
    - ``PauliX``, ``PauliY``, ``PauliZ``, and ``Hadamard``.
  * - Expectation Value
    - All observables; Finite-shots supported except for ``Hermitian``
    - All observables; Finite-shots supported except for ``Hermitian``
    - All observables; Finite-shots supported
    - All observables; Finite-shots supported
  * - Variance
    - All observables; Finite-shots supported except for ``Hermitian``
    - All observables; Finite-shots supported except for ``Hermitian``
    - All observables; Finite-shots supported
    - Not supported
  * - Probability
    - Only for the computational basis on the supplied qubits; Finite-shots supported except for ``Hermitian``
    - Only for the computational basis on the supplied qubits; Finite-shots supported except for ``Hermitian``
    - Not currently supported
    - Only for the computational basis on the supplied qubits
  * - Sampling
    - Only for the computational basis on the supplied qubits
    - Only for the computational basis on the supplied qubits
    - The computational basis on all active qubits; Finite-shots supported
    - The computational basis on all active qubits; Finite-shots supported
  * - Mid-Circuit Measurement
    - Only for the computational basis on the supplied qubit
    - Only for the computational basis on the supplied qubit
    - Not supported
    - Not supported
  * - Gradient
    - The Adjoint-Jacobian method for expectation values on all observables
    - The Adjoint-Jacobian method for expectation values on all observables
    - Not supported
    - Not supported
