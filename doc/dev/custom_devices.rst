
Custom Devices
##############

Differences between PennyLane and Catalyst
==========================================

PennyLane and Catalyst treat devices a bit differently.
In PennyLane, one is able to `define devices <https://docs.pennylane.ai/en/stable/development/plugins.html>`_ in Python.
Catalyst cannot interface with Python devices yet.
Instead, Catalyst can only interact with devices that implement the `QuantumDevice <../api/file_runtime_include_QuantumDevice.hpp.html>`_ class.

Here is an example of a custom ``QuantumDevice`` in which every single quantum operation is implemented as a no-operation.
Additionally, all measurements will always return ``true``.

.. code-block:: c++

        #include <QuantumDevice.hpp>

        struct CustomDevice final : public Catalyst::Runtime::QuantumDevice {
            CustomDevice([[maybe_unused]] const std::string &kwargs = "{}") {}
            ~CustomDevice() = default;

            CustomDevice &operator=(const QuantumDevice &) = delete;
            CustomDevice(const CustomDevice &) = delete;
            CustomDevice(CustomDevice &&) = delete;
            CustomDevice &operator=(QuantumDevice &&) = delete;

            auto AllocateQubit() -> QubitIdType override { return 0; }
            auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override {
                return std::vector<QubitIdType>(num_qubits);
            }
            [[nodiscard]] auto Zero() const -> Result override { return NULL; }
            [[nodiscard]] auto One() const -> Result override { return NULL; }
            auto Observable(ObsId, const std::vector<std::complex<double>> &,
                                    const std::vector<QubitIdType> &) -> ObsIdType override {
                return 0;
            }
            auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType override { return 0; }
            auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
                -> ObsIdType override {
                return 0;
            }
            auto Measure(QubitIdType) -> Result override {
                bool *ret = (bool *)malloc(sizeof(bool));
                *ret = true;
                return ret;
            }

            void ReleaseQubit(QubitIdType) override {}
            void ReleaseAllQubits() override {}
            [[nodiscard]] auto GetNumQubits() const -> size_t override { return 0; }
            void SetDeviceShots(size_t shots) override {}
            [[nodiscard]] auto GetDeviceShots() const -> size_t override { return 0; }
            void StartTapeRecording() override {}
            void StopTapeRecording() override {}
            void PrintState() override {}
            void NamedOperation(const std::string &, const std::vector<double> &,
                                        const std::vector<QubitIdType> &, bool) override {}
            void MatrixOperation(const std::vector<std::complex<double>> &,
                                        const std::vector<QubitIdType> &, bool) override{}

            auto Expval(ObsIdType) -> double override { return 0.0; }
            auto Var(ObsIdType) -> double override { return 0.0; }
            void State(DataView<std::complex<double>, 1> &) override {}
            void Probs(DataView<double, 1> &) override {}
            void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) override {}
            void Sample(DataView<double, 2> &, size_t) override {}
            void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &, size_t) override {}
            void Counts(DataView<double, 1> &, DataView<int64_t, 1> &, size_t) override {}

            void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                                    const std::vector<QubitIdType> &, size_t) override {}

            void Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &) override {}
        };

In addition to implementing the ``QuantumDevice`` class, one must implement an entry point for the
device library with the name ``<DeviceIdentifier>Factory``, where ``DeviceIdentifier`` is used to
uniquely identify the entry point symbol. As an example, we use the identifier ``CustomDevice``:

.. code-block:: c++

    extern "C" Catalyst::Runtime::QuantumDevice*
    CustomDeviceFactory(const std::string &kwargs) {
        return new CustomDevice(kwargs);
    }

The entry point function acts as a factory method for the device class.
Note that a plugin library may also provide several factory methods in case it packages
multiple devices into the same library. However, it is important that the device identifier
be unique, as best as possible, to avoid clashes with other plugins.

Importantly, the ``<DeviceIdentifier>`` string in the entry point function needs to match
exactly what is supplied to the ``__quantum__rt__device("rtd_name", "<DeviceIdentifier>")``
runtime instruction in compiled user programs, or what is returned from the ``get_c_interface``
function when integrating the device into a PennyLane plugin. Please see the "Integration with
Python devices" section further down for details.

``CustomDevice(kwargs)`` serves as a constructor for your custom device, with ``kwargs``
as a string of device specifications and options, represented in Python dictionary format.
An example could be the default number of device shots, encoded as the following string:
``"{'shots': 1000}"``.

Note that these parameters are automatically initialized in the frontend if the library is
provided as a PennyLane plugin device (see :func:`qml.device() <pennylane.device>`).

The destructor of ``CustomDevice`` will be automatically called by the runtime.

.. warning::

    This interface might change quickly in the near future.
    Please check back regularly for updates and to ensure your device is compatible with
    a specific version of Catalyst.

How to compile custom devices
=============================

One can follow the ``catalyst/runtime/tests/third_party/CMakeLists.txt`` `as an example. <https://github.com/PennyLaneAI/catalyst/blob/26b412b298f22565fea529d2019554e7ad9b9624/runtime/tests/third_party/CMakeLists.txt>`_

.. code-block:: cmake

        cmake_minimum_required(VERSION 3.20)

        project(third_party_device)

        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        add_library(dummy_device SHARED dummy_device.cpp)
        target_include_directories(dummy_device PUBLIC ${runtime_includes})
        set_property(TARGET dummy_device PROPERTY POSITION_INDEPENDENT_CODE ON)

Integration with Python devices
===============================

There are two things that are needed in order to integrate with PennyLane devices:

* Adding a ``get_c_interface`` method to your ``qml.QubitDevice`` class.
* Adding a ``config`` class variable pointing to your configuration file.

If you already have a custom PennyLane device defined in Python and have added a shared object that corresponds to your implementation of the ``QuantumDevice`` class, then all you need to do is to add a ``get_c_interface`` method to your PennyLane device.
The ``get_c_interface`` method should be a static method that takes no parameters and returns the complete path to your shared library with the ``QuantumDevice`` implementation.

.. note::

    The first result of ``get_c_interface`` needs to match the ``<DeviceIdentifier>``
    as described in the first section.

.. code-block:: python

    class CustomDevice(qml.QubitDevice):
        """Dummy Device"""

        name = "Dummy Device"
        short_name = "dummy.device"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"
        author = "Erick Ochoa"
        config = pathlib.Path("path/to/configuration/file.toml")

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Your normal definitions"""

        @staticmethod
        def get_c_interface():
            """ Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            return "CustomDevice", "absolute/path/to/libdummy_device.so"

    @qjit
    @qml.qnode(CustomDevice(wires=1))
    def f():
        return measure(0)


Below is an example configuration file

.. code-block:: toml

        schema = 1

        [device]
        name = "dummy.device"
        version = "v0.32.0"
        precision = ['float32', 'float64']

        [operations]
        observables = [
                "NamedObs",
                "HermitianObs",
                "TensorObs",
                "HamiltonianObs",
        ]

        [[operations.gates]]
        full = [
                "Identity",
                "PauliX",
                "PauliY",
                "PauliZ",
                "Hadamard",
                "S",
                "T",
                "PhaseShift",
                "RX",
                "RY",
                "RZ",
                "Rot",
                "CNOT",
                "CY",
                "CZ",
                "SWAP",
                "IsingXX",
                "IsingXY",
                "IsingYY",
                "IsingZZ",
                "ControlledPhaseShift",
                "CRX",
                "CRY",
                "CRZ",
                "CRot",
                "Toffoli",
                "CSWAP",
                "MultiRZ",
        ]

        # Gates which should be decomposed to qml.QubitUnitary.
        matrix = [
                "OrbitalRotation",
                "MultiControlledX",
                "DoubleExcitation",
                "DoubleExcitationMinus",
                "DoubleExcitationPlus",
                "SingleExcitation",
                "SingleExcitationMinus",
                "SingleExcitationPlus",
        ]

        [measurements]
        exactshots = [
                "Expval",
                "Var",
        ]
        finiteshots = [
                "Probs",
                "Sample",
                "Measure",
        ]

        [compilation]
        qjit_compatible = true
        control_flow = true
        dynamic_qubit_management = false

