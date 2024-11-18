
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
                                        const std::vector<QubitIdType> &,
                                        bool,
                                        const std::vector<QubitIdType> &,
                                        const std::vector<bool> &
                                        ) override {}
            void MatrixOperation(const std::vector<std::complex<double>> &,
                                        const std::vector<QubitIdType> &,
                                        bool,
                                        const std::vector<QubitIdType> &,
                                        const std::vector<bool> &
                                        ) override{}
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
    CustomDeviceFactory(const char *kwargs) {
        return new CustomDevice(std::string(kwargs));
    }

For simplicity, you can use the ``GENERATE_DEVICE_FACTORY(IDENTIFIER, CONSTRUCTOR)`` macro to
define this function, where ``IDENTIFIER`` is the device identifier, and ``CONSTRUCTOR`` is the
C++ device constructor including the namespace. For this example, both the device identifier and
constructor are the same:

.. code-block:: c++

    GENERATE_DEVICE_FACTORY(CustomDevice, CustomDevice);

The entry point function acts as a factory method for the device class.
Note that a plugin library may also provide several factory methods in case it packages
multiple devices into the same library. However, it is important that the device identifier
be unique, as best as possible, to avoid clashes with other plugins.

Importantly, the ``<DeviceIdentifier>`` string in the entry point function needs to match
exactly what is supplied to the ``__catalyst__rt__device("rtd_name", "<DeviceIdentifier>")``
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

* Adding a ``get_c_interface`` method to your ``qml.devices.Device`` class.
* Adding a ``config_filepath`` class variable pointing to your configuration file. This file should be a `toml file <https://toml.io/en/>`_ with fields that describe what gates and features are supported by your device.
* Optionally, adding a ``device_kwargs`` dictionary for runtime parameters to pass from the PennyLane device to the ``QuantumDevice`` upon initialization.

If you already have a custom PennyLane device defined in Python and have added a shared object that corresponds to your implementation of the ``QuantumDevice`` class, then all you need to do is to add a ``get_c_interface`` method to your PennyLane device.
The ``get_c_interface`` method should be a static method that takes no parameters and returns the complete path to your shared library with the ``QuantumDevice`` implementation.

.. note::

    The first result of ``get_c_interface`` needs to match the ``<DeviceIdentifier>``
    as described in the first section.

The Pennylane device API allows you to build a QJIT compatible device in a simple way:

.. code-block:: python

    class CustomDevice(qml.devices.Device):
        """Custom Device"""

        config_filepath = pathlib.Path("absolute/path/to/configuration/file.toml")

        @staticmethod
        def get_c_interface():
            """ Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            return "CustomDevice", "absolute/path/to/librtd_custom.so"

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def execute(self, circuits, config):
            """Your normal definitions"""

    @qjit
    @qml.qnode(CustomDevice(wires=1))
    def f():
        return measure(0)

Below is an example configuration file with inline descriptions of how to fill out the fields. All
headers and fields are generally required, unless stated otherwise.

.. code-block:: toml

    schema = 3

    # The set of all gate types supported at the runtime execution interface of the
    # device, i.e., what is supported by the `execute` method. The gate definitions
    # should have the following format:
    #
    #   GATE = { properties = [ PROPS ], conditions = [ CONDS ] }
    #
    # where PROPS and CONS are zero or more comma separated quoted strings.
    #
    # PROPS: additional support provided for each gate.
    #        - "controllable": if a controlled version of this gate is supported.
    #        - "invertible": if the adjoint of this operation is supported.
    #        - "differentiable": if device gradient is supported for this gate.
    # CONDS: constraints on the support for each gate.
    #        - "analytic" or "finiteshots": if this operation is only supported in
    #          either analytic execution or with shots, respectively.
    #
    [operators.gates]

    PauliX = { properties = ["controllable", "invertible"] }
    PauliY = { properties = ["controllable", "invertible"] }
    PauliZ = { properties = ["controllable", "invertible"] }
    RY = { properties = ["controllable", "invertible", "differentiable"] }
    RZ = { properties = ["controllable", "invertible", "differentiable"] }
    CRY = { properties = ["invertible", "differentiable"] }
    CRZ = { properties = ["invertible", "differentiable"] }
    CNOT = { properties = ["invertible"] }

    # Observables supported by the device for measurements. The observables defined
    # in this section should have the following format:
    #
    #   OBSERVABLE = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each observable.
    #        - "analytic" or "finiteshots": if this observable is only supported in
    #          either analytic execution or with shots, respectively.
    #        - "terms-commute": if a composite operator is only supported under the
    #          condition that its terms commute.
    #
    [operators.observables]

    PauliX = { }
    PauliY = { }
    PauliZ = { }
    Hamiltonian = { conditions = [ "terms-commute" ] }
    Sum = { conditions = [ "terms-commute" ] }
    SProd = { }
    Prod = { }

    # Types of measurement processes supported on the device. The measurements in
    # this section should have the following format:
    #
    #   MEASUREMENT_PROCESS = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each measurement process.
    #        - "analytic" or "finiteshots": if this measurement is only supported
    #          in either analytic execution or with shots, respectively.
    #
    [measurement_processes]

    ExpectationMP = { }
    SampleMP = { }
    CountsMP = { conditions = ["finiteshots"] }
    StateMP = { conditions = ["analytic"] }

    # Additional support that the device may provide that informs the compilation
    # process. All accepted fields and their default values are listed below.
    [compilation]

    # Whether the device is compatible with qjit.
    qjit_compatible = false

    # Whether the device requires run time generation of the quantum circuit.
    runtime_code_generation = false

    # Whether the device supports allocating and releasing qubits during execution.
    dynamic_qubit_management = false

    # Whether simultaneous measurements on overlapping wires is supported.
    overlapping_observables = true

    # Whether simultaneous measurements of non-commuting observables is supported.
    # If false, a circuit with multiple non-commuting measurements will have to be
    # split into multiple executions for each subset of commuting measurements.
    non_commuting_observables = false

    # Whether the device supports initial state preparation.
    initial_state_prep = false

    # The methods of handling mid-circuit measurements that the device supports,
    # e.g., "one-shot", "tree-traversal", "device", etc. An empty list indicates
    # that the device does not support mid-circuit measurements.
    supported_mcm_methods = [ ]

This TOML file is used by both Catalyst frontend and PennyLane. Regular circuit execution is
performed by your implementation of ``Device.execute``, whereas for a QJIT-compiled workflow,
execution is performed by the ``QuantumDevice``. The TOML file should declare the capabilities
of the two execution interfaces. If one of the interfaces have additional support that the other
does not have, include them in a separate section:

.. code-block:: toml

    # Gates supported by the Python implementation of Device.execute but not by the QuantumDevice.
    [pennylane.operators.gates]

    MultiControlledX  = { }

    # Observables supported by the QuantumDevice but not by your implementation of Device.execute.
    [qjit.operators.observables]

    Sum = { }

Additionally, any runtime parameters to be passed to the ``QuantumDevice`` upon initialization
should be specified in a dictionary class property ``device_kwargs`` that links keyword arguments
of the ``QuantumDevice`` constructor to attributes of the ``qml.device.Device`` implementation.
For example:

.. code-block:: python

    class CustomDevice(qml.devices.Device):
        """Custom Device"""

        config_filepath = pathlib.Path("absolute/path/to/configuration/file.toml")

        def __init__(self, wires, do_something=False, special_param=""):
            ...
            self.device_kwargs = {
              'cpp_do_something' = do_something,
              'cpp_special_param' = special_param
            }

In the above example, a dictionary will be constructed at runtime and passed to the constructor of
the ``QuantumDevice`` implementation.
