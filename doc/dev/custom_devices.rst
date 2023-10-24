
Custom Devices
##############

Differences between PennyLane and Catalyst
==========================================

PennyLane and Catalyst treat devices a bit differently.
In PennyLane, one is able to `define devices <https://docs.pennylane.ai/en/stable/development/plugins.html>`_ in Python.
Catalyst cannot interface with Python devices yet.
Instead, Catalyst can only interact with devices that implement the `QuantumDevice <../api/file_runtime_include_QuantumDevice.hpp.html>`_ class.

Here is an example of a custom ``QuantumDevice`` which every single quantum operation is implemented as a no-operation.
Additionally, all measurements will always return ``true``.

.. code-block:: c++

        #include <QuantumDevice.hpp>

        struct DummyDevice : public Catalyst::Runtime::QuantumDevice {
            DummyDevice() = default;          // LCOV_EXCL_LINE
            virtual ~DummyDevice() = default; // LCOV_EXCL_LINE

            DummyDevice &operator=(const QuantumDevice &) = delete;
            DummyDevice(const DummyDevice &) = delete;
            DummyDevice(DummyDevice &&) = delete;
            DummyDevice &operator=(QuantumDevice &&) = delete;

            virtual std::string getName(void) { return "DummyDevice"; }

            auto AllocateQubit() -> QubitIdType { return 0; }
            virtual auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
            {
                return std::vector<QubitIdType>(num_qubits);
            }
            [[nodiscard]] virtual auto Zero() const -> Result { return NULL; }
            [[nodiscard]] virtual auto One() const -> Result { return NULL; }
            virtual auto Observable(ObsId, const std::vector<std::complex<double>> &,
                                    const std::vector<QubitIdType> &) -> ObsIdType
            {
                return 0;
            }
            virtual auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType { return 0; }
            virtual auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
                -> ObsIdType
            {
                return 0;
            }
            virtual auto Measure(QubitIdType) -> Result
            {
                bool *ret = (bool *)malloc(sizeof(bool));
                *ret = true;
                return ret;
            }

            virtual void ReleaseQubit(QubitIdType) {}
            virtual void ReleaseAllQubits() {}
            [[nodiscard]] virtual auto GetNumQubits() const -> size_t { return 0; }
            virtual void SetDeviceShots(size_t shots) {}
            [[nodiscard]] virtual auto GetDeviceShots() const -> size_t { return 0; }
            virtual void StartTapeRecording() {}
            virtual void StopTapeRecording() {}
            virtual void PrintState() {}
            virtual void NamedOperation(const std::string &, const std::vector<double> &,
                                        const std::vector<QubitIdType> &, bool)
            {
            }

            virtual void MatrixOperation(const std::vector<std::complex<double>> &,
                                         const std::vector<QubitIdType> &, bool)
            {
            }

            virtual auto Expval(ObsIdType) -> double { return 0.0; }
            virtual auto Var(ObsIdType) -> double { return 0.0; }
            virtual void State(DataView<std::complex<double>, 1> &) {}
            virtual void Probs(DataView<double, 1> &) {}
            virtual void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) {}
            virtual void Sample(DataView<double, 2> &, size_t) {}
            virtual void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &, size_t) {}
            virtual void Counts(DataView<double, 1> &, DataView<int64_t, 1> &, size_t) {}

            virtual void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                                       const std::vector<QubitIdType> &, size_t)
            {
            }

            virtual void Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &) {}
        };

In addition to implementing the ``QuantumDevice`` class, one must implement the following method:

.. code-block:: c++

    extern "C" Catalyst::Runtime::QuantumDevice*
    getCustomDevice() { return new CustomDevice(); }

where ``CustomDevice()`` is a constructor for your custom device.
``CustomDevice``'s destructor will be called by the runtime.

.. note::

    This interface might change quickly in the near future.
    Please check back regularly for updates and to ensure your device is compatible with a specific version of Catalyst.

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

If you already have a custom PennyLane device defined in Python and have added a shared object that corresponds to your implementation of the ``QuantumDevice`` class, then all you need to do is to add a ``get_c_interface`` method to your PennyLane device.
The ``get_c_interface`` method should be a static method that takes no parameters and returns the complete path to your shared library with the ``QuantumDevice`` implementation.
After doing so, Catalyst should be able to interface with your custom device with no problem.

.. code-block:: python

    class DummyDevice(qml.QubitDevice):
        """Dummy Device"""

        name = "Dummy Device"
        short_name = "dummy.device"
        pennylane_requires = "0.32.0"
        version = "0.0.1"
        author = "Erick Ochoa"

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Your normal definitions"""

        @staticmethod
        def get_c_interface():
            """Location to shared object with C/C++ implementation"""
            return "/libdummy_device.so"

    @qjit
    @qml.qnode(DummyDevice(wires=1))
    def f():
        return measure(0)

