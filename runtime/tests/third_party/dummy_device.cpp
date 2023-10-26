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

extern "C" Catalyst::Runtime::QuantumDevice *getCustomDevice() { return new DummyDevice(); }
