#include <QuantumDevice.hpp>

struct DummyDevice final : public Catalyst::Runtime::QuantumDevice {
    DummyDevice([[maybe_unused]] const std::string &kwargs) {}
    ~DummyDevice() = default; // LCOV_EXCL_LINE

    DummyDevice &operator=(const QuantumDevice &) = delete;
    DummyDevice(const DummyDevice &) = delete;
    DummyDevice(DummyDevice &&) = delete;
    DummyDevice &operator=(QuantumDevice &&) = delete;

    auto AllocateQubit() -> QubitIdType override { return 0; }
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override
    {
        return std::vector<QubitIdType>(num_qubits);
    }
    [[nodiscard]] auto Zero() const -> Result override { return NULL; }
    [[nodiscard]] auto One() const -> Result override { return NULL; }
    auto Observable(ObsId, const std::vector<std::complex<double>> &,
                    const std::vector<QubitIdType> &) -> ObsIdType override
    {
        return 0;
    }
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType override { return 0; }
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType override
    {
        return 0;
    }
    auto Measure(QubitIdType, std::optional<int32_t>) -> Result override
    {
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
                        const std::vector<QubitIdType> &, bool) override
    {
    }

    void MatrixOperation(const std::vector<std::complex<double>> &,
                         const std::vector<QubitIdType> &, bool) override
    {
    }

    auto Expval(ObsIdType) -> double override { return 0.0; }
    auto Var(ObsIdType) -> double override { return 0.0; }
    void State(DataView<std::complex<double>, 1> &) override {}
    void Probs(DataView<double, 1> &) override {}
    void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) override {}
    void Sample(DataView<double, 2> &, size_t) override {}
    void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &, size_t) override {}
    void Counts(DataView<double, 1> &, DataView<int64_t, 1> &, size_t) override {}

    void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                       const std::vector<QubitIdType> &, size_t) override
    {
    }

    void Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &) override {}
};

GENERATE_DEVICE_FACTORY(DummyDevice, DummyDevice);
