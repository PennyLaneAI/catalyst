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
    virtual auto AllocateQubits(__attribute__((unused)) size_t num_qubits)
        -> std::vector<QubitIdType>
    {
        return std::vector<QubitIdType>(num_qubits);
    }
    [[nodiscard]] virtual auto Zero() const -> Result { return NULL; }
    [[nodiscard]] virtual auto One() const -> Result { return NULL; }
    virtual auto Observable(__attribute__((unused)) ObsId id,
                            __attribute__((unused)) const std::vector<std::complex<double>> &matrix,
                            __attribute__((unused)) const std::vector<QubitIdType> &wires)
        -> ObsIdType
    {
        return 0;
    }
    virtual auto TensorObservable(__attribute__((unused)) const std::vector<ObsIdType> &obs)
        -> ObsIdType
    {
        return 0;
    }
    virtual auto HamiltonianObservable(__attribute__((unused)) const std::vector<double> &coeffs,
                                       __attribute__((unused)) const std::vector<ObsIdType> &obs)
        -> ObsIdType
    {
        return 0;
    }
    virtual auto Measure(__attribute__((unused)) QubitIdType wire) -> Result
    {
        bool *ret = (bool *)malloc(sizeof(bool));
        *ret = true;
        return ret;
    }

    virtual void ReleaseQubit(__attribute__((unused)) QubitIdType qubit) {}
    virtual void ReleaseAllQubits() {}
    [[nodiscard]] virtual auto GetNumQubits() const -> size_t { return 0; }
    virtual void SetDeviceShots(__attribute__((unused)) size_t shots) {}
    [[nodiscard]] virtual auto GetDeviceShots() const -> size_t { return 0; }
    virtual void StartTapeRecording() {}
    virtual void StopTapeRecording() {}
    virtual void PrintState() {}
    virtual void NamedOperation(__attribute__((unused)) const std::string &name,
                                __attribute__((unused)) const std::vector<double> &params,
                                __attribute__((unused)) const std::vector<QubitIdType> &wires,
                                __attribute__((unused)) bool inverse)
    {
    }

    virtual void MatrixOperation(__attribute__((unused))
                                 const std::vector<std::complex<double>> &matrix,
                                 __attribute__((unused)) const std::vector<QubitIdType> &wires,
                                 __attribute__((unused)) bool inverse)
    {
    }

    virtual auto Expval(__attribute__((unused)) ObsIdType obsKey) -> double { return 0.0; }
    virtual auto Var(__attribute__((unused)) ObsIdType obsKey) -> double { return 0.0; }
    virtual void State(__attribute__((unused)) DataView<std::complex<double>, 1> &state) {}
    virtual void Probs(__attribute__((unused)) DataView<double, 1> &probs) {}
    virtual void PartialProbs(__attribute__((unused)) DataView<double, 1> &probs,
                              __attribute__((unused)) const std::vector<QubitIdType> &wires)
    {
    }
    virtual void Sample(__attribute__((unused)) DataView<double, 2> &samples,
                        __attribute__((unused)) size_t shots)
    {
    }
    virtual void PartialSample(__attribute__((unused)) DataView<double, 2> &samples,
                               __attribute__((unused)) const std::vector<QubitIdType> &wires,
                               __attribute__((unused)) size_t shots)
    {
    }
    virtual void Counts(__attribute__((unused)) DataView<double, 1> &eigvals,
                        __attribute__((unused)) DataView<int64_t, 1> &counts,
                        __attribute__((unused)) size_t shots)
    {
    }

    virtual void PartialCounts(__attribute__((unused)) DataView<double, 1> &eigvals,
                               __attribute__((unused)) DataView<int64_t, 1> &counts,
                               __attribute__((unused)) const std::vector<QubitIdType> &wires,
                               __attribute__((unused)) size_t shots)
    {
    }

    virtual void Gradient(__attribute__((unused)) std::vector<DataView<double, 1>> &gradients,
                          __attribute__((unused)) const std::vector<size_t> &trainParams)
    {
    }
};

extern "C" Catalyst::Runtime::QuantumDevice *getCustomDevice() { return new DummyDevice(); }
