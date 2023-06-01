// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#define __device_openqasm

#include <algorithm>
#include <bitset>
#include <memory>
#include <string>
#include <vector>

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#include "CacheManager.hpp"
#include "QubitManager.hpp"

#include "OpenQasmBuilder.hpp"
#include "OpenQasmRunner.hpp"

namespace Catalyst::Runtime::Device {
class OpenQasmDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST{true};
    static constexpr bool GLOBAL_RESULT_FALSE_CONST{false};

    static constexpr size_t default_device_shots{1000}; // tidy: readability-magic-numbers

    Simulator::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OpenQasm::OpenQasmBuilder> builder;
    std::unique_ptr<OpenQasm::OpenQasmRunner> runner;

    Simulator::CacheManager cache_manager{};
    bool tape_recording{false};

    size_t device_shots{0};
    OpenQasm::BuilderType builder_type;
    std::string concrete_device_name;

    inline auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>
    {
        std::vector<size_t> res;
        res.reserve(wires.size());
        std::transform(wires.begin(), wires.end(), std::back_inserter(res),
                       [this](auto w) { return qubit_manager.getDeviceId(w); });
        return res;
    }

    inline auto isValidQubits(const std::vector<QubitIdType> &wires) -> bool
    {
        return std::all_of(wires.begin(), wires.end(),
                           [this](QubitIdType w) { return qubit_manager.isValidQubitId(w); });
    }

  public:
    explicit OpenQasmDevice(
        [[maybe_unused]] bool status = false, size_t shots = default_device_shots,
        std::string hw_name = "arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        : tape_recording(status), device_shots(shots), concrete_device_name(std::move(hw_name))
    {
        builder_type = concrete_device_name.find("aws:braket") != std::string::npos
                           ? OpenQasm::BuilderType::Braket
                           : OpenQasm::BuilderType::Common;

        switch (builder_type) {
        case OpenQasm::BuilderType::Common:
            builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
            runner = std::make_unique<OpenQasm::OpenQasmRunner>();
            break;
        case OpenQasm::BuilderType::Braket:
            builder = std::make_unique<OpenQasm::BraketBuilder>();
            runner = std::make_unique<OpenQasm::BraketRunner>();
            break;
        default:
            RT_ASSERT("Invalid OpenQasm builder type");
        }
    }
    ~OpenQasmDevice() = default;

    OpenQasmDevice(const OpenQasmDevice &) = delete;
    OpenQasmDevice &operator=(const OpenQasmDevice &) = delete;
    OpenQasmDevice(OpenQasmDevice &&) = delete;
    OpenQasmDevice &operator=(OpenQasmDevice &&) = delete;

    // RT
    auto AllocateQubit() -> QubitIdType override;
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override;
    void ReleaseQubit(QubitIdType q) override;
    void ReleaseAllQubits() override;
    [[nodiscard]] auto GetNumQubits() const -> size_t override;
    void StartTapeRecording() override;
    void StopTapeRecording() override;
    void SetDeviceShots(size_t shots) override;
    [[nodiscard]] auto GetDeviceShots() const -> size_t override;
    void PrintState() override;
    [[nodiscard]] auto Zero() const -> Result override;
    [[nodiscard]] auto One() const -> Result override;

    // Circuit RT
    [[nodiscard]] auto Circuit() const -> std::string { return builder->toOpenQasm(); }

    // QIS
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse) override;
    void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse) override;
    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires) -> ObsIdType override;
    auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType override;
    auto HamiltonianObservable(const std::vector<double> &coeffs, const std::vector<ObsIdType> &obs)
        -> ObsIdType override;
    auto Expval(ObsIdType obsKey) -> double override;
    auto Var(ObsIdType obsKey) -> double override;
    void State(DataView<std::complex<double>, 1> &state) override;
    void Probs(DataView<double, 1> &probs) override;
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) override;
    void Sample(DataView<double, 2> &samples, size_t shots) override;
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires,
                       size_t shots) override;
    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts, size_t shots) override;
    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires, size_t shots) override;
    auto Measure(QubitIdType wire) -> Result override;
    void Gradient(std::vector<DataView<double, 1>> &gradients,
                  const std::vector<size_t> &trainParams) override;
};
} // namespace Catalyst::Runtime::Device
