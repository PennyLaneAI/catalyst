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
#include <unordered_map>
#include <vector>

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#include "CacheManager.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

#include "OpenQasmBuilder.hpp"
#include "OpenQasmObsManager.hpp"
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

    size_t device_shots;
    OpenQasm::OpenQasmObsManager obs_manager{};
    OpenQasm::BuilderType builder_type;
    std::unordered_map<std::string, std::string> device_kwargs;

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
        [[maybe_unused]] bool status = false,
        const std::string &kwargs = "{device_type : braket.local.qubit, backend : default}")
        : tape_recording(status)
    {
        device_kwargs = Simulator::parse_kwargs(kwargs);
        device_shots = device_kwargs.contains("shots")
                           ? static_cast<size_t>(std::stoll(device_kwargs["shots"]))
                           : default_device_shots;

        if (device_kwargs.contains("device_type")) {
            if (device_kwargs["device_type"] == "braket.aws.qubit") {
                builder_type = OpenQasm::BuilderType::BraketRemote;
                if (!device_kwargs.contains("device_arn")) {
                    device_kwargs["device_arn"] =
                        "arn:aws:braket:::device/quantum-simulator/amazon/sv1";
                }
            }
            else if (device_kwargs["device_type"] == "braket.local.qubit") {
                builder_type = OpenQasm::BuilderType::BraketLocal;
                if (!device_kwargs.contains("backend")) {
                    device_kwargs["backend"] = "default";
                }
            }
            else {
                RT_ASSERT("Invalid OpenQasm device type");
            }
        }
        else {
            builder_type = OpenQasm::BuilderType::Common;
            builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
            runner = std::make_unique<OpenQasm::OpenQasmRunner>();
        }

        if (builder_type != OpenQasm::BuilderType::Common) {
            builder = std::make_unique<OpenQasm::BraketBuilder>();
            runner = std::make_unique<OpenQasm::BraketRunner>();
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
