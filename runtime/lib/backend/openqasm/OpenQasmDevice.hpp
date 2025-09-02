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
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "QuantumDevice.hpp"
#include "QubitManager.hpp"

#include "OpenQasmBuilder.hpp"
#include "OpenQasmObsManager.hpp"
#include "OpenQasmRunner.hpp"

namespace Catalyst::Runtime::Device {
class OpenQasmDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OpenQasm::OpenQasmBuilder> builder;
    std::unique_ptr<OpenQasm::OpenQasmRunner> runner;

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
        const std::string &kwargs = "{device_type : braket.local.qubit, backend : default}")
    {
        device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);

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

    auto AllocateQubits(size_t) -> std::vector<QubitIdType> override;
    void ReleaseAllQubits() override;
    auto GetNumQubits() const -> size_t override;
    void SetDeviceShots(size_t) override;
    auto GetDeviceShots() const -> size_t override;

    void NamedOperation(const std::string &, const std::vector<double> &,
                        const std::vector<QubitIdType> &, bool = false,
                        const std::vector<QubitIdType> & = {},
                        const std::vector<bool> & = {}) override;
    void MatrixOperation(const std::vector<std::complex<double>> &,
                         const std::vector<QubitIdType> &, bool = false,
                         const std::vector<QubitIdType> & = {},
                         const std::vector<bool> & = {}) override;
    auto Measure(QubitIdType, std::optional<int32_t> = std::nullopt) -> Result override;

    auto Observable(ObsId, const std::vector<std::complex<double>> &,
                    const std::vector<QubitIdType> &) -> ObsIdType override;
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType override;
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType override;

    void Sample(DataView<double, 2> &) override;
    void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &) override;
    void Counts(DataView<double, 1> &, DataView<int64_t, 1> &) override;
    void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                       const std::vector<QubitIdType> &) override;
    void Probs(DataView<double, 1> &) override;
    void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) override;
    auto Expval(ObsIdType) -> double override;
    auto Var(ObsIdType) -> double override;
    void State(DataView<std::complex<double>, 1> &) override;

    // Circuit RT
    [[nodiscard]] auto Circuit() const -> std::string { return builder->toOpenQasm(); }
};
} // namespace Catalyst::Runtime::Device
