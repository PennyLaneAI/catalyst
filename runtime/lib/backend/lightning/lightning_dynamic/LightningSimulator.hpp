// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#define __device_lightning

#include <bitset>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>

#include "StateVectorLQubitDynamic.hpp"

#include "CacheManager.hpp"
#include "Exception.hpp"
#include "LightningObsManager.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

namespace Catalyst::Runtime::Simulator {
class LightningSimulator final : public Catalyst::Runtime::QuantumDevice {
  private:
    using StateVectorT = Pennylane::LightningQubit::StateVectorLQubitDynamic<double>;

    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST = true;
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;

    static constexpr size_t default_device_shots{1000}; // tidy: readability-magic-numbers
    static constexpr size_t default_num_burning{100};   // tidy: readability-magic-numbers
    static constexpr std::string_view default_kernel_name{
        "Local"}; // tidy: readability-magic-numbers

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    Catalyst::Runtime::CacheManager cache_manager{};
    bool tape_recording{false};
    size_t device_shots;

    bool mcmc{false};
    size_t num_burning{0};
    std::string kernel_name;

    std::unique_ptr<StateVectorT> device_sv = std::make_unique<StateVectorT>(0);
    LightningObsManager<double> obs_manager{};

    inline auto isValidQubit(QubitIdType wire) -> bool
    {
        return this->qubit_manager.isValidQubitId(wire);
    }

    inline auto isValidQubits(const std::vector<QubitIdType> &wires) -> bool
    {
        return std::all_of(wires.begin(), wires.end(),
                           [this](QubitIdType w) { return this->isValidQubit(w); });
    }

    inline auto isValidQubits(size_t numWires, const QubitIdType *wires) -> bool
    {
        return std::all_of(wires, wires + numWires,
                           [this](QubitIdType w) { return this->isValidQubit(w); });
    }

    inline auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>
    {
        std::vector<size_t> res;
        res.reserve(wires.size());
        std::transform(wires.begin(), wires.end(), std::back_inserter(res),
                       [this](auto w) { return this->qubit_manager.getDeviceId(w); });
        return res;
    }

  public:
    explicit LightningSimulator(const std::string &kwargs = "{}")
    {
        auto &&args = Catalyst::Runtime::parse_kwargs(kwargs);
        device_shots = args.contains("shots") ? static_cast<size_t>(std::stoll(args["shots"]))
                                              : default_device_shots;
        mcmc = args.contains("mcmc") ? args["mcmc"].compare("True") : false;
        num_burning = args.contains("num_burning")
                          ? static_cast<size_t>(std::stoll(args["num_burning"]))
                          : default_num_burning;
        kernel_name = args.contains("kernel_name") ? args["kernel_name"] : default_kernel_name;
    }
    ~LightningSimulator() override = default;

    LightningSimulator(const LightningSimulator &) = delete;
    LightningSimulator &operator=(const LightningSimulator &) = delete;
    LightningSimulator(LightningSimulator &&) = delete;
    LightningSimulator &operator=(LightningSimulator &&) = delete;

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

    auto CacheManagerInfo()
        -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>;

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
} // namespace Catalyst::Runtime::Simulator
