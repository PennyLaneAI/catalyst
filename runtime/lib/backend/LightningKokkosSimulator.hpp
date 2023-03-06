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

#if !__has_include("StateVectorKokkos.hpp")
throw std::logic_error("StateVectorKokkos.hpp: No such header file");
#endif

#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>

#include "AdjointDiffKokkos.hpp"
#include "StateVectorKokkos.hpp"

#include "CacheManager.hpp"
#include "LightningUtils.hpp"
#include "ObsManager.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"

namespace Catalyst::Runtime::Simulator {
class LightningKokkosSimulator final : public Catalyst::Runtime::QuantumDevice {
  private:
    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST = true;
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;

    QubitManager<QubitIdType, size_t> qubit_manager{};
    CacheManager cache_manager{};
    bool cache_recording{false};

    size_t device_shots{0};

    std::unique_ptr<Pennylane::StateVectorKokkos<double>> device_sv =
        std::make_unique<Pennylane::StateVectorKokkos<double>>(0);
    LightningKokkosObsManager<double> obs_manager{};

    inline auto isValidQubit(QubitIdType wire) -> bool
    {
        return qubit_manager.isValidQubitId(wire);
    }

    inline auto isValidQubits(const std::vector<QubitIdType> &wires) -> bool
    {
        return std::all_of(wires.begin(), wires.end(),
                           [this](QubitIdType w) { return this->isValidQubit(w); });
    }

    inline auto isValidQubits(size_t numWires, QubitIdType *wires) -> bool
    {
        return std::all_of(wires, wires + numWires,
                           [this](QubitIdType w) { return this->isValidQubit(w); });
    }

    inline auto getDeviceWires(size_t numWires, QubitIdType *wires) -> std::vector<size_t>
    {
        std::vector<size_t> res;
        res.reserve(numWires);
        std::transform(wires, wires + numWires, std::back_inserter(res),
                       [this](auto w) { return this->qubit_manager.getDeviceId(w); });
        return res;
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
    LightningKokkosSimulator(bool status = false, size_t shots = 1000)
        : cache_recording(status), device_shots(shots)
    {
    }
    ~LightningKokkosSimulator() = default;

    LightningKokkosSimulator(const LightningKokkosSimulator &) = delete;
    LightningKokkosSimulator &operator=(const LightningKokkosSimulator &) = delete;
    LightningKokkosSimulator(LightningKokkosSimulator &&) = delete;
    LightningKokkosSimulator &operator=(LightningKokkosSimulator &&) = delete;

    // RT
    auto AllocateQubit() -> QubitIdType override;
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override;
    void ReleaseQubit(QubitIdType q) override;
    void ReleaseAllQubits() override;
    auto GetNumQubits() const -> size_t override;
    void StartTapeRecording() override;
    void StopTapeRecording() override;
    void SetDeviceShots(size_t shots) override;
    auto GetDeviceShots() const -> size_t override;
    void PrintState() override;
    auto Zero() const -> Result override;
    auto One() const -> Result override;

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
    auto State() -> std::vector<std::complex<double>> override;
    auto Probs() -> std::vector<double> override;
    auto PartialProbs(const std::vector<QubitIdType> &wires) -> std::vector<double> override;
    auto Sample(size_t shots) -> std::vector<double> override;
    auto PartialSample(const std::vector<QubitIdType> &wires, size_t shots)
        -> std::vector<double> override;
    auto Counts(size_t shots) -> std::tuple<std::vector<double>, std::vector<int64_t>> override;
    auto PartialCounts(const std::vector<QubitIdType> &wires, size_t shots)
        -> std::tuple<std::vector<double>, std::vector<int64_t>> override;
    auto Measure(QubitIdType wire) -> Result override;
    auto Gradient(const std::vector<size_t> &trainParams)
        -> std::vector<std::vector<double>> override;
};
} // namespace Catalyst::Runtime::Simulator
