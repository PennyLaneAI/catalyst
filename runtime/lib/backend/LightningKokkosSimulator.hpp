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

#include "BaseUtils.hpp"
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

    QubitManager<QubitIdType, size_t> qubit_manager;
    CacheManager cache_manager;
    bool cache_recording;

    size_t device_shots;

    std::unique_ptr<Pennylane::StateVectorKokkos<double>> device_sv;
    LightningKokkosObsManager<double> obs_manager;

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
        : qubit_manager(QubitManager<QubitIdType, size_t>()), cache_recording(status),
          cache_manager(CacheManager()), device_shots(shots),
          obs_manager(LightningKokkosObsManager<double>()),
          device_sv(std::make_unique<Pennylane::StateVectorKokkos<double>>(0))
    {
    }
    ~LightningKokkosSimulator() = default;

    // RT
    auto AllocateQubit() -> QubitIdType override;
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override;
    void ReleaseQubit(QubitIdType q) override;
    void ReleaseAllQubits() override;
    auto GetNumQubits() -> size_t override;
    void StartTapeRecording() override;
    void StopTapeRecording() override;
    void SetDeviceShots(size_t shots) override;
    auto GetDeviceShots() -> size_t override;
    void PrintState() override;
    auto DumpState() -> VectorCplxT<double> override;
    auto Zero() -> Result override;
    auto One() -> Result override;

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
    void State(CplxT_double *stateVec, size_t numAlloc) override;
    void Probs(double *probs, size_t numAlloc) override;
    void PartialProbs(double *probs, size_t numAlloc,
                      const std::vector<QubitIdType> &wires) override;
    void Sample(double *samples, size_t numAlloc, size_t shots) override;
    void PartialSample(double *samples, size_t numAlloc, const std::vector<QubitIdType> &wires,
                       size_t shots) override;
    void Counts(double *eigvals, int64_t *counts, size_t numAlloc, size_t shots) override;
    void PartialCounts(double *eigvals, int64_t *counts, size_t numAlloc,
                       const std::vector<QubitIdType> &wires, size_t shots) override;
    auto Measure(QubitIdType wire) -> Result override;
    auto Gradient(const std::vector<size_t> &trainParams)
        -> std::vector<std::vector<double>> override;
};
} // namespace Catalyst::Runtime::Simulator
