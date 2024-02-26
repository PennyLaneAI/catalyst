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

#define __device_lightning_kokkos

#include <bitset>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

#include "AdjointJacobianKokkos.hpp"
#include "MeasurementsKokkos.hpp"
#include "StateVectorKokkos.hpp"

#include "CacheManager.hpp"
#include "Exception.hpp"
#include "LightningKokkosObsManager.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

namespace Catalyst::Runtime::Simulator {
class LightningKokkosSimulator final : public Catalyst::Runtime::QuantumDevice {
  private:
    using StateVectorT = Pennylane::LightningKokkos::StateVectorKokkos<double>;

    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST = true;
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    Catalyst::Runtime::CacheManager<Kokkos::complex<double>> cache_manager{};
    bool tape_recording{false};

    size_t device_shots;

    std::unique_ptr<StateVectorT> device_sv = std::make_unique<StateVectorT>(0);
    LightningKokkosObsManager<double> obs_manager{};

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
    explicit LightningKokkosSimulator(const std::string &kwargs = "{}")
    {
        auto &&args = Catalyst::Runtime::parse_kwargs(kwargs);
        device_shots = args.contains("shots") ? static_cast<size_t>(std::stoll(args["shots"])) : 0;
    }
    ~LightningKokkosSimulator() = default;

    QUANTUM_DEVICE_DEL_DECLARATIONS(LightningKokkosSimulator);

    QUANTUM_DEVICE_RT_DECLARATIONS;
    QUANTUM_DEVICE_QIS_DECLARATIONS;

    auto CacheManagerInfo()
        -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>;
};
} // namespace Catalyst::Runtime::Simulator
