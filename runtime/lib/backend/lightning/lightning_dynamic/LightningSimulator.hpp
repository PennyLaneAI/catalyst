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

    static constexpr size_t default_num_burnin{100}; // tidy: readability-magic-numbers
    static constexpr std::string_view default_kernel_name{
        "Local"}; // tidy: readability-magic-numbers

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    Catalyst::Runtime::CacheManager<std::complex<double>> cache_manager{};
    bool tape_recording{false};
    size_t device_shots;

    std::mt19937 *gen = nullptr;

    bool mcmc{false};
    size_t num_burnin{0};
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
        device_shots = args.contains("shots") ? static_cast<size_t>(std::stoll(args["shots"])) : 0;
        mcmc = args.contains("mcmc") ? args["mcmc"] == "True" : false;
        num_burnin = args.contains("num_burnin")
                         ? static_cast<size_t>(std::stoll(args["num_burnin"]))
                         : default_num_burnin;
        kernel_name = args.contains("kernel_name") ? args["kernel_name"] : default_kernel_name;
    }
    ~LightningSimulator() override = default;

    void SetDevicePRNG(std::mt19937 *) override;
    void SetState(DataView<std::complex<double>, 1> &, std::vector<QubitIdType> &) override;
    void SetBasisState(DataView<int8_t, 1> &, std::vector<QubitIdType> &) override;

    QUANTUM_DEVICE_DEL_DECLARATIONS(LightningSimulator);

    QUANTUM_DEVICE_RT_DECLARATIONS;
    QUANTUM_DEVICE_QIS_DECLARATIONS;

    auto CacheManagerInfo()
        -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>;
    auto GenerateSamplesMetropolis(size_t shots) -> std::vector<size_t>;
    auto GenerateSamples(size_t shots) -> std::vector<size_t>;
};
} // namespace Catalyst::Runtime::Simulator
