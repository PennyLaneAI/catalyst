// Copyright 2024 Xanadu Quantum Technologies Inc.

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

#include <algorithm>
#include <bitset>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "Exception.hpp"
#include "QuantumDevice.hpp"

// catalyst/runtime/lib/backend/common/

#include "CacheManager.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

#include "OQCRunner.hpp"
#include "OpenQASM2Builder.hpp"

using namespace Catalyst::Runtime::OpenQASM2;

namespace Catalyst::Runtime::Device {
class OQCDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST{true};
    static constexpr bool GLOBAL_RESULT_FALSE_CONST{false};

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OpenQASM2Builder> builder;
    std::unique_ptr<OQCRunner> runner;

    Catalyst::Runtime::CacheManager<std::complex<double>> cache_manager{};
    bool tape_recording{false};
    size_t device_shots;

    std::unordered_map<std::string, std::string> device_kwargs;

    inline auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>
    {
        std::vector<size_t> res;
        res.reserve(wires.size());
        std::transform(wires.begin(), wires.end(), std::back_inserter(res),
                       [this](auto w) { return qubit_manager.getDeviceId(w); });
        return res;
    }

  public:
    explicit OQCDevice(const std::string &kwargs = "{device_type : oqc, backend : default}")
    {
        device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);
        device_shots = device_kwargs.contains("shots")
                           ? static_cast<size_t>(std::stoll(device_kwargs["shots"]))
                           : 0;
        builder = std::make_unique<OpenQASM2Builder>();
        runner = std::make_unique<OQCRunner>();
    }
    ~OQCDevice() = default;

    QUANTUM_DEVICE_DEL_DECLARATIONS(OQCDevice);

    QUANTUM_DEVICE_RT_DECLARATIONS;
    QUANTUM_DEVICE_QIS_DECLARATIONS;

    // Circuit RT
    [[nodiscard]] auto Circuit() const -> std::string { return builder->toOpenQASM2(); }
};
} // namespace Catalyst::Runtime::Device
