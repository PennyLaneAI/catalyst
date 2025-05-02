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

#include "QuantumDevice.hpp"
#include "QubitManager.hpp"

#include "OQCRunner.hpp"
#include "OpenQASM2Builder.hpp"

using namespace Catalyst::Runtime::OpenQASM2;

namespace Catalyst::Runtime::Device {
class OQCDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OpenQASM2Builder> builder;
    std::unique_ptr<OQCRunner> runner;

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

    auto AllocateQubits(size_t) -> std::vector<QubitIdType> override;
    void ReleaseAllQubits() override;
    auto GetNumQubits() const -> size_t override;
    void SetDeviceShots(size_t) override;
    auto GetDeviceShots() const -> size_t override;

    void NamedOperation(const std::string &, const std::vector<double> &,
                        const std::vector<QubitIdType> &, bool = false,
                        const std::vector<QubitIdType> & = {},
                        const std::vector<bool> & = {}) override;
    auto Measure(QubitIdType, std::optional<int32_t> = std::nullopt) -> Result override;

    void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                       const std::vector<QubitIdType> &) override;

    // Circuit RT
    [[nodiscard]] auto Circuit() const -> std::string { return builder->toOpenQASM2(); }
};
} // namespace Catalyst::Runtime::Device
