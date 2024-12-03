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

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OpenQasm::OpenQasmBuilder> builder;
    std::unique_ptr<OpenQasm::OpenQasmRunner> runner;

    Catalyst::Runtime::CacheManager<std::complex<double>> cache_manager{};
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

    QUANTUM_DEVICE_DEL_DECLARATIONS(OpenQasmDevice);

    QUANTUM_DEVICE_RT_DECLARATIONS;
    QUANTUM_DEVICE_QIS_DECLARATIONS;

    // Circuit RT
    [[nodiscard]] auto Circuit() const -> std::string { return builder->toOpenQasm(); }
};
} // namespace Catalyst::Runtime::Device
