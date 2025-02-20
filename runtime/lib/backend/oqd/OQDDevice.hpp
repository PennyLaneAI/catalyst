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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "OQDRuntimeCAPI.h"
#include "QuantumDevice.hpp"

// catalyst/runtime/lib/backend/common/

#include "CacheManager.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

#include "OQDRunner.hpp"

namespace Catalyst::Runtime::Device {
class OQDDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST{true};
    static constexpr bool GLOBAL_RESULT_FALSE_CONST{false};

    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<OQDRunner> runner;

    Catalyst::Runtime::CacheManager<std::complex<double>> cache_manager{};
    bool tape_recording{false};
    size_t device_shots;
    std::string ion_specs;
    std::vector<std::string> phonon_specs;

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
    explicit OQDDevice(const std::string &kwargs = "{device_type : oqd, backend : default}")
    {
        __catalyst__oqd__rt__initialize();

        // The OQD kwarg string format is:
        // deviceKwargs.str() + "ION:" + std::string(ion_json.dump()) + "PHONON:" +
        // std::string(phonon_json1.dump()) + ... where deviceKwargs are the usual keyword arguments
        // like {'shots': 0, 'mcmc': False}, ion_json is a JSON string specifying the ion
        // configuration, and phonon_json1, phonon_json2, etc. are JSON strings specifying phonon
        // configurations.
        std::string ion_token = "ION:";
        std::string phonon_token = "PHONON:";
        size_t ion_token_pos = kwargs.find(ion_token);
        if (ion_token_pos != std::string::npos) {
            size_t ion_start_pos = ion_token_pos + ion_token.length();
            size_t phonon_token_pos = kwargs.find(phonon_token);
            ion_specs = kwargs.substr(ion_start_pos, phonon_token_pos - ion_start_pos);
        }

        phonon_specs.clear();
        size_t phonon_token_pos = kwargs.find(phonon_token);
        while (phonon_token_pos != std::string::npos) {
            size_t phonon_start_pos = phonon_token_pos + phonon_token.length();
            phonon_token_pos = kwargs.find(phonon_token, phonon_start_pos);
            phonon_specs.push_back(
                kwargs.substr(phonon_start_pos, phonon_token_pos - phonon_start_pos));
        }

        device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs.substr(0, ion_token_pos));
        device_shots = device_kwargs.contains("shots")
                           ? static_cast<size_t>(std::stoll(device_kwargs["shots"]))
                           : 0;
        runner = std::make_unique<OQDRunner>();
    }
    ~OQDDevice() { __catalyst__oqd__rt__finalize(); };

    QUANTUM_DEVICE_DEL_DECLARATIONS(OQDDevice);

    QUANTUM_DEVICE_RT_DECLARATIONS;
    QUANTUM_DEVICE_QIS_DECLARATIONS;

    const std::string &getIonSpecs() { return ion_specs; }
    const std::vector<std::string> &getPhononSpecs() { return phonon_specs; }
};
} // namespace Catalyst::Runtime::Device
