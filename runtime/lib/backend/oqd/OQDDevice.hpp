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
#include "QubitManager.hpp"

namespace Catalyst::Runtime::Device {
class OQDDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};

    size_t device_shots;
    std::string ion_specs;
    std::string openapl_file_name;
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
        openapl_file_name = device_kwargs.contains("openapl_file_name")
                                ? device_kwargs["openapl_file_name"]
                                : "__openapl__output.json";
    }
    ~OQDDevice() { __catalyst__oqd__rt__finalize(openapl_file_name); };

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

    const std::string &getIonSpecs() { return ion_specs; }
    const std::vector<std::string> &getPhononSpecs() { return phonon_specs; }
    const std::string &getOutputFile() { return openapl_file_name; }
};
} // namespace Catalyst::Runtime::Device
