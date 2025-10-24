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

#include "QuantumDevice.hpp"
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

struct FTQCDevice final : public Catalyst::Runtime::QuantumDevice {
    // Static device parameters can be passed down from the Python device to the constructor.
    FTQCDevice(const std::string &kwargs = "{}") {}
    ~FTQCDevice() = default;

    auto GetNumQubits() const -> size_t override { return device_nqubits; }
    void SetDeviceShots(size_t shots) override { device_shots = shots; }
    auto GetDeviceShots() const -> size_t override { return device_shots; }

    // Allocations produce numeric qubit identifiers that are passed to other operations.
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override
    {
        std::vector<QubitIdType> ids(num_qubits);
        std::iota(ids.begin(), ids.end(), 0);
        device_nqubits = num_qubits;
        return ids;
    }
    void ReleaseQubits(const std::vector<QubitIdType> &qubits) override {}

    // Here we don't do anything yet.
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires,
                        const std::vector<bool> &controlled_values) override
    {
    }
    auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result override
    {
        return static_cast<Result>(malloc(sizeof(bool)));
    }

  private:
    size_t device_nqubits;
    size_t device_shots;
};

GENERATE_DEVICE_FACTORY(ftqc, FTQCDevice);