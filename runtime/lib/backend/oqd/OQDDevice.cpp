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

#include <algorithm>

#include "OQDDevice.hpp"
#include "OQDRuntimeCAPI.h"

namespace Catalyst::Runtime::Device {

auto OQDDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    for (size_t i = 0; i < num_qubits; i++) {
        __catalyst__oqd__ion(this->ion_specs);
    }
    __catalyst__oqd__modes(this->phonon_specs);

    // need to return a vector from 0 to num_qubits
    std::vector<QubitIdType> result(num_qubits);
    std::generate_n(result.begin(), num_qubits,
                    [&]() { return this->qubit_manager.Allocate(num_qubits); });
    return result;
}

void OQDDevice::ReleaseAllQubits()
{
    this->ion_specs = "";
    this->phonon_specs.clear();
    this->qubit_manager.ReleaseAll();
}

auto OQDDevice::GetNumQubits() const -> size_t { RT_FAIL("GetNumQubits unsupported by device"); }

void OQDDevice::SetDeviceShots([[maybe_unused]] size_t shots) { device_shots = shots; }

auto OQDDevice::GetDeviceShots() const -> size_t { return device_shots; }

void OQDDevice::NamedOperation(const std::string &, const std::vector<double> &,
                               const std::vector<QubitIdType> &, bool,
                               const std::vector<QubitIdType> &, const std::vector<bool> &)
{
    RT_FAIL("NamedOperation unsupported by device");
}

void OQDDevice::PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                              const std::vector<QubitIdType> &)
{
    // Note that we do not support this in OQD device.
    // This is a just a fake readout method for testing purposes.
    // TODO: change this back to unsupported functionality once null measurements
    // is supported in catalyst.
    return;
}

auto OQDDevice::Measure(QubitIdType, std::optional<int32_t>) -> Result
{
    RT_FAIL("Measure unsupported by device");
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(oqd, Catalyst::Runtime::Device::OQDDevice);
