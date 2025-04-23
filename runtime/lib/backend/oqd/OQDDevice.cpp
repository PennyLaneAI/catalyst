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

auto OQDDevice::AllocateQubit() -> QubitIdType { RT_FAIL("Unsupported functionality"); }

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

void OQDDevice::ReleaseQubit([[maybe_unused]] QubitIdType q)
{
    RT_FAIL("Unsupported functionality");
}

auto OQDDevice::GetNumQubits() const -> size_t { RT_FAIL("Unsupported functionality"); }

void OQDDevice::StartTapeRecording()
{
    RT_FAIL_IF(tape_recording, "Cannot re-activate the cache manager");
    tape_recording = true;
    cache_manager.Reset();
}

void OQDDevice::StopTapeRecording()
{
    RT_FAIL_IF(!tape_recording, "Cannot stop an already stopped cache manager");
    tape_recording = false;
}

void OQDDevice::SetDeviceShots([[maybe_unused]] size_t shots) { device_shots = shots; }

auto OQDDevice::GetDeviceShots() const -> size_t { return device_shots; }

auto OQDDevice::Zero() const -> Result { return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST); }

auto OQDDevice::One() const -> Result { return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST); }

void OQDDevice::NamedOperation(const std::string &name, const std::vector<double> &params,
                               const std::vector<QubitIdType> &wires, bool inverse,
                               const std::vector<QubitIdType> &controlled_wires,
                               const std::vector<bool> &controlled_values)
{
    RT_FAIL("Unsupported functionality");
}

void OQDDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                              const std::vector<QubitIdType> &wires)
{
    // Note that we do not support this in OQD device.
    // This is a just a fake readout method for testing purposes.
    // TODO: change this back to unsupported functionality once null measurements
    // is supported in catalyst.
    return;
}

void OQDDevice::PrintState() { RT_FAIL("Unsupported functionality"); }

void OQDDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts)
{
    RT_FAIL("Unsupported functionality");
}

auto OQDDevice::Measure([[maybe_unused]] QubitIdType wire, std::optional<int32_t> postselect)
    -> Result
{
    RT_FAIL("Unsupported functionality");
}

ObsIdType OQDDevice::Observable(ObsId, const std::vector<std::complex<double>> &,
                                const std::vector<QubitIdType> &)
{
    RT_FAIL("Unsupported functionality");
}

ObsIdType OQDDevice::TensorObservable(const std::vector<ObsIdType> &)
{
    RT_FAIL("Unsupported functionality");
};

ObsIdType OQDDevice::HamiltonianObservable(const std::vector<double> &,
                                           const std::vector<ObsIdType> &)
{
    RT_FAIL("Unsupported functionality");
}

void OQDDevice::MatrixOperation(const std::vector<std::complex<double>> &,
                                const std::vector<QubitIdType> &, bool,
                                const std::vector<QubitIdType> &, const std::vector<bool> &)
{
    RT_FAIL("Unsupported functionality");
}

double OQDDevice::Expval(ObsIdType) { RT_FAIL("Unsupported functionality"); };
double OQDDevice::Var(ObsIdType) { RT_FAIL("Unsupported functionality"); };
void OQDDevice::State(DataView<std::complex<double>, 1> &)
{
    RT_FAIL("Unsupported functionality");
};
void OQDDevice::Probs(DataView<double, 1> &) { RT_FAIL("Unsupported functionality"); };
void OQDDevice::PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &)
{
    RT_FAIL("Unsupported functionality");
};
void OQDDevice::Sample(DataView<double, 2> &) { RT_FAIL("Unsupported functionality"); };
void OQDDevice::PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &)
{
    RT_FAIL("Unsupported functionality");
}

void OQDDevice::Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &)
{
    RT_FAIL("Unsupported functionality");
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(oqd, Catalyst::Runtime::Device::OQDDevice);
