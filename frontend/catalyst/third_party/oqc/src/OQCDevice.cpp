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

#include "Exception.hpp"

#include "OQCDevice.hpp"

namespace Catalyst::Runtime::Device {

auto OQCDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (!num_qubits) {
        return {};
    }

    builder = std::make_unique<OpenQASM2Builder>();

    builder->AddRegisters("qubits", num_qubits, "cbits", num_qubits);

    return qubit_manager.AllocateRange(0, num_qubits);
}

void OQCDevice::ReleaseAllQubits() { builder = std::make_unique<OpenQASM2Builder>(); }

auto OQCDevice::GetNumQubits() const -> size_t { return builder->getNumQubits(); }

void OQCDevice::SetDeviceShots(size_t shots) { device_shots = shots; }

auto OQCDevice::GetDeviceShots() const -> size_t { return device_shots; }

void OQCDevice::NamedOperation(const std::string &name, const std::vector<double> &params,
                               const std::vector<QubitIdType> &wires, bool inverse,
                               const std::vector<QubitIdType> &controlled_wires,
                               const std::vector<bool> &controlled_values)
{
    RT_FAIL_IF(!controlled_wires.empty() || !controlled_values.empty(),
               "OpenQasm device does not support native quantum control.");

    using namespace Catalyst::Runtime::Simulator::Lightning;

    // First, check operation specifications
    auto &&[op_num_wires, op_num_params] = lookup_gates(simulator_gate_info, name);

    // Check the validity of number of qubits and parameters
    RT_FAIL_IF((!wires.size() && wires.size() != op_num_wires), "Invalid number of qubits");
    RT_FAIL_IF(params.size() != op_num_params, "Invalid number of parameters");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);

    builder->AddGate(name, params, dev_wires);
}

void OQCDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                              const std::vector<QubitIdType> &wires)
{
    const size_t numQubits = GetNumQubits();
    // Add the measurements on the given wires
    for (auto wire : wires) {
        builder->AddMeasurement(wire, wire);
    }
    std::iota(eigvals.begin(), eigvals.end(), 0);

    auto &&results = runner->Counts(builder->toOpenQASM2(), "", device_shots, GetNumQubits());

    int i = 0;
    for (auto r : results) {
        counts(i) = r;
        i++;
    }
}

auto OQCDevice::Measure(QubitIdType, std::optional<int32_t>) -> Result
{
    RT_FAIL("Measure is unsupported by device");
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(oqc, Catalyst::Runtime::Device::OQCDevice);
