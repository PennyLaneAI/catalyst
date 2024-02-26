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

#include "OQCDevice.hpp"

namespace Catalyst::Runtime::Device {

auto OQCDevice::AllocateQubit() -> QubitIdType
{
    RT_FAIL("Unsupported functionality");
    return QubitIdType{};
}

auto OQCDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (!num_qubits) {
        return {};
    }

    const size_t cur_num_qubits = builder->getNumQubits();
    RT_FAIL_IF(cur_num_qubits, "Partial qubits allocation is not supported by OQCDevice");

    const size_t new_num_qubits = cur_num_qubits + num_qubits;
    if (cur_num_qubits) {
        builder = std::make_unique<OQC::OQCBuilder>();
    }

    builder->Register(OQC::RegisterType::Qubit, "qubits", new_num_qubits);

    return qubit_manager.AllocateRange(cur_num_qubits, num_qubits);
}

void OQCDevice::ReleaseAllQubits()
{
    // refresh the builder for device re-use.
    if (builder_type != OpenQasm::BuilderType::Common) {
        builder = std::make_unique<OpenQasm::BraketBuilder>();
    }
    else {
        builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
    }
}

void OQCDevice::ReleaseQubit([[maybe_unused]] QubitIdType q)
{
    RT_FAIL("Unsupported functionality");
}

auto OQCDevice::GetNumQubits() const -> size_t { return builder->getNumQubits(); }

void OQCDevice::StartTapeRecording()
{
    RT_FAIL_IF(tape_recording, "Cannot re-activate the cache manager");
    tape_recording = true;
    cache_manager.Reset();
}

void OQCDevice::StopTapeRecording()
{
    RT_FAIL_IF(!tape_recording, "Cannot stop an already stopped cache manager");
    tape_recording = false;
}

void OQCDevice::SetDeviceShots([[maybe_unused]] size_t shots) { device_shots = shots; }

auto OQCDevice::GetDeviceShots() const -> size_t { return device_shots; }


auto OQCDevice::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto OQCDevice::One() const -> Result { return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST); }

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

    builder->Gate(name, params, {}, dev_wires, inverse);
}

void OQCDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                            size_t shots)
{
    
    // TODO: COUNTS
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(OQCDevice, Catalyst::Runtime::Device::OQCDevice);
