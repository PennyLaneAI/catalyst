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

#include "OpenQasmDevice.hpp"

namespace Catalyst::Runtime::Device {

auto OpenQasmDevice::AllocateQubit() -> QubitIdType
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::AllocateQubits([[maybe_unused]] size_t num_qubits) -> std::vector<QubitIdType>
{
    if (num_qubits == 0U) {
        return {};
    }

    const size_t cur_num_qubits = this->device->getNumQubits();
    const size_t new_num_qubits = cur_num_qubits + num_qubits;
    if (cur_num_qubits) {
        this->device = std::make_unique<OpenQasm::OpenQasmBuilder>();
    }

    this->device->Register(OpenQasm::RegisterType::Qubit, "qubits", new_num_qubits);

    return this->qubit_manager.AllocateRange(cur_num_qubits, new_num_qubits);
}

void OpenQasmDevice::ReleaseAllQubits()
{
    // do nothing
}

void OpenQasmDevice::ReleaseQubit([[maybe_unused]] QubitIdType q)
{
    RT_FAIL("Unsupported functionality");
}

auto OpenQasmDevice::GetNumQubits() const -> size_t { return this->device->getNumQubits(); }

void OpenQasmDevice::StartTapeRecording() { RT_FAIL("Unsupported functionality"); }

void OpenQasmDevice::StopTapeRecording() { RT_FAIL("Unsupported functionality"); }

void OpenQasmDevice::SetDeviceShots([[maybe_unused]] size_t shots)
{
    RT_FAIL("Unsupported functionality");
}

auto OpenQasmDevice::GetDeviceShots() const -> size_t
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

void OpenQasmDevice::PrintState() { RT_FAIL("Unsupported functionality"); }

auto OpenQasmDevice::Zero() const -> Result
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::One() const -> Result
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

void OpenQasmDevice::PrintCircuit() { std::cout << this->device->toOpenQasm(); }

auto OpenQasmDevice::DumpCircuit() -> std::string { return this->device->toOpenQasm(); }

void OpenQasmDevice::NamedOperation([[maybe_unused]] const std::string &name,
                                    [[maybe_unused]] const std::vector<double> &params,
                                    [[maybe_unused]] const std::vector<QubitIdType> &wires,
                                    [[maybe_unused]] bool inverse)
{
    using namespace Catalyst::Runtime::Simulator::Lightning;

    // First, check operation specifications
    auto &&[op_num_wires, op_num_params] = lookup_gates(simulator_gate_info, name);

    // Check the validity of number of qubits and parameters
    RT_FAIL_IF((!wires.size() && wires.size() != op_num_wires), "Invalid number of qubits");
    RT_FAIL_IF(params.size() != op_num_params, "Invalid number of parameters");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);

    this->device->Gate(name, params, {}, dev_wires, inverse);
}

void OpenQasmDevice::MatrixOperation(
    [[maybe_unused]] const std::vector<std::complex<double>> &matrix,
    [[maybe_unused]] const std::vector<QubitIdType> &wires, [[maybe_unused]] bool inverse)
{
    RT_FAIL("Unsupported functionality");
}

auto OpenQasmDevice::Observable([[maybe_unused]] ObsId id,
                                [[maybe_unused]] const std::vector<std::complex<double>> &matrix,
                                [[maybe_unused]] const std::vector<QubitIdType> &wires) -> ObsIdType
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::TensorObservable([[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::HamiltonianObservable([[maybe_unused]] const std::vector<double> &coeffs,
                                           [[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::Expval([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

auto OpenQasmDevice::Var([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_FAIL("Unsupported functionality");
    return 0;
}

void OpenQasmDevice::State([[maybe_unused]] DataView<std::complex<double>, 1> &state)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::Probs([[maybe_unused]] DataView<double, 1> &probs)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::PartialProbs([[maybe_unused]] DataView<double, 1> &probs,
                                  [[maybe_unused]] const std::vector<QubitIdType> &wires)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::Sample([[maybe_unused]] DataView<double, 2> &samples,
                            [[maybe_unused]] size_t shots)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::PartialSample([[maybe_unused]] DataView<double, 2> &samples,
                                   [[maybe_unused]] const std::vector<QubitIdType> &wires,
                                   [[maybe_unused]] size_t shots)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::Counts([[maybe_unused]] DataView<double, 1> &eigvals,
                            [[maybe_unused]] DataView<int64_t, 1> &counts,
                            [[maybe_unused]] size_t shots)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::PartialCounts([[maybe_unused]] DataView<double, 1> &eigvals,
                                   [[maybe_unused]] DataView<int64_t, 1> &counts,
                                   [[maybe_unused]] const std::vector<QubitIdType> &wires,
                                   [[maybe_unused]] size_t shots)
{
    RT_FAIL("Unsupported functionality");
}

auto OpenQasmDevice::Measure([[maybe_unused]] QubitIdType wire) -> Result
{
    // Convert wire to device wire
    auto &&dev_wire = getDeviceWires({wire});

    auto num_qubits = this->GetNumQubits();
    if (this->device->getNumBits() != num_qubits) {
        this->device->Register(OpenQasm::RegisterType::Bit, "bits", num_qubits);
    }

    this->device->Measure(dev_wire[0], dev_wire[0]);

    return 0;
}

// Gradient
void OpenQasmDevice::Gradient([[maybe_unused]] std::vector<DataView<double, 1>> &gradients,
                              [[maybe_unused]] const std::vector<size_t> &trainParams)
{
    RT_FAIL("Unsupported functionality");
}

} // namespace Catalyst::Runtime::Device
