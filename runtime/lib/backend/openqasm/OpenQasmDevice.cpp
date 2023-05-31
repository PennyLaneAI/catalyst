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
    return QubitIdType{};
}

auto OpenQasmDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (num_qubits == 0U) {
        return {};
    }

    const size_t cur_num_qubits = builder->getNumQubits();
    const size_t new_num_qubits = cur_num_qubits + num_qubits;
    if (cur_num_qubits) {
        builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
    }

    builder->Register(OpenQasm::RegisterType::Qubit, "qubits", new_num_qubits);

    return qubit_manager.AllocateRange(cur_num_qubits, new_num_qubits);
}

void OpenQasmDevice::ReleaseAllQubits()
{
    // do nothing
}

void OpenQasmDevice::ReleaseQubit([[maybe_unused]] QubitIdType q)
{
    RT_FAIL("Unsupported functionality");
}

auto OpenQasmDevice::GetNumQubits() const -> size_t { return builder->getNumQubits(); }

void OpenQasmDevice::StartTapeRecording()
{
    RT_FAIL_IF(tape_recording, "Cannot re-activate the cache manager");
    tape_recording = true;
    cache_manager.Reset();
}

void OpenQasmDevice::StopTapeRecording()
{
    RT_FAIL_IF(!tape_recording, "Cannot stop an already stopped cache manager");
    tape_recording = false;
}

void OpenQasmDevice::SetDeviceShots([[maybe_unused]] size_t shots) { device_shots = shots; }

auto OpenQasmDevice::GetDeviceShots() const -> size_t { return device_shots; }

void OpenQasmDevice::PrintState() { RT_FAIL("Unsupported functionality"); }

auto OpenQasmDevice::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto OpenQasmDevice::One() const -> Result { return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST); }

void OpenQasmDevice::PrintCircuit() { std::cout << builder->toOpenQasm(); }

void OpenQasmDevice::ExecuteCircuit(const std::string &hw_name)
{
    runner->runCircuit(builder->toOpenQasm(), hw_name, GetDeviceShots());
}

auto OpenQasmDevice::Circuit() -> std::string { return builder->toOpenQasm(); }

void OpenQasmDevice::NamedOperation(const std::string &name, const std::vector<double> &params,
                                    const std::vector<QubitIdType> &wires, bool inverse)
{
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
    return ObsIdType{};
}

auto OpenQasmDevice::TensorObservable([[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    RT_FAIL("Unsupported functionality");
    return ObsIdType{};
}

auto OpenQasmDevice::HamiltonianObservable([[maybe_unused]] const std::vector<double> &coeffs,
                                           [[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    RT_FAIL("Unsupported functionality");
    return ObsIdType{};
}

auto OpenQasmDevice::Expval([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_FAIL("Unsupported functionality");
    return double{};
}

auto OpenQasmDevice::Var([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_FAIL("Unsupported functionality");
    return double{};
}

void OpenQasmDevice::State([[maybe_unused]] DataView<std::complex<double>, 1> &state)
{
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::Probs(DataView<double, 1> &probs)
{
    auto &&dv_probs =
        runner->Probs(builder->toOpenQasm(), concrete_device_name, device_shots, GetNumQubits());

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void OpenQasmDevice::PartialProbs([[maybe_unused]] DataView<double, 1> &probs,
                                  [[maybe_unused]] const std::vector<QubitIdType> &wires)
{
    // TODO: custom implementation...
    RT_FAIL("Unsupported functionality");
}

void OpenQasmDevice::Sample(DataView<double, 2> &samples, size_t shots)
{
    auto &&li_samples =
        runner->Sample(builder->toOpenQasm(), concrete_device_name, device_shots, GetNumQubits());
    RT_FAIL_IF(samples.size() != li_samples.size(), "Invalid size for the pre-allocated samples");

    const size_t numQubits = this->GetNumQubits();

    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < shots; shot++) {
        for (size_t wire = 0; wire < numQubits; wire++) {
            *(samplesIter++) = static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}

void OpenQasmDevice::PartialSample(DataView<double, 2> &samples,
                                   const std::vector<QubitIdType> &wires, size_t shots)
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF(samples.size() != shots * numWires,
               "Invalid size for the pre-allocated partial-samples");

    // // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    auto &&li_samples =
        runner->Sample(builder->toOpenQasm(), concrete_device_name, device_shots, GetNumQubits());

    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < shots; shot++) {
        for (auto wire : dev_wires) {
            *(samplesIter++) = static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}

void OpenQasmDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                            size_t shots)
{
    const size_t numQubits = this->GetNumQubits();
    const size_t numElements = 1U << numQubits;

    RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
               "Invalid size for the pre-allocated counts");

    auto &&li_samples =
        runner->Sample(builder->toOpenQasm(), concrete_device_name, device_shots, GetNumQubits());

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = 0;
        for (size_t wire = 0; wire < numQubits; wire++) {
            basisState[idx++] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

void OpenQasmDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                   const std::vector<QubitIdType> &wires, size_t shots)
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();
    const size_t numElements = 1U << numWires;

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF((eigvals.size() != numElements || counts.size() != numElements),
               "Invalid size for the pre-allocated partial-counts");

    auto &&dev_wires = getDeviceWires(wires);

    auto &&li_samples =
        runner->Sample(builder->toOpenQasm(), concrete_device_name, device_shots, GetNumQubits());

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = 0;
        for (auto wire : dev_wires) {
            basisState[idx++] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

auto OpenQasmDevice::Measure([[maybe_unused]] QubitIdType wire) -> Result
{
    if (builder_type == OpenQasm::BuilderType::Braket) {
        RT_FAIL("Unsupported functionality");
        return Result{};
    }

    // Convert wire to device wire
    auto &&dev_wire = getDeviceWires({wire});

    auto num_qubits = GetNumQubits();
    if (builder->getNumBits() != num_qubits) {
        builder->Register(OpenQasm::RegisterType::Bit, "bits", num_qubits);
    }

    builder->Measure(dev_wire[0], dev_wire[0]);
    return Result{};
}

// Gradient
void OpenQasmDevice::Gradient([[maybe_unused]] std::vector<DataView<double, 1>> &gradients,
                              [[maybe_unused]] const std::vector<size_t> &trainParams)
{
    RT_FAIL("Unsupported functionality");
}

} // namespace Catalyst::Runtime::Device
