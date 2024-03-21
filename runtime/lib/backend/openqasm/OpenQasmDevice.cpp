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
    if (!num_qubits) {
        return {};
    }

    const size_t cur_num_qubits = builder->getNumQubits();
    RT_FAIL_IF(cur_num_qubits, "Partial qubits allocation is not supported by OpenQasmDevice");

    const size_t new_num_qubits = cur_num_qubits + num_qubits;
    if (cur_num_qubits) {
        builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
    }

    builder->Register(OpenQasm::RegisterType::Qubit, "qubits", new_num_qubits);

    return qubit_manager.AllocateRange(cur_num_qubits, num_qubits);
}

void OpenQasmDevice::ReleaseAllQubits()
{
    // refresh the builder for device re-use.
    if (builder_type != OpenQasm::BuilderType::Common) {
        builder = std::make_unique<OpenQasm::BraketBuilder>();
    }
    else {
        builder = std::make_unique<OpenQasm::OpenQasmBuilder>();
    }
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

void OpenQasmDevice::PrintState()
{
    using std::cout;
    using std::endl;

    std::ostringstream oss;
    oss << "#pragma braket result state_vector";
    auto &&circuit = builder->toOpenQasmWithCustomInstructions(oss.str());

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&state = runner->State(circuit, device_info, device_shots, GetNumQubits(), s3_folder_str);

    const size_t num_qubits = GetNumQubits();
    const size_t size = 1UL << num_qubits;
    size_t idx = 0;
    cout << "*** State-Vector of Size " << size << " ***" << endl;
    cout << "[";

    for (; idx < size - 1; idx++) {
        cout << state[idx] << ", ";
    }
    cout << state[idx] << "]" << endl;
}

auto OpenQasmDevice::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto OpenQasmDevice::One() const -> Result { return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST); }

void OpenQasmDevice::NamedOperation(const std::string &name, const std::vector<double> &params,
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

void OpenQasmDevice::MatrixOperation(
    [[maybe_unused]] const std::vector<std::complex<double>> &matrix,
    [[maybe_unused]] const std::vector<QubitIdType> &wires, [[maybe_unused]] bool inverse,
    [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires,
    [[maybe_unused]] const std::vector<bool> &controlled_values)
{
    RT_FAIL_IF(builder_type == OpenQasm::BuilderType::Common, "Unsupported functionality");
    // TODO: Remove when controlled wires API is supported
    RT_FAIL_IF(!controlled_wires.empty() || !controlled_values.empty(),
               "OpenQasm device does not support native quantum control.");

    // Convert wires to device wires
    // with checking validity of wires
    auto &&dev_wires = getDeviceWires(wires);

    builder->Gate(matrix, dev_wires, inverse);
}

auto OpenQasmDevice::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                const std::vector<QubitIdType> &wires) -> ObsIdType
{
    RT_FAIL_IF(wires.size() > GetNumQubits(), "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);

    if (id == ObsId::Hermitian) {
        return obs_manager.createHermitianObs(matrix, dev_wires);
    }

    return obs_manager.createNamedObs(id, dev_wires);
}

auto OpenQasmDevice::TensorObservable([[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    return obs_manager.createTensorProdObs(obs);
}

auto OpenQasmDevice::HamiltonianObservable([[maybe_unused]] const std::vector<double> &coeffs,
                                           [[maybe_unused]] const std::vector<ObsIdType> &obs)
    -> ObsIdType
{
    return obs_manager.createHamiltonianObs(coeffs, obs);
}

auto OpenQasmDevice::Expval([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_ASSERT(builder->getQubits().size());
    RT_FAIL_IF(!obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");
    auto &&obs = obs_manager.getObservable(obsKey);
    RT_FAIL_IF(obs->getName() == "QasmHamiltonianObs",
               "Unsupported observable: QasmHamiltonianObs");

    std::ostringstream oss;
    oss << "#pragma braket result expectation " << obs->toOpenQasm(builder->getQubits()[0]);
    auto &&circuit = builder->toOpenQasmWithCustomInstructions(oss.str(), 9);

    // update tape caching
    if (tape_recording) {
        cache_manager.addObservable(obsKey, Catalyst::Runtime::MeasurementsT::Expval);
    }

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    return runner->Expval(circuit, device_info, device_shots, s3_folder_str);
}

auto OpenQasmDevice::Var([[maybe_unused]] ObsIdType obsKey) -> double
{
    RT_ASSERT(builder->getQubits().size());
    RT_FAIL_IF(!obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");
    auto &&obs = obs_manager.getObservable(obsKey);
    RT_FAIL_IF(obs->getName() == "QasmHamiltonianObs",
               "Unsupported observable: QasmHamiltonianObs");

    std::ostringstream oss;
    oss << "#pragma braket result variance " << obs->toOpenQasm(builder->getQubits()[0]);
    auto &&circuit = builder->toOpenQasmWithCustomInstructions(oss.str(), 9);

    // update tape caching
    if (tape_recording) {
        cache_manager.addObservable(obsKey, Catalyst::Runtime::MeasurementsT::Var);
    }

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    return runner->Var(circuit, device_info, device_shots, s3_folder_str);
}

void OpenQasmDevice::State([[maybe_unused]] DataView<std::complex<double>, 1> &state)
{
    std::ostringstream oss;
    oss << "#pragma braket result state_vector";
    auto &&circuit = builder->toOpenQasmWithCustomInstructions(oss.str());

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&dv_state =
        runner->State(circuit, device_info, device_shots, GetNumQubits(), s3_folder_str);
    RT_FAIL_IF(state.size() != dv_state.size(), "Invalid size for the pre-allocated state vector");

    std::move(dv_state.begin(), dv_state.end(), state.begin());
}

void OpenQasmDevice::Probs(DataView<double, 1> &probs)
{
    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&dv_probs = runner->Probs(builder->toOpenQasm(), device_info, device_shots,
                                    GetNumQubits(), s3_folder_str);

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void OpenQasmDevice::PartialProbs([[maybe_unused]] DataView<double, 1> &probs,
                                  [[maybe_unused]] const std::vector<QubitIdType> &wires)
{
    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    std::ostringstream oss;
    oss << "#pragma braket result probability "
        << builder->getQubits()[0].toOpenQasm(OpenQasm::RegisterMode::Slice, dev_wires);
    auto &&circuit = builder->toOpenQasmWithCustomInstructions(oss.str());

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&dv_probs =
        runner->Probs(circuit, device_info, device_shots, wires.size(), s3_folder_str);

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void OpenQasmDevice::Sample(DataView<double, 2> &samples, size_t shots)
{
    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&li_samples = runner->Sample(builder->toOpenQasm(), device_info, device_shots,
                                       GetNumQubits(), s3_folder_str);
    RT_FAIL_IF(samples.size() != li_samples.size(), "Invalid size for the pre-allocated samples");

    const size_t numQubits = GetNumQubits();

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
    const size_t numQubits = GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF(samples.size() != shots * numWires,
               "Invalid size for the pre-allocated partial-samples");

    // // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&li_samples = runner->Sample(builder->toOpenQasm(), device_info, device_shots,
                                       GetNumQubits(), s3_folder_str);

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
    const size_t numQubits = GetNumQubits();
    const size_t numElements = 1U << numQubits;

    RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
               "Invalid size for the pre-allocated counts");

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&li_samples = runner->Sample(builder->toOpenQasm(), device_info, device_shots,
                                       GetNumQubits(), s3_folder_str);

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = numQubits;
        for (size_t wire = 0; wire < numQubits; wire++) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

void OpenQasmDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                   const std::vector<QubitIdType> &wires, size_t shots)
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();
    const size_t numElements = 1U << numWires;

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF((eigvals.size() != numElements || counts.size() != numElements),
               "Invalid size for the pre-allocated partial-counts");

    auto &&dev_wires = getDeviceWires(wires);

    std::string s3_folder_str{};
    if (device_kwargs.contains("s3_destination_folder")) {
        s3_folder_str = device_kwargs["s3_destination_folder"];
    }

    std::string device_info{};
    if (builder_type == OpenQasm::BuilderType::BraketRemote) {
        device_info = device_kwargs["device_arn"];
    }
    else if (builder_type == OpenQasm::BuilderType::BraketLocal) {
        device_info = device_kwargs["backend"];
    }

    auto &&li_samples = runner->Sample(builder->toOpenQasm(), device_info, device_shots,
                                       GetNumQubits(), s3_folder_str);

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = dev_wires.size();
        for (auto wire : dev_wires) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

auto OpenQasmDevice::Measure([[maybe_unused]] QubitIdType wire, std::optional<int32_t> postselect)
    -> Result
{
    RT_FAIL_IF(postselect, "Post-selection is not supported yet");

    if (builder_type != OpenQasm::BuilderType::Common) {
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
    // TODO: custom implementation
    RT_FAIL("Unsupported functionality");
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(OpenQasmDevice, Catalyst::Runtime::Device::OpenQasmDevice);
