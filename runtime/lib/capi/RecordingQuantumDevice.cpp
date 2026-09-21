// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "RecordingQuantumDevice.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iterator>
#include <system_error>

#include "ResourceTracker.hpp"
#include "Utils.hpp"

namespace Catalyst::Runtime {
namespace {

int hexDigit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

std::string percentDecode(const std::string &value) {
    std::string decoded;
    decoded.reserve(value.size());
    for (size_t i = 0; i < value.size(); i++) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int high = hexDigit(value[i + 1]);
            const int low = hexDigit(value[i + 2]);
            if (high >= 0 && low >= 0) {
                decoded.push_back(static_cast<char>((high << 4) | low));
                i += 2;
                continue;
            }
        }
        decoded.push_back(value[i]);
    }
    return decoded;
}

std::string removeRecordingKwarg(const std::string &kwargs) {
    const auto key_pos = kwargs.find("record_device_calls");
    if (key_pos == std::string::npos) {
        return kwargs;
    }

    const auto open = kwargs.rfind('{', key_pos);
    const auto close = kwargs.find('}', key_pos);
    const auto left_comma = kwargs.rfind(',', key_pos);
    const auto right_comma = kwargs.find(',', key_pos);
    if (open == std::string::npos || close == std::string::npos) {
        return kwargs;
    }
    if (left_comma != std::string::npos && left_comma > open) {
        const auto token_end =
            right_comma != std::string::npos && right_comma < close ? right_comma : close;
        auto result = kwargs;
        result.erase(left_comma, token_end - left_comma);
        return result;
    }
    if (right_comma != std::string::npos && right_comma < close) {
        auto result = kwargs;
        result.erase(open + 1, right_comma - open);
        return result;
    }
    return "{}";
}

} // namespace

DeviceKwargsSplit splitDeviceKwargs(const std::string &kwargs) {
    DeviceKwargsSplit split;
    auto map = parse_kwargs(kwargs);
    auto it = map.find("record_device_calls");
    if (it == map.end()) {
        split.factory_kwargs = kwargs;
        return split;
    }

    split.record_path = percentDecode(it->second);
    split.factory_kwargs = removeRecordingKwarg(kwargs);
    return split;
}

RecordingQuantumDevice::RecordingQuantumDevice(std::unique_ptr<QuantumDevice> inner,
                                               const std::string &record_path)
    : inner_(std::move(inner)), tracker_(std::make_unique<ResourceTracker>()) {
    RT_FAIL_IF(!inner_, "RecordingQuantumDevice requires an inner device");
    RT_FAIL_IF(record_path.empty(), "RecordingQuantumDevice requires a non-empty record path");
    // ResourceTracker::WriteOut creates the file exclusively, so a leftover file from a previous
    // execution would make the write fail. The path is explicitly chosen by the user, so each
    // execution replaces it.
    std::error_code remove_error;
    std::filesystem::remove(record_path, remove_error);
    tracker_->SetResourcesFilename(record_path);
}

RecordingQuantumDevice::~RecordingQuantumDevice() {
    // A throw here would escape a noexcept destructor and abort the whole process.
    try {
        tracker_->WriteOut();
    } catch (const std::exception &e) {
        RT_WARN(e.what());
    } catch (...) {
        RT_WARN("Unknown error while writing recorded device resources");
    }
}

auto RecordingQuantumDevice::rawObservable(ObsIdType id) const -> ObsIdType {
    auto it = raw_observable_ids_.find(id);
    return it == raw_observable_ids_.end() ? id : it->second;
}

auto RecordingQuantumDevice::trackedObservable(ObsIdType id) const -> ObsIdType {
    auto it = tracked_observable_ids_.find(id);
    return it == tracked_observable_ids_.end() ? id : it->second;
}

auto RecordingQuantumDevice::registerObservable(ObsIdType raw_id, ObsIdType tracked_id)
    -> ObsIdType {
    const auto id = next_observable_id_++;
    raw_observable_ids_[id] = raw_id;
    tracked_observable_ids_[id] = tracked_id;
    return id;
}

auto RecordingQuantumDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> {
    auto qubits = inner_->AllocateQubits(num_qubits);
    for (auto qubit : qubits) {
        tracker_->AllocateQubit(qubit);
    }
    return qubits;
}

void RecordingQuantumDevice::ReleaseQubits(const std::vector<QubitIdType> &qubits) {
    inner_->ReleaseQubits(qubits);
    for (auto qubit : qubits) {
        tracker_->ReleaseQubit(qubit);
    }
}

auto RecordingQuantumDevice::GetNumQubits() const -> size_t { return inner_->GetNumQubits(); }

auto RecordingQuantumDevice::AllocateQubit() -> QubitIdType {
    auto qubit = inner_->AllocateQubit();
    tracker_->AllocateQubit(qubit);
    return qubit;
}

void RecordingQuantumDevice::ReleaseQubit(QubitIdType qubit) {
    inner_->ReleaseQubit(qubit);
    tracker_->ReleaseQubit(qubit);
}

void RecordingQuantumDevice::SetDeviceShots(size_t shots) { inner_->SetDeviceShots(shots); }

auto RecordingQuantumDevice::GetDeviceShots() const -> size_t { return inner_->GetDeviceShots(); }

void RecordingQuantumDevice::SetDevicePRNG(std::mt19937 *gen) { inner_->SetDevicePRNG(gen); }

void RecordingQuantumDevice::NamedOperation(const std::string &name,
                                            const std::vector<double> &params,
                                            const std::vector<QubitIdType> &wires, bool inverse,
                                            const std::vector<QubitIdType> &controlled_wires,
                                            const std::vector<bool> &controlled_values,
                                            const std::vector<std::string> &optional_params) {
    inner_->NamedOperation(name, params, wires, inverse, controlled_wires, controlled_values,
                           optional_params);
    tracker_->NamedOperation(name, inverse, wires, controlled_wires);
}

auto RecordingQuantumDevice::Measure(QubitIdType wire, std::optional<int32_t> postselect)
    -> Result {
    auto result = inner_->Measure(wire, postselect);
    tracker_->MidMeasurement();
    return result;
}

void RecordingQuantumDevice::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                             const std::vector<QubitIdType> &wires, bool inverse,
                                             const std::vector<QubitIdType> &controlled_wires,
                                             const std::vector<bool> &controlled_values) {
    inner_->MatrixOperation(matrix, wires, inverse, controlled_wires, controlled_values);
    tracker_->MatrixOperation(inverse, wires, controlled_wires);
}

void RecordingQuantumDevice::SetBasisState(DataView<int8_t, 1> &n,
                                           std::vector<QubitIdType> &wires) {
    inner_->SetBasisState(n, wires);
    tracker_->SetBasisState(wires);
}

void RecordingQuantumDevice::SetState(DataView<std::complex<double>, 1> &state,
                                      std::vector<QubitIdType> &wires) {
    inner_->SetState(state, wires);
    tracker_->SetState(wires);
}

auto RecordingQuantumDevice::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                        const std::vector<QubitIdType> &wires) -> ObsIdType {
    auto raw_id = inner_->Observable(id, matrix, wires);
    return registerObservable(raw_id, tracker_->Observable(id));
}

auto RecordingQuantumDevice::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType {
    std::vector<ObsIdType> raw_obs;
    raw_obs.reserve(obs.size());
    std::transform(obs.begin(), obs.end(), std::back_inserter(raw_obs),
                   [this](ObsIdType id) { return rawObservable(id); });
    auto raw_id = inner_->TensorObservable(raw_obs);
    return registerObservable(raw_id, tracker_->CombinedObservable("Prod", obs.size()));
}

auto RecordingQuantumDevice::HamiltonianObservable(const std::vector<double> &coeffs,
                                                   const std::vector<ObsIdType> &obs) -> ObsIdType {
    std::vector<ObsIdType> raw_obs;
    raw_obs.reserve(obs.size());
    std::transform(obs.begin(), obs.end(), std::back_inserter(raw_obs),
                   [this](ObsIdType id) { return rawObservable(id); });
    auto raw_id = inner_->HamiltonianObservable(coeffs, raw_obs);
    return registerObservable(raw_id, tracker_->CombinedObservable("Hamiltonian", obs.size()));
}

void RecordingQuantumDevice::Sample(DataView<double, 2> &samples) {
    inner_->Sample(samples);
    tracker_->AnalyticalMeasurement("sample", "all");
}

void RecordingQuantumDevice::PartialSample(DataView<double, 2> &samples,
                                           const std::vector<QubitIdType> &wires) {
    inner_->PartialSample(samples, wires);
    tracker_->AnalyticalMeasurement("sample", std::to_string(wires.size()));
}

void RecordingQuantumDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts) {
    inner_->Counts(eigvals, counts);
    tracker_->AnalyticalMeasurement("counts", "all");
}

void RecordingQuantumDevice::PartialCounts(DataView<double, 1> &eigvals,
                                           DataView<int64_t, 1> &counts,
                                           const std::vector<QubitIdType> &wires) {
    inner_->PartialCounts(eigvals, counts, wires);
    tracker_->AnalyticalMeasurement("counts", std::to_string(wires.size()));
}

void RecordingQuantumDevice::Probs(DataView<double, 1> &probs) {
    inner_->Probs(probs);
    tracker_->AnalyticalMeasurement("probs", "all");
}

void RecordingQuantumDevice::PartialProbs(DataView<double, 1> &probs,
                                          const std::vector<QubitIdType> &wires) {
    inner_->PartialProbs(probs, wires);
    tracker_->AnalyticalMeasurement("probs", std::to_string(wires.size()));
}

auto RecordingQuantumDevice::Expval(ObsIdType obsKey) -> double {
    auto result = inner_->Expval(rawObservable(obsKey));
    tracker_->ObsMeasurement("expval", trackedObservable(obsKey));
    return result;
}

auto RecordingQuantumDevice::Var(ObsIdType obsKey) -> double {
    auto result = inner_->Var(rawObservable(obsKey));
    tracker_->ObsMeasurement("var", trackedObservable(obsKey));
    return result;
}

void RecordingQuantumDevice::State(DataView<std::complex<double>, 1> &state) {
    inner_->State(state);
    tracker_->AnalyticalMeasurement("state", "all");
}

auto RecordingQuantumDevice::PauliMeasure(const std::string &pauli_word,
                                          const std::vector<QubitIdType> &wires) -> Result {
    auto result = inner_->PauliMeasure(pauli_word, wires);
    tracker_->PauliMeasure("PauliMeasure-w" + std::to_string(wires.size()), wires);
    return result;
}

void RecordingQuantumDevice::Gradient(std::vector<DataView<double, 1>> &gradients,
                                      const std::vector<size_t> &trainParams) {
    inner_->Gradient(gradients, trainParams);
}

void RecordingQuantumDevice::StartTapeRecording() { inner_->StartTapeRecording(); }

void RecordingQuantumDevice::StopTapeRecording() { inner_->StopTapeRecording(); }

} // namespace Catalyst::Runtime
