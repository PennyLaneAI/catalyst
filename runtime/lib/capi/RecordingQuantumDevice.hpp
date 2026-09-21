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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "QuantumDevice.hpp"

namespace Catalyst::Runtime {

struct ResourceTracker;

struct DeviceKwargsSplit {
    std::string factory_kwargs;
    std::string record_path;
};

DeviceKwargsSplit splitDeviceKwargs(const std::string &kwargs);

class RecordingQuantumDevice final : public QuantumDevice {
  private:
    std::unique_ptr<QuantumDevice> inner_;
    std::unique_ptr<ResourceTracker> tracker_;
    ObsIdType next_observable_id_{1};
    std::unordered_map<ObsIdType, ObsIdType> raw_observable_ids_;
    std::unordered_map<ObsIdType, ObsIdType> tracked_observable_ids_;

    [[nodiscard]] auto rawObservable(ObsIdType id) const -> ObsIdType;
    [[nodiscard]] auto trackedObservable(ObsIdType id) const -> ObsIdType;
    auto registerObservable(ObsIdType raw_id, ObsIdType tracked_id) -> ObsIdType;

  public:
    RecordingQuantumDevice(std::unique_ptr<QuantumDevice> inner, const std::string &record_path);
    ~RecordingQuantumDevice() override;

    RecordingQuantumDevice(const RecordingQuantumDevice &) = delete;
    RecordingQuantumDevice &operator=(const RecordingQuantumDevice &) = delete;
    RecordingQuantumDevice(RecordingQuantumDevice &&) = delete;
    RecordingQuantumDevice &operator=(RecordingQuantumDevice &&) = delete;

    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override;
    void ReleaseQubits(const std::vector<QubitIdType> &qubits) override;
    auto GetNumQubits() const -> size_t override;
    auto AllocateQubit() -> QubitIdType override;
    void ReleaseQubit(QubitIdType qubit) override;

    void SetDeviceShots(size_t shots) override;
    auto GetDeviceShots() const -> size_t override;
    void SetDevicePRNG(std::mt19937 *gen) override;

    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse = false,
                        const std::vector<QubitIdType> &controlled_wires = {},
                        const std::vector<bool> &controlled_values = {},
                        const std::vector<std::string> &optional_params = {}) override;
    auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result override;
    void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse = false,
                         const std::vector<QubitIdType> &controlled_wires = {},
                         const std::vector<bool> &controlled_values = {}) override;
    void SetBasisState(DataView<int8_t, 1> &n, std::vector<QubitIdType> &wires) override;
    void SetState(DataView<std::complex<double>, 1> &state,
                  std::vector<QubitIdType> &wires) override;

    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires) -> ObsIdType override;
    auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType override;
    auto HamiltonianObservable(const std::vector<double> &coeffs, const std::vector<ObsIdType> &obs)
        -> ObsIdType override;

    void Sample(DataView<double, 2> &samples) override;
    void PartialSample(DataView<double, 2> &samples,
                       const std::vector<QubitIdType> &wires) override;
    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts) override;
    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires) override;
    void Probs(DataView<double, 1> &probs) override;
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) override;
    auto Expval(ObsIdType obsKey) -> double override;
    auto Var(ObsIdType obsKey) -> double override;
    void State(DataView<std::complex<double>, 1> &state) override;
    auto PauliMeasure(const std::string &pauli_word, const std::vector<QubitIdType> &wires)
        -> Result override;

    void Gradient(std::vector<DataView<double, 1>> &gradients,
                  const std::vector<size_t> &trainParams) override;
    void StartTapeRecording() override;
    void StopTapeRecording() override;
};

} // namespace Catalyst::Runtime
