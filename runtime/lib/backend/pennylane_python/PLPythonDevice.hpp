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

#define __device_pennylane_python

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "PLPythonRunner.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"

namespace Catalyst::Runtime::Device {

/**
 * @brief A Catalyst runtime backend for PennyLane devices implemented in Python.
 *
 * Similar to the OpenQasmDevice (Braket), the device does not simulate anything itself. Every
 * quantum instruction of the running program is streamed, in program order, to the Python side of
 * the bridge (``catalyst.device.python_device``), which records it on a PennyLane tape. When a
 * terminal measurement is requested, the tape is completed with the terminal measurements of the
 * QNode and executed on the PennyLane device (with its own preprocessing); the results are
 * streamed back.
 *
 * Mid-circuit measurements are recorded on the tape; their outcomes are not available to the
 * program at runtime. Branches conditioned on them are rewritten at compile time into gates
 * controlled on "virtual" wires beyond the device wires, which the Python side turns into
 * ``Conditional`` operations.
 */
class PLPythonDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<PLPython::PLPythonRunner> runner;
    std::string device_kwargs;
    size_t session{0};
    size_t num_qubits{0};
    size_t device_shots{0};
    // Placeholder outcomes of mid-circuit measurements (stable addresses for ``Result``)
    std::deque<bool> mcm_results;

    [[nodiscard]] auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>;
    [[nodiscard]] auto call(const std::string &method, PLPython::Payload &payload)
        -> std::vector<double>;
    [[nodiscard]] auto measurement(const std::string &method, PLPython::Payload &&payload,
                                   size_t expected_size) -> std::vector<double>;

  public:
    explicit PLPythonDevice(const std::string &kwargs = "{}");
    ~PLPythonDevice() override = default;

    PLPythonDevice(const PLPythonDevice &) = delete;
    PLPythonDevice &operator=(const PLPythonDevice &) = delete;
    PLPythonDevice(PLPythonDevice &&) = delete;
    PLPythonDevice &operator=(PLPythonDevice &&) = delete;

    auto AllocateQubits(size_t) -> std::vector<QubitIdType> override;
    void ReleaseQubits(const std::vector<QubitIdType> &) override;
    auto GetNumQubits() const -> size_t override;
    void SetDeviceShots(size_t) override;
    auto GetDeviceShots() const -> size_t override;

    void NamedOperation(const std::string &, const std::vector<double> &,
                        const std::vector<QubitIdType> &, bool = false,
                        const std::vector<QubitIdType> & = {}, const std::vector<bool> & = {},
                        const std::vector<std::string> & = {}) override;
    void MatrixOperation(const std::vector<std::complex<double>> &,
                         const std::vector<QubitIdType> &, bool = false,
                         const std::vector<QubitIdType> & = {},
                         const std::vector<bool> & = {}) override;
    void SetState(DataView<std::complex<double>, 1> &, std::vector<QubitIdType> &) override;
    void SetBasisState(DataView<int8_t, 1> &, std::vector<QubitIdType> &) override;
    auto Measure(QubitIdType, std::optional<int32_t> = std::nullopt) -> Result override;

    auto Observable(ObsId, const std::vector<std::complex<double>> &,
                    const std::vector<QubitIdType> &) -> ObsIdType override;
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType override;
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType override;

    void Sample(DataView<double, 2> &) override;
    void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &) override;
    void Counts(DataView<double, 1> &, DataView<int64_t, 1> &) override;
    void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                       const std::vector<QubitIdType> &) override;
    void Probs(DataView<double, 1> &) override;
    void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) override;
    auto Expval(ObsIdType) -> double override;
    auto Var(ObsIdType) -> double override;
    void State(DataView<std::complex<double>, 1> &) override;
};
} // namespace Catalyst::Runtime::Device
