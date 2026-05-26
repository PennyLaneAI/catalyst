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

#pragma once

#define __device_pennylane_python

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "PLTapeBuilder.hpp"
#include "PLPythonObsManager.hpp"
#include "PLPythonRunner.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "Utils.hpp"

namespace Catalyst::Runtime::Device {

/**
 * @brief A Catalyst runtime backend that accumulates quantum operations into a
 *        PennyLane QuantumScript (tape) and executes it against an arbitrary
 *        PennyLane Python device plugin via a callback into the embedded Python
 *        interpreter.
 *
 * This mirrors the OpenQasmDevice architecture, but instead of building an
 * OpenQASM string the operations are serialized into a JSON tape representation,
 * which the companion nanobind Python module (`pennylane_python_module.cpp`)
 * deserializes into a PennyLane QuantumScript and submits to the target device.
 *
 * Decomposition to the target device's gate set is handled at compile time by
 * the Catalyst frontend (via the device TOML + PennyLane's decomposition system).
 * The C++ runtime only sees operations that the device already natively supports.
 */
class PLPythonDevice final : public Catalyst::Runtime::QuantumDevice {
  private:
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    std::unique_ptr<PLTape::PLTapeBuilder> builder;
    std::unique_ptr<PLTape::PLPythonRunner> runner;

    size_t device_shots;

    PLTape::PLPythonObsManager obs_manager{};

    std::set<QubitIdType> initial_allocated_QubitIds;
    std::unordered_map<std::string, std::string> device_kwargs;

    inline auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>
    {
        std::vector<size_t> res;
        res.reserve(wires.size());
        std::transform(wires.begin(), wires.end(), std::back_inserter(res),
                       [this](auto w) { return qubit_manager.getDeviceId(w); });
        return res;
    }

    inline auto isValidQubits(const std::vector<QubitIdType> &wires) -> bool
    {
        return std::all_of(wires.begin(), wires.end(),
                           [this](QubitIdType w) { return qubit_manager.isValidQubitId(w); });
    }

  public:
    explicit PLPythonDevice(const std::string &kwargs = "{}")
    {
        device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);
        builder = std::make_unique<PLTape::PLTapeBuilder>();
        runner = std::make_unique<PLTape::PLPythonRunner>();
    }
    ~PLPythonDevice() = default;

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

    // For debugging / testing
    [[nodiscard]] auto TapeJSON() const -> std::string { return builder->toJSON(); }
};
} // namespace Catalyst::Runtime::Device
