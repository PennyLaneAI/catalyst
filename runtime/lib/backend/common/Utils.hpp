// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <algorithm>
#include <array>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "Exception.hpp"
#include "Types.h"

#define QUANTUM_DEVICE_DEL_DECLARATIONS(CLASSNAME)                                                 \
    CLASSNAME(const CLASSNAME &) = delete;                                                         \
    CLASSNAME &operator=(const CLASSNAME &) = delete;                                              \
    CLASSNAME(CLASSNAME &&) = delete;                                                              \
    CLASSNAME &operator=(CLASSNAME &&) = delete;

#define QUANTUM_DEVICE_RT_DECLARATIONS                                                             \
    auto AllocateQubit()->QubitIdType override;                                                    \
    auto AllocateQubits(size_t num_qubits)->std::vector<QubitIdType> override;                     \
    void ReleaseQubit(QubitIdType q) override;                                                     \
    void ReleaseAllQubits() override;                                                              \
    [[nodiscard]] auto GetNumQubits() const->size_t override;                                      \
    void StartTapeRecording() override;                                                            \
    void StopTapeRecording() override;                                                             \
    void SetDeviceShots(size_t shots) override;                                                    \
    [[nodiscard]] auto GetDeviceShots() const->size_t override;                                    \
    void PrintState() override;                                                                    \
    [[nodiscard]] auto Zero() const->Result override;                                              \
    [[nodiscard]] auto One() const->Result override;

#define QUANTUM_DEVICE_QIS_DECLARATIONS                                                            \
    void NamedOperation(                                                                           \
        const std::string &name, const std::vector<double> &params,                                \
        const std::vector<QubitIdType> &wires, [[maybe_unused]] bool inverse = false,              \
        [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires = {},                    \
        [[maybe_unused]] const std::vector<bool> &controlled_values = {}) override;                \
    using Catalyst::Runtime::QuantumDevice::MatrixOperation;                                       \
    void MatrixOperation(                                                                          \
        const std::vector<std::complex<double>> &matrix, const std::vector<QubitIdType> &wires,    \
        [[maybe_unused]] bool inverse = false,                                                     \
        [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires = {},                    \
        [[maybe_unused]] const std::vector<bool> &controlled_values = {}) override;                \
    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,                     \
                    const std::vector<QubitIdType> &wires)                                         \
        ->ObsIdType override;                                                                      \
    auto TensorObservable(const std::vector<ObsIdType> &obs)->ObsIdType override;                  \
    auto HamiltonianObservable(const std::vector<double> &coeffs,                                  \
                               const std::vector<ObsIdType> &obs)                                  \
        ->ObsIdType override;                                                                      \
    auto Expval(ObsIdType obsKey)->double override;                                                \
    auto Var(ObsIdType obsKey)->double override;                                                   \
    void State(DataView<std::complex<double>, 1> &state) override;                                 \
    void Probs(DataView<double, 1> &probs) override;                                               \
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) override; \
    void Sample(DataView<double, 2> &samples, size_t shots) override;                              \
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires,        \
                       size_t shots) override;                                                     \
    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts, size_t shots)          \
        override;                                                                                  \
    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,                 \
                       const std::vector<QubitIdType> &wires, size_t shots) override;              \
    auto Measure(QubitIdType wire, std::optional<int32_t> postselect = std::nullopt)               \
        ->Result override;                                                                         \
    void Gradient(std::vector<DataView<double, 1>> &gradients,                                     \
                  const std::vector<size_t> &trainParams) override;

namespace Catalyst::Runtime {
static inline auto parse_kwargs(std::string kwargs) -> std::unordered_map<std::string, std::string>
{
    // cleaning kwargs
    if (kwargs.empty()) {
        return {};
    }

    std::unordered_map<std::string, std::string> map;
    size_t s3_pos = kwargs.find("\'s3_destination_folder\'");
    if (s3_pos != std::string::npos) {
        auto opening_pos = kwargs.find('(', s3_pos);
        RT_ASSERT(opening_pos != std::string::npos);
        auto closing_pos = kwargs.find(')', opening_pos);
        RT_ASSERT(closing_pos != std::string::npos);
        map["s3_destination_folder"] = kwargs.substr(opening_pos, closing_pos - opening_pos + 1);
    }

    auto kwargs_end_iter = (s3_pos == std::string::npos) ? kwargs.end() : kwargs.begin() + s3_pos;

    kwargs.erase(std::remove_if(kwargs.begin(), kwargs_end_iter,
                                [](char c) {
                                    switch (c) {
                                    case '{':
                                    case '}':
                                    case ' ':
                                    case '\'':
                                        return true;
                                    default:
                                        return false;
                                    }
                                }),
                 kwargs.end());

    // constructing map
    std::istringstream iss(kwargs);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::istringstream issp(token);
        std::string pair[2];
        std::getline(issp, pair[0], ':');
        std::getline(issp, pair[1]);
        map[pair[0]] = pair[1];
    }

    return map;
}

enum class MeasurementsT : uint8_t {
    None, // = 0
    Expval,
    Var,
    Probs,
    State,
};

} // namespace Catalyst::Runtime

namespace Catalyst::Runtime::Simulator::Lightning {
enum class SimulatorGate : uint8_t {
    // 1-qubit
    Identity, // = 0
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    T,
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
    // 2-qubit
    CNOT,
    CY,
    CZ,
    SWAP,
    ISWAP,
    PSWAP,
    IsingXX,
    IsingYY,
    IsingXY,
    IsingZZ,
    ControlledPhaseShift,
    CRX,
    CRY,
    CRZ,
    CRot,
    // 3-qubit
    CSWAP,
    Toffoli,
    // n-qubit
    MultiRZ,
};

constexpr std::array simulator_observable_support = {
    // ObsId, ObsName, SimulatorSupport
    std::tuple<ObsId, std::string_view, bool>{ObsId::Identity, "Identity", true},
    std::tuple<ObsId, std::string_view, bool>{ObsId::PauliX, "PauliX", true},
    std::tuple<ObsId, std::string_view, bool>{ObsId::PauliY, "PauliY", true},
    std::tuple<ObsId, std::string_view, bool>{ObsId::PauliZ, "PauliZ", true},
    std::tuple<ObsId, std::string_view, bool>{ObsId::Hadamard, "Hadamard", true},
};

using GateInfoTupleT = std::tuple<SimulatorGate, std::string_view, size_t, size_t>;

constexpr std::array simulator_gate_info = {
    // 1-qubit
    GateInfoTupleT{SimulatorGate::Identity, "Identity", 1, 0},
    GateInfoTupleT{SimulatorGate::PauliX, "PauliX", 1, 0},
    GateInfoTupleT{SimulatorGate::PauliY, "PauliY", 1, 0},
    GateInfoTupleT{SimulatorGate::PauliZ, "PauliZ", 1, 0},
    GateInfoTupleT{SimulatorGate::Hadamard, "Hadamard", 1, 0},
    GateInfoTupleT{SimulatorGate::S, "S", 1, 0},
    GateInfoTupleT{SimulatorGate::T, "T", 1, 0},
    GateInfoTupleT{SimulatorGate::PhaseShift, "PhaseShift", 1, 1},
    GateInfoTupleT{SimulatorGate::RX, "RX", 1, 1},
    GateInfoTupleT{SimulatorGate::RY, "RY", 1, 1},
    GateInfoTupleT{SimulatorGate::RZ, "RZ", 1, 1},
    GateInfoTupleT{SimulatorGate::Rot, "Rot", 1, 3},
    // 2-qubit
    GateInfoTupleT{SimulatorGate::CNOT, "CNOT", 2, 0},
    GateInfoTupleT{SimulatorGate::CY, "CY", 2, 0},
    GateInfoTupleT{SimulatorGate::CZ, "CZ", 2, 0},
    GateInfoTupleT{SimulatorGate::SWAP, "SWAP", 2, 0},
    GateInfoTupleT{SimulatorGate::ISWAP, "ISWAP", 2, 0},
    GateInfoTupleT{SimulatorGate::PSWAP, "PSWAP", 2, 1},
    GateInfoTupleT{SimulatorGate::IsingXX, "IsingXX", 2, 1},
    GateInfoTupleT{SimulatorGate::IsingYY, "IsingYY", 2, 1},
    GateInfoTupleT{SimulatorGate::IsingXY, "IsingXY", 2, 1},
    GateInfoTupleT{SimulatorGate::IsingZZ, "IsingZZ", 2, 1},
    GateInfoTupleT{SimulatorGate::ControlledPhaseShift, "ControlledPhaseShift", 2, 1},
    GateInfoTupleT{SimulatorGate::CRX, "CRX", 2, 1},
    GateInfoTupleT{SimulatorGate::CRY, "CRY", 2, 1},
    GateInfoTupleT{SimulatorGate::CRZ, "CRZ", 2, 1},
    GateInfoTupleT{SimulatorGate::CRot, "CRot", 2, 3},
    // 3-qubit
    GateInfoTupleT{SimulatorGate::CSWAP, "CSWAP", 3, 0},
    GateInfoTupleT{SimulatorGate::Toffoli, "Toffoli", 3, 0},
    // n-qubit
    GateInfoTupleT{SimulatorGate::MultiRZ, "MultiRZ", 0, 1},
};

constexpr size_t simulator_gate_info_size = simulator_gate_info.size();
constexpr size_t simulator_observable_support_size = simulator_observable_support.size();

template <size_t size = simulator_gate_info_size>
using SimulatorGateInfoDataT = std::array<GateInfoTupleT, size>;

template <size_t size = simulator_observable_support_size>
constexpr auto lookup_obs(const std::array<std::tuple<ObsId, std::string_view, bool>, size> &arr,
                          const ObsId key) -> std::string_view
{
    for (size_t idx = 0; idx < size; idx++) {
        auto &&[op_id, op_str, op_support] = arr[idx];
        if (op_id == key && op_support) {
            return op_str;
        }
    }
    throw std::range_error("The given observable is not supported by the simulator");
}

template <size_t size = simulator_gate_info_size>
constexpr auto lookup_gates(const SimulatorGateInfoDataT<size> &arr, const std::string &key)
    -> std::pair<size_t, size_t>
{
    for (size_t idx = 0; idx < size; idx++) {
        auto &&[op, op_str, op_num_wires, op_num_params] = arr[idx];
        if (op_str == key) {
            return std::make_pair(op_num_wires, op_num_params);
        }
    }
    throw std::range_error("The given operation is not supported by the simulator");
}

template <size_t size = simulator_gate_info_size>
constexpr auto has_gate(const SimulatorGateInfoDataT<size> &arr, const std::string &key) -> bool
{
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<1>(arr[idx]) == key) {
            return true;
        }
    }
    return false;
}

static inline auto
simulateDraw(const std::vector<double> &probs, std::optional<int32_t> postselect,
             std::mt19937 *gen = nullptr) // NOLINT(readability-non-const-parameter)
    -> bool
{
    if (postselect) {
        auto postselect_value = postselect.value();
        RT_FAIL_IF(postselect_value < 0 || postselect_value > 1, "Invalid postselect value");
        RT_FAIL_IF(probs[postselect_value] == 0, "Probability of postselect value is 0");
        return static_cast<bool>(postselect_value == 1);
    }

    // Normal flow, no post-selection
    // Draw a number according to the given distribution
    std::uniform_real_distribution<> dis(0., 1.);

    float draw;
    if (gen != nullptr) {
        draw = dis(*gen);
        (*gen)();
    }
    else {
        std::random_device rd;
        std::mt19937 gen_no_seed(rd());
        draw = dis(gen_no_seed);
    }

    return draw > probs[0];
}

} // namespace Catalyst::Runtime::Simulator::Lightning
