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

#include <stdexcept>
#include <array>
#include <tuple>
#include <string_view>

#include "Types.h"

// #if __has_include("StateVectorKokkos.hpp")
// // this macro is used in the C++ test suite
// #define _KOKKOS
// #endif

namespace Catalyst::Runtime::Simulator {
static inline void QFailIf(bool condition, const char *message)
{
    if (condition) {
        throw std::runtime_error(message);
    }
}
} // namespace Catalyst::Runtime::Simulator

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

enum class Measurements : uint8_t {
    None, // = 0
    Expval,
    Var,
    Probs,
    State,
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

} // namespace Catalyst::Runtime::Simulator::Lightning
