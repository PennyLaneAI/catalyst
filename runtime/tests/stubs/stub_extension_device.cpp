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

// Devices for testing the device ABI loader against libraries that null.qubit does not cover:
// - StubLegacyDevice has a factory but no `<Name>DeviceAbiVersion`, like a library built
//   before device ABI 1, and fails if it is ever queried for a table.
// - StubExtensionDevice is ABI 1, and returns the tables its kwargs name:
//   "none" (no tables), "stale" (a table older than requested), or "no_pauli_measure" (a
//   Delegated table whose optional `pauli_measure` is null).

#include <string>
#include <vector>

#include "DeviceExtensions.h"
#include "QuantumDevice.hpp"

using namespace Catalyst::Runtime;

namespace {

struct StubDevice : QuantumDevice {
    size_t num_qubits{0};
    size_t shots{0};

    auto AllocateQubits(size_t n) -> std::vector<QubitIdType> override {
        std::vector<QubitIdType> ids(n);
        for (size_t i = 0; i < n; i++) {
            ids[i] = static_cast<QubitIdType>(num_qubits + i);
        }
        num_qubits += n;
        return ids;
    }
    void ReleaseQubits(const std::vector<QubitIdType> &) override { num_qubits = 0; }
    auto GetNumQubits() const -> size_t override { return num_qubits; }
    void SetDeviceShots(size_t s) override { shots = s; }
    auto GetDeviceShots() const -> size_t override { return shots; }
    void NamedOperation(const std::string &, const std::vector<double> &,
                        const std::vector<QubitIdType> &, bool, const std::vector<QubitIdType> &,
                        const std::vector<bool> &, const std::vector<std::string> &) override {}
    auto Measure(QubitIdType, std::optional<int32_t>) -> Result override { return falseResult(); }
    auto PauliMeasure(const std::string &, const std::vector<QubitIdType> &) -> Result override {
        return falseResult();
    }

    static auto falseResult() -> Result { return const_cast<Result>(&RESULT_FALSE); }
    static constexpr bool RESULT_FALSE = false;
};

struct StubLegacyDevice final : StubDevice {
    explicit StubLegacyDevice(const std::string &) {}
    auto QueryInterface(const char *, uint32_t) -> const void * override {
        RT_FAIL("A version-0 device must never be queried for a table");
    }
};

struct StubExtensionDevice final : StubDevice {
    std::string mode;
    CatalystDelegatedTable delegated{};

    explicit StubExtensionDevice(const std::string &kwargs) {
        for (const char *m : {"none", "stale", "no_pauli_measure"}) {
            if (kwargs.find(m) != std::string::npos) {
                mode = m;
            }
        }
        delegated.header = {.abi_version = mode == "stale" ? 0u : 1u, .reserved = 0, .self = this};
        delegated.result_constant = [](void *, bool) { return falseResult(); };
    }

    auto QueryInterface(const char *name, uint32_t) -> const void * override {
        if (mode == "none" || std::string(name) != "Delegated") {
            return nullptr;
        }
        return &delegated;
    }
};

} // namespace

// The factory of a library built before device ABI 1: no `StubLegacyDeviceDeviceAbiVersion`.
extern "C" QuantumDevice *StubLegacyDeviceFactory(const char *kwargs) {
    return new StubLegacyDevice(std::string(kwargs));
}

GENERATE_DEVICE_FACTORY(StubExtensionDevice, StubExtensionDevice);
