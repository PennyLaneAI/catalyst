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

// The device ABI loader against devices that null.qubit does not cover. The devices are in
// stubs/stub_extension_device.cpp.

#include <cstdint>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "DeviceExtensions.h"
#include "RuntimeCAPI.h"
#include "Types.h"

using namespace Catch::Matchers;

namespace {

/// Initializes the runtime with a stub device, and releases both at scope exit.
struct StubDevice {
    StubDevice(const std::string &name, const std::string &kwargs) {
        __catalyst__rt__initialize(nullptr);
        std::string lib{RTD_STUB_EXTENSION_DEVICE_LIB};
        __catalyst__rt__device_init((int8_t *)lib.c_str(), (int8_t *)name.c_str(),
                                    (int8_t *)kwargs.c_str(), 0, false);
    }
    ~StubDevice() {
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
};

void require(const char *name, int32_t min_version) {
    __catalyst__rt__device_require((int8_t *)name, min_version);
}

auto allocateWire() -> QubitIdType {
    QirArray *reg = __catalyst__rt__qubit_allocate_array(1);
    return (*reinterpret_cast<std::vector<QubitIdType> *>(reg))[0];
}

/// Without a usable Delegated table, the runtime applies the sign of a Pauli measurement itself.
void checkHostSidePauliMeasure(QubitIdType wire) {
    CHECK(*__catalyst__qis__PauliMeasure("Z", true, nullptr, false, true, 1, wire) == true);
    CHECK(*__catalyst__qis__PauliMeasure("Z", false, nullptr, false, true, 1, wire) == false);
}

} // namespace

TEST_CASE("Test a device library without the version symbol is ABI 0 and never queried",
          "[DeviceExtensions]") {
    StubDevice device("StubLegacyDevice", "");

    // The stub fails if QueryInterface is called.
    REQUIRE_THROWS_WITH(require("Delegated", 1),
                        ContainsSubstring("device StubLegacyDevice (device ABI 0) does not"));
    REQUIRE_THROWS_WITH(__catalyst__qis__ResultConstant(true),
                        ContainsSubstring("needs the device extension Delegated >= 1"));

    QubitIdType wire = allocateWire();
    checkHostSidePauliMeasure(wire);
    REQUIRE_NOTHROW(__catalyst__qis__BeginGadget("Z", 0.5, CATALYST_GADGET_VERIFIED, 1, wire));
    REQUIRE_NOTHROW(__catalyst__qis__EndGadget());
}

TEST_CASE("Test an ABI 1 device without tables", "[DeviceExtensions]") {
    StubDevice device("StubExtensionDevice", "none");

    REQUIRE_THROWS_WITH(require("Delegated", 1),
                        ContainsSubstring("device StubExtensionDevice (device ABI 1) does not"));
    REQUIRE_THROWS_WITH(__catalyst__qis__BeginDelegatedProgram(CATALYST_POSTSELECT_FILL_SHOTS),
                        ContainsSubstring("needs the device extension Delegated >= 1"));

    QubitIdType wire = allocateWire();
    checkHostSidePauliMeasure(wire);
    REQUIRE_NOTHROW(__catalyst__qis__BeginGadget("Z", 0.5, CATALYST_GADGET_VERIFIED, 1, wire));
    REQUIRE_NOTHROW(__catalyst__qis__EndGadget());
}

TEST_CASE("Test a table older than requested is refused", "[DeviceExtensions]") {
    StubDevice device("StubExtensionDevice", "stale");

    REQUIRE_THROWS_WITH(require("Delegated", 1), ContainsSubstring("older than requested"));
}

TEST_CASE("Test a Delegated table without pauli_measure keeps the host-side sign",
          "[DeviceExtensions]") {
    StubDevice device("StubExtensionDevice", "no_pauli_measure");

    REQUIRE_NOTHROW(require("Delegated", 1));
    CHECK(*__catalyst__qis__ResultConstant(true) == false); // The stub's table answers.
    checkHostSidePauliMeasure(allocateWire());
}

TEST_CASE("Test invalid device extension requirements", "[DeviceExtensions]") {
    {
        StubDevice device("StubExtensionDevice", "none");
        REQUIRE_THROWS_WITH(require(nullptr, 1),
                            ContainsSubstring("Invalid device extension requirement"));
        REQUIRE_THROWS_WITH(require("Delegated", 0),
                            ContainsSubstring("Invalid device extension requirement"));
    }

    __catalyst__rt__initialize(nullptr);
    REQUIRE_THROWS_WITH(require("Delegated", 1), ContainsSubstring("without an active device"));
    REQUIRE_THROWS_WITH(__catalyst__qis__ResultConstant(true),
                        ContainsSubstring("without an active device"));
    REQUIRE_THROWS_WITH(__catalyst__qis__EndGadget(),
                        ContainsSubstring("without an active device"));
    __catalyst__rt__finalize();
}
