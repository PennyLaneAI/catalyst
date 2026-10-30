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

// Tests for lightning.cliffish, a device built in pennylane-lightning. The runner is built only
// when LIGHTNING_CLIFFISH_LIB names its library; see runtime/Makefile.

#include <cstdint>
#include <memory>
#include <string>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "TestUtils.hpp"

using namespace Catch::Matchers;

using namespace Catalyst::Runtime;

namespace {
const std::string rtd_lib{LIGHTNING_CLIFFISH_LIB};
const std::string rtd_name{"LightningCliffishSimulator"};
} // namespace

/**
 * @brief A test fixture that initializes the Catalyst Runtime and loads lightning.cliffish
 *        through the C-API with shots=100 and auto_qubit_management=false. The teardown
 *        operation releases the device and terminates the Catalyst Runtime.
 */
struct LightningCliffishRuntimeFixture {
    LightningCliffishRuntimeFixture() {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)"", 100, /* shots */
                                    false              /* auto_qubit_management */
        );
    }

    ~LightningCliffishRuntimeFixture() {
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
};

TEST_CASE("Test loading lightning.cliffish through its factory", "[LightningCliffish]") {
    std::unique_ptr<QuantumDevice> device(loadDevice(rtd_name, rtd_lib));
    REQUIRE(device);
    CHECK(device->GetNumQubits() == 0);

    auto qubits = device->AllocateQubits(3);
    CHECK(qubits.size() == 3);
    CHECK(device->GetNumQubits() == 3);

    device->ReleaseQubits(qubits);
    CHECK(device->GetNumQubits() == 0);
}

TEST_CASE_METHOD(LightningCliffishRuntimeFixture, "Test qubit allocation through the C-API",
                 "[LightningCliffish]") {
    QirArray *qs = __catalyst__rt__qubit_allocate_array(2);
    CHECK(__catalyst__rt__num_qubits() == 2);

    __catalyst__rt__qubit_release_array(qs);
    CHECK(__catalyst__rt__num_qubits() == 0);
}

TEST_CASE_METHOD(LightningCliffishRuntimeFixture,
                 "Test that operations fail loudly through the C-API", "[LightningCliffish]") {
    QirArray *qs = __catalyst__rt__qubit_allocate_array(2);
    QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);

    REQUIRE_THROWS_WITH(__catalyst__qis__Hadamard(*target, nullptr),
                        ContainsSubstring("lightning.cliffish cannot apply 'Hadamard' yet"));
    REQUIRE_THROWS_WITH(__catalyst__qis__Measure(*target, -1),
                        ContainsSubstring("lightning.cliffish cannot measure yet"));

    __catalyst__rt__qubit_release_array(qs);
}
