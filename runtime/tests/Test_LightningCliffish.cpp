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

#include <cmath>
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

TEST_CASE_METHOD(LightningCliffishRuntimeFixture, "Test a Bell state through the C-API",
                 "[LightningCliffish]") {
    QirArray *qs = __catalyst__rt__qubit_allocate_array(2);
    QUBIT *q0 = *(QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
    QUBIT *q1 = *(QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

    __catalyst__qis__Hadamard(q0, nullptr);
    __catalyst__qis__CNOT(q0, q1, nullptr);

    // ZZ is +1 on every shot, so its estimate from samples is exact.
    const ObsIdType z0 = __catalyst__qis__NamedObs(ObsId::PauliZ, q0);
    const ObsIdType z1 = __catalyst__qis__NamedObs(ObsId::PauliZ, q1);
    CHECK(__catalyst__qis__Expval(__catalyst__qis__TensorObs(2, z0, z1)) == 1.0);

    const bool first = *__catalyst__qis__Measure(q0, -1);
    CHECK(*__catalyst__qis__Measure(q1, -1) == first);

    __catalyst__rt__qubit_release_array(qs);
}

TEST_CASE_METHOD(LightningCliffishRuntimeFixture,
                 "Test PauliRot and PauliMeasure through the C-API", "[LightningCliffish]") {
    QirArray *qs = __catalyst__rt__qubit_allocate_array(2);
    QUBIT *q0 = *(QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
    QUBIT *q1 = *(QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

    // A conditional rotation whose condition is false never reaches the device.
    __catalyst__qis__PauliRot("XX", M_PI, nullptr, false, 2, q0, q1);
    CHECK_FALSE(*__catalyst__qis__PauliMeasure("ZI", false, nullptr, false, true, 2, q0, q1));

    // exp(-i pi/2 XX) is XX up to a phase, taking |00> to |11>.
    __catalyst__qis__PauliRot("XX", M_PI, nullptr, true, 2, q0, q1);
    CHECK(*__catalyst__qis__Measure(q0, -1));
    CHECK_FALSE(*__catalyst__qis__PauliMeasure("ZZ", false, nullptr, false, true, 2, q0, q1));
    // The runtime flips a negated measurement's outcome.
    CHECK(*__catalyst__qis__PauliMeasure("ZZ", true, nullptr, false, true, 2, q0, q1));

    __catalyst__rt__qubit_release_array(qs);
}

TEST_CASE_METHOD(LightningCliffishRuntimeFixture,
                 "Test that non-Clifford gates fail loudly through the C-API",
                 "[LightningCliffish]") {
    QirArray *qs = __catalyst__rt__qubit_allocate_array(1);
    QUBIT *q0 = *(QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);

    // The device buffers ops, so the error surfaces when a value is demanded.
    __catalyst__qis__T(q0, nullptr);
    REQUIRE_THROWS_WITH(__catalyst__qis__Measure(q0, -1),
                        ContainsSubstring("lightning.cliffish cannot apply 'T'"));

    __catalyst__rt__qubit_release_array(qs);
}
