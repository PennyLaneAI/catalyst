// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "RuntimeCAPI.h"

// -------------------------------------------------------------------------- //
// MBQC Runtime Tests
// -------------------------------------------------------------------------- //

TEST_CASE("Test __catalyst__mbqc__measure_in_basis, device=null.qubit", "[MBQC]")
{
    __catalyst__rt__initialize(nullptr);

    const std::string rtd_name{"null.qubit"};
    __catalyst__rt__device_init((int8_t *)rtd_name.c_str(), nullptr, nullptr, 0, false);

    const size_t num_qubits = 1;
    QirArray *qs = __catalyst__rt__qubit_allocate_array(num_qubits);

    QUBIT **q0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);

    CHECK(reinterpret_cast<QubitIdType>(*q0) == 0);

    // Recall that the basis states for arbitrary-basis measurements are parameterized by a plane
    // (either XY, YZ or ZX) and a rotation angle about that plane. See the `mbqc.measure_in_basis`
    // op definition in mlir/include/MBQC/IR/MBQCOps.td for details on these parameters and how they
    // are encoded.
    RESULT *mres = __catalyst__mbqc__measure_in_basis(*q0, 0U /*plane*/, 0.0 /*angle*/, -1);

    CHECK(*mres == false); // For null.qubit, measurement result is always 0 (false)

    __catalyst__rt__device_release();
    __catalyst__rt__finalize();
}
