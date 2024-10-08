
// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ExecutionContext.hpp"
#include "NullQubit.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Devices;

TEST_CASE("Test success of loading a device", "[Null Qubit]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    CHECK(loadDevice("NullQubit", "librtd_null_qubit" + get_dylib_ext()));
}

TEST_CASE("Test __catalyst__rt__device_init registering device=null.qubit", "[Null Qubit]")
{
    __catalyst__rt__initialize(nullptr);

    char rtd_name[11] = "null.qubit";
    __catalyst__rt__device_init((int8_t *)rtd_name, nullptr, nullptr);

    __catalyst__rt__device_release();

    __catalyst__rt__finalize();
}

TEST_CASE("Test NullQubit loading is successful.", "[Null Qubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();
    sim->AllocateQubit();
}
