
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
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;

TEST_CASE("Test dummy", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>("default");
    std::string file("this-file-does-not-exist.so");
    REQUIRE_THROWS_WITH(driver->loadDevice(file), Catch::Contains("No such file or directory"));
}

TEST_CASE("Test error message function not found", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>("default");
    std::string file("libm.so.6");
    REQUIRE_THROWS_WITH(driver->loadDevice(file),
                        Catch::Contains("undefined symbol: getCustomDevice"));
}

TEST_CASE("Test return false if cannot init device", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>("default");
    std::string file("libm.so.6");
    CHECK(!driver->initDevice(file));
}

TEST_CASE("Test success of loading dummy device", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>("default");
    std::string file("libdummy_device.so");
    CHECK(driver->initDevice(file));
}

TEST_CASE("Test __rt__device registering a custom device with shots=500 and device=lightning",
          "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "backend";
    char dev_value[17] = "lightning.qubit";
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);

    char dev2[7] = "device";
    char dev2_value[15] = "backend.other";
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev2, (int8_t *)dev_value),
                        Catch::Contains("[Function:__quantum__rt__device] Error in Catalyst "
                                        "Runtime: Invalid device specification"));

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev2_value),
                        Catch::Contains("Failed initialization of the backend device"));

    REQUIRE_THROWS_WITH(__quantum__rt__device(nullptr, nullptr),
                        Catch::Contains("Invalid device specification"));

    __quantum__rt__finalize();

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Invalid use of the global driver before initialization"));
}

TEST_CASE("Test __rt__device registering the OpenQasm device", "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "backend";
    char dev_value[30] = "braket.aws.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();

    __quantum__rt__initialize();

    char dev_kwargs[20] = "kwargs";
    char dev_value_kwargs[70] = "device_arn : arn:aws:braket:::device/quantum-simulator/amazon/sv1";

    __quantum__rt__device((int8_t *)dev_kwargs, (int8_t *)dev_value_kwargs);

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();

    __quantum__rt__initialize();

    char dev_lcl[8] = "backend";
    char dev_value_lcl[30] = "braket.local.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev_lcl, (int8_t *)dev_value_lcl);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev_lcl, (int8_t *)dev_value_lcl),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();
}
