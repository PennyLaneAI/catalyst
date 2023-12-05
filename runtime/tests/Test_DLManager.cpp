
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

QuantumDevice *loadDevice(std::string device_name, std::string filename)
{
    std::unique_ptr<SharedLibraryManager> init_rtd_dylib =
        std::make_unique<SharedLibraryManager>(filename);
    std::string factory_name{device_name + "Factory"};
    void *f_ptr = init_rtd_dylib->getSymbol(factory_name);
    return f_ptr ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)("")
                 : nullptr; // LCOV_EXCL_LINE
}

TEST_CASE("Test dummy", "[Third Party]")
{
    std::string file("this-file-does-not-exist" + get_dylib_ext());
    REQUIRE_THROWS_WITH(loadDevice("DummyDevice", file),
                        Catch::Contains("No such file or directory"));
}

#ifdef __linux__
TEST_CASE("Test error message function not found", "[Third Party]")
{
    std::string file("libm.so.6");
    REQUIRE_THROWS_WITH(loadDevice("DummyDevice", file),
                        Catch::Contains("undefined symbol: DummyDeviceFactory"));
}

TEST_CASE("Test error message if init device fails", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    std::string file("libm.so.6");
    REQUIRE_THROWS_WITH(driver->initDevice(file), Catch::Contains("undefined symbol: Factory"));
}
#endif

TEST_CASE("Test success of loading dummy device", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    std::string file("libdummy_device" + get_dylib_ext());
    driver->setDeviceName("DummyDevice");
    CHECK(driver->initDevice(file));
}

TEST_CASE("Test __rt__device registering a custom device with shots=500 and device=lightning.qubit",
          "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "rtd_lib";
    char dev_value[17] = "lightning.qubit";
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);

    char dev2[7] = "device";
    char dev2_value[15] = "backend.other";
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev2, (int8_t *)dev_value),
                        Catch::Contains("[Function:__quantum__rt__device] Error in Catalyst "
                                        "Runtime: Invalid device specification"));

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev2_value),
                        Catch::Contains("cannot open shared object file"));

    REQUIRE_THROWS_WITH(__quantum__rt__device(nullptr, nullptr),
                        Catch::Contains("Invalid device specification"));

    __quantum__rt__finalize();

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Invalid use of the global driver before initialization"));
}

#ifdef __device_lightning_kokkos
TEST_CASE("Test __rt__device registering device=lightning.kokkos", "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "rtd_lib";
    char dev_value[18] = "lightning.kokkos";
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);

    __quantum__rt__finalize();
}
#endif
