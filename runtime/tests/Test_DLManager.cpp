
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

    // LCOV_EXCL_START
    return f_ptr ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)("") : nullptr;
    // LCOV_EXCL_STOP
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
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    std::string file("libm.so.6");
    REQUIRE_THROWS_WITH(loadDevice("DummyDevice", file),
                        Catch::Contains("undefined symbol: DummyDeviceFactory"));
}

TEST_CASE("Test error message if init device fails", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    std::string file("libm.so.6");
    REQUIRE_THROWS_WITH(loadDevice("", file), Catch::Contains("undefined symbol: Factory"));
}
#endif

TEST_CASE("Test success of loading dummy device", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    CHECK(loadDevice("DummyDevice", "librtd_dummy" + get_dylib_ext()));
}

TEST_CASE("Test __catalyst__rt__device_init registering a custom device with shots=500 and "
          "device=lightning.qubit",
          "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);

    char dev1[17] = "lightning.qubit";
    __catalyst__rt__device_init((int8_t *)dev1, nullptr, nullptr);
    __catalyst__rt__device_release();

    char dev2[15] = "backend.other";
    REQUIRE_THROWS_WITH(__catalyst__rt__device_init((int8_t *)dev2, nullptr, nullptr),
                        Catch::Contains("cannot open shared object file"));

    REQUIRE_THROWS_WITH(__catalyst__rt__device_init(nullptr, nullptr, nullptr),
                        Catch::Contains("Invalid device library"));

    __catalyst__rt__finalize();

    REQUIRE_THROWS_WITH(__catalyst__rt__device_init((int8_t *)dev1, nullptr, nullptr),
                        Catch::Contains("Invalid use of the global driver before initialization"));
}

#ifdef __device_lightning_kokkos
TEST_CASE("Test __catalyst__rt__device_init registering device=lightning.kokkos", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);

    char rtd_name[18] = "lightning.kokkos";
    __catalyst__rt__device_init((int8_t *)rtd_name, nullptr, nullptr);

    __catalyst__rt__device_release();

    __catalyst__rt__finalize();
}
#endif
