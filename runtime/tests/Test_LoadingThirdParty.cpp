
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

TEST_CASE("Test success of loading dummy device", "[Third Party]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>("default");
    std::string file("libdummy_device.so");
    CHECK(driver->initDevice(file));
}
