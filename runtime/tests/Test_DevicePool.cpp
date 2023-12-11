
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

#include <future>
#include <thread>

#include "ExecutionContext.hpp"
#include "QuantumDevice.hpp"
#include "QuantumDeviceInterface.hpp"
#include "RuntimeCAPI.h"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;

TEST_CASE("Test getter/setter methods in RTDeviceType", "[device_pool]")
{
    std::unique_ptr<RTDeviceType> device = std::make_unique<RTDeviceType>(
        "librtd_lightning" + get_dylib_ext(), "LightningSimulator", "{shots: 100}");
    std::cerr << *device << std::endl;
    auto &&[other_lib, other_name, other_kwargs] = device->getDeviceInfo();
    CHECK(*device == RTDeviceType(other_lib, other_name, other_kwargs));
    CHECK(device->getQuantumDevicePtr() != nullptr);
    CHECK(device->getQuantumDevicePtr() != nullptr);
    device->setTapeRecorderStatus(true);
    CHECK(device->getTapeRecorderStatus());

    std::unique_ptr<RTDeviceType> dev_oq3 = std::make_unique<RTDeviceType>("braket.aws.qubit");
    CHECK(*dev_oq3 == RTDeviceType("librtd_openqasm" + get_dylib_ext(), "OpenQasmDevice"));
}

TEST_CASE("Test the device pool in ExecutionContext", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    auto dev0 = driver->getDevice("lightning.qubit");
    auto dev1 = driver->getDevice("lightning.qubit");
    auto dev2 = driver->getDevice("lightning.qubit");
    auto dev3 = driver->getDevice("lightning.qubit");

    CHECK(dev0.first == 0);
    CHECK(dev1.first == 1);
    CHECK(dev2.first == 2);
    CHECK(dev3.first == 3);

    CHECK(driver->getDevice(dev0.first) != nullptr);
    CHECK(driver->getDevice(dev3.first) != nullptr);
    CHECK(driver->getDevice(50) == nullptr);

    driver->releaseDevice(dev1.first);
    driver->releaseDevice(dev3.first);

    CHECK(driver->getDevice(dev1.first) == nullptr);
    CHECK(driver->getDevice(dev3.first) == nullptr);
}

TEST_CASE("Test the released device re-use by the ExecutionContext manager", "[device_pool]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    auto dev1 = driver->getDevice("lightning.qubit");
    auto dev2 = driver->getDevice("lightning.qubit", "LightningQubit");
    auto dev3 = driver->getDevice(std::string("lightning.qubit"), std::string("LightningQubit"),
                                  std::string("{shots: 1000}"));

    CHECK(dev1.first == 0);
    CHECK(dev2.first == 1);
    CHECK(dev3.first == 2);

    driver->releaseDevice(dev1.first);
    auto dev4 = driver->getDevice("lightning.qubit");
    CHECK(dev4.first == dev1.first); // re-use dev1

    driver->releaseDevice(dev2.first);
    auto dev5 = driver->getDevice("lightning.qubit", "LightningQubit");
    CHECK(dev5.first == dev2.first); // re-use dev2

    auto dev6 = driver->getDevice("lightning.qubit", "LightningQubit");
    CHECK(dev6.first == 3);
}
