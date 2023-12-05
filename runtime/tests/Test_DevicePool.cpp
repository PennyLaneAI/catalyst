
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

TEST_CASE("Test getter/setter methods in RTDeviceInfoT", "[device_pool]")
{
    std::unique_ptr<RTDeviceInfoT> device = std::make_unique<RTDeviceInfoT>(
        "librtd_lightning" + get_dylib_ext(), "LightningSimulator", "{shots: 100}");
    std::cerr << *device << std::endl;
    auto &&[other_lib, other_name, other_kwargs] = device->getDeviceInfo();
    CHECK(*device == RTDeviceInfoT(other_lib, other_name, other_kwargs));
    CHECK(device->getQuantumDevicePtr() != nullptr);
    CHECK(device->getQuantumDevicePtr() != nullptr);
    device->setTapeRecorderStatus(true);
    CHECK(device->getTapeRecorderStatus());

    std::unique_ptr<RTDeviceInfoT> dev_oq3 = std::make_unique<RTDeviceInfoT>("braket.aws.qubit");
    CHECK(*dev_oq3 == RTDeviceInfoT("librtd_openqasm" + get_dylib_ext(), "OpenQasmDevice"));
}

TEST_CASE("Test the device pool in ExecutionContext (single-thread)", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    auto dev0 = driver->addDevice("lightning.qubit");
    auto dev1 = driver->addDevice("lightning.qubit");
    auto dev2 = driver->addDevice("lightning.qubit");
    auto dev3 = driver->addDevice("lightning.qubit");

    CHECK(dev0.first == 0);
    CHECK(dev1.first == 1);
    CHECK(dev2.first == 2);
    CHECK(dev3.first == 3);

    CHECK(driver->getDevice(dev0.first) != nullptr);
    CHECK(driver->getDevice(dev3.first) != nullptr);
    CHECK(driver->getDevice(50) == nullptr);

    driver->removeDevice(dev1.first);
    driver->removeDevice(dev3.first);

    CHECK(driver->getDevice(dev1.first) == nullptr);
    CHECK(driver->getDevice(dev3.first) == nullptr);
}

void addDeviceToDriver(std::unique_ptr<ExecutionContext> &driver, std::string name)
{
    auto result = driver->addDevice(std::move(name));
    CHECK(result.first < driver->getPoolSize());
}

void getDeviceToDriver(std::unique_ptr<ExecutionContext> &driver, size_t key)
{
    auto result = driver->getDevice(key);
    CHECK(result != nullptr);
}

void removeDeviceToDriver(std::unique_ptr<ExecutionContext> &driver, size_t key)
{
    driver->removeDevice(key);
}

TEST_CASE("Test the device pool in ExecutionContext (multi-thread)", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    std::thread thr1(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr2(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr3(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr4(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr5(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr6(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    thr1.join();
    thr2.join();
    thr3.join();
    thr4.join();
    thr5.join();
    thr6.join();

    CHECK(driver->getPoolSize() == 6);

    std::thread thr7(removeDeviceToDriver, std::ref(driver), 0);
    std::thread thr8(removeDeviceToDriver, std::ref(driver), 1);
    std::thread thr9(removeDeviceToDriver, std::ref(driver), 2);
    thr7.join();
    thr8.join();
    thr9.join();

    CHECK(driver->getPoolSize() == 3);

    std::thread thr10(getDeviceToDriver, std::ref(driver), 3);
    std::thread thr11(getDeviceToDriver, std::ref(driver), 4);
    std::thread thr12(getDeviceToDriver, std::ref(driver), 5);
    thr10.join();
    thr11.join();
    thr12.join();

    std::thread thr13(removeDeviceToDriver, std::ref(driver), 3);
    std::thread thr14(removeDeviceToDriver, std::ref(driver), 4);
    std::thread thr15(removeDeviceToDriver, std::ref(driver), 5);
    thr13.join();
    thr14.join();
    thr15.join();

    CHECK(driver->getPoolSize() == 0);
}

TEST_CASE("Test QuantumDeviceInterface (single-thread)", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    auto dev0 = driver->addDevice("lightning.qubit");
    auto dev1 = driver->addDevice("lightning.qubit");

    auto circuit1 = [](QuantumDevice *device, double param) {
        std::vector<QubitIdType> Qs = QuantumDeviceInterface::AllocateQubits(device, 2);
        QuantumDeviceInterface::NamedOperation(device, "Hadamard", {}, {Qs[0]}, false);
        QuantumDeviceInterface::NamedOperation(device, "CRX", {param}, {Qs[0], Qs[1]}, false);
        ObsIdType pz = QuantumDeviceInterface::Observable(device, ObsId::PauliZ, {}, {Qs[1]});
        auto result = QuantumDeviceInterface::Expval(device, pz);
        CHECK(result == Approx(0.9900332889).margin(1e-5));
    };

    auto circuit2 = [](QuantumDevice *device, double param) {
        std::vector<QubitIdType> Qs = QuantumDeviceInterface::AllocateQubits(device, 2);
        QuantumDeviceInterface::NamedOperation(device, "Hadamard", {}, {Qs[0]}, false);
        QuantumDeviceInterface::NamedOperation(device, "CRX", {param}, {Qs[0], Qs[1]}, false);
        ObsIdType pz = QuantumDeviceInterface::Observable(device, ObsId::PauliZ, {}, {Qs[1]});
        return std::make_pair(QuantumDeviceInterface::Expval(device, pz),
                              QuantumDeviceInterface::Var(device, pz));
    };

    auto dev0_qdp = dev0.second;
    auto dev1_qdp = driver->getDevice(dev1.first);

    circuit1(dev0_qdp.get(), 0.2);
    circuit1(dev1_qdp.get(), 0.2);

    auto &&[expval0, var0] = circuit2(dev0_qdp.get(), 0.1);
    auto &&[expval1, var1] = circuit2(dev1_qdp.get(), 0.1);

    CHECK(expval0 == Approx(expval1).margin(1e-5));
    CHECK(var0 == Approx(var1).margin(1e-5));
}

TEST_CASE("Test QuantumDeviceInterface (multi-thread)", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    std::thread thr1(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr2(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    thr1.join();
    thr2.join();

    auto circuit1 = [](QuantumDevice *device, double param) {
        std::vector<QubitIdType> Qs = QuantumDeviceInterface::AllocateQubits(device, 2);
        QuantumDeviceInterface::NamedOperation(device, "Hadamard", {}, {Qs[0]}, false);
        QuantumDeviceInterface::NamedOperation(device, "CRX", {param}, {Qs[0], Qs[1]}, false);
        ObsIdType pz = QuantumDeviceInterface::Observable(device, ObsId::PauliZ, {}, {Qs[1]});
        auto result = QuantumDeviceInterface::Expval(device, pz);
        CHECK(result == Approx(0.9900332889).margin(1e-5));
    };

    auto dev0_qdp = driver->getDevice(0);
    auto dev1_qdp = driver->getDevice(1);

    std::jthread thr3(circuit1, dev0_qdp.get(), 0.2);
    std::jthread thr4(circuit1, dev1_qdp.get(), 0.2);
}

TEST_CASE("Test QuantumDeviceInterface with std::async (multi-thread)", "device_pool")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    std::thread thr1(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    std::thread thr2(addDeviceToDriver, std::ref(driver), std::string("lightning.qubit"));
    thr1.join();
    thr2.join();

    auto circuit2 = [](QuantumDevice *device, double param) {
        std::vector<QubitIdType> Qs = QuantumDeviceInterface::AllocateQubits(device, 2);
        QuantumDeviceInterface::NamedOperation(device, "Hadamard", {}, {Qs[0]}, false);
        QuantumDeviceInterface::NamedOperation(device, "CRX", {param}, {Qs[0], Qs[1]}, false);
        ObsIdType pz = QuantumDeviceInterface::Observable(device, ObsId::PauliZ, {}, {Qs[1]});
        return std::make_pair(QuantumDeviceInterface::Expval(device, pz),
                              QuantumDeviceInterface::Var(device, pz));
    };

    auto dev0_qdp = driver->getDevice(0);
    auto dev1_qdp = driver->getDevice(1);

    std::future<std::pair<double, double>> future0 =
        std::async(std::launch::async, circuit2, dev0_qdp.get(), 0.1);
    std::future<std::pair<double, double>> future1 =
        std::async(std::launch::async, circuit2, dev1_qdp.get(), 0.1);

    auto &&[expval0, var0] = future0.get();
    auto &&[expval1, var1] = future1.get();

    CHECK(expval0 == Approx(expval1).margin(1e-5));
    CHECK(var0 == Approx(var1).margin(1e-5));
}
