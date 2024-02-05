
// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <numeric>
#include <string>

#include "ExecutionContext.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "Utils.hpp"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEST_CASE("Test parse_kwargs coverage", "[Utils]")
{
    std::string case1{""};
    CHECK(parse_kwargs(case1).empty());

    std::string case2{"{shots : 1000}"};
    std::string case3{"shots : 1000"};
    std::string case4{"'shots':'1000'"};
    CHECK(parse_kwargs(case2) == parse_kwargs(case3));
    CHECK(parse_kwargs(case3) == parse_kwargs(case4));

    std::string case5{"{'A':'B', 'C':'D', 'E':'F'}"};
    auto res5 = parse_kwargs(case5);
    CHECK(res5.size() == 3);
    CHECK((res5.contains("A") && res5["A"] == "B"));
    CHECK((res5.contains("C") && res5["C"] == "D"));
    CHECK((res5.contains("E") && res5["E"] == "F"));

    std::string case6{"device_type : braket.aws.qubit,{'shots': 0, 'device_arn': 'sv1', "
                      "'s3_destination_folder': \"('catalyst-op3-s3', 'prefix')\"}"};
    auto res6 = parse_kwargs(case6);
    CHECK(res6.size() == 4);
    CHECK((res6.contains("device_type") && res6["device_type"] == "braket.aws.qubit"));
    CHECK((res6.contains("shots") && res6["shots"] == "0"));
    CHECK((res6.contains("device_arn") && res6["device_arn"] == "sv1"));
    CHECK((res6.contains("s3_destination_folder") &&
           res6["s3_destination_folder"] == "('catalyst-op3-s3', 'prefix')"));
}

TEST_CASE("Test Driver", "[Driver]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();

    // check the scope of memory-manager
    CHECK(driver->getMemoryManager() != nullptr);

    // check device default specs
    CHECK(driver->getDeviceRecorderStatus() == false);

    // check device specs update
    driver->setDeviceRecorderStatus(true);
    CHECK(driver->getDeviceRecorderStatus() == true);
}

TEMPLATE_LIST_TEST_CASE("lightning Basis vector", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    QubitIdType q = sim->AllocateQubit();
    q = sim->AllocateQubit();
    q = sim->AllocateQubit();

    sim->ReleaseQubit(q);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(view(0).real() == Approx(1.0).epsilon(1e-5));
    CHECK(view(0).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(1).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(1).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(2).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(2).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(3).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(3).imag() == Approx(0.0).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("Qubit allocatation and deallocation", "[Driver]", LightningSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    constexpr size_t n = 1;
    constexpr size_t sz = (1UL << n);

    QubitIdType q;
    for (size_t i = 0; i < n; i++) {
        q = sim->AllocateQubit();
    }

    CHECK(n == static_cast<size_t>(q) + 1);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.size() == (1UL << n));
    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < sz; i++) {
        sum += state[i];
    }

    CHECK(sum.real() == Approx(0.0).epsilon(1e-5));
    CHECK(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = n; i > 0; i--) {
        CHECK(state.size() == sz);

        sim->ReleaseQubit(i - 1);
        sim->AllocateQubit();

        DataView<std::complex<double>, 1> view(state);
        sim->State(view);
    }
}

TEMPLATE_LIST_TEST_CASE("test AllocateQubits", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    CHECK(sim->AllocateQubits(0).size() == 0);

    auto &&q = sim->AllocateQubits(2);

    sim->ReleaseQubit(q[0]);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("test multiple AllocateQubits", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    auto &&q1 = sim->AllocateQubits(2);
    CHECK(q1[0] == 0);
    CHECK(q1[1] == 1);

    auto &&q2 = sim->AllocateQubits(3);
    CHECK(q2.size() == 3);
    CHECK(q2[0] == 2);
    CHECK(q2[2] == 4);
}

TEMPLATE_LIST_TEST_CASE("test DeviceShots", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    CHECK(sim->GetDeviceShots() == 0);

    sim->SetDeviceShots(500);

    CHECK(sim->GetDeviceShots() == 500);
}

TEMPLATE_LIST_TEST_CASE("compute register tests", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    constexpr size_t n = 10;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    // allocate a few qubits
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    // release some of them
    sim->ReleaseQubit(n - 1);
    sim->ReleaseQubit(n - 2);

    const size_t new_n = n - 2;

    // check the correctness
    std::vector<QubitIdType> Qs_expected(new_n);
    std::iota(Qs_expected.begin(), Qs_expected.end(), static_cast<QubitIdType>(0));

    for (size_t i = 0; i < new_n; i++) {
        CHECK(Qs_expected[i] == Qs[i]);
    }
}

TEMPLATE_LIST_TEST_CASE("Check an unsupported operation", "[Driver]", SimTypes)
{
    REQUIRE_THROWS_WITH(
        Lightning::lookup_gates(Lightning::simulator_gate_info, "UnsupportedGateName"),
        Catch::Contains("The given operation is not supported by the simulator"));
}

TEMPLATE_TEST_CASE("QuantumDevice object test [lightning.qubit]", "[Driver]", LightningSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 10;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Identity", {}, {Qs[0]}, false);
    sim->NamedOperation("Identity", {}, {Qs[2]}, false);
    sim->NamedOperation("Identity", {}, {Qs[4]}, false);
    sim->NamedOperation("Identity", {}, {Qs[6]}, false);
    sim->NamedOperation("Identity", {}, {Qs[8]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < state.size(); i++) {
        sum += state[i];
    }

    CHECK(sum.real() == Approx(0.0).epsilon(1e-5));
    CHECK(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = 0; i < n; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 0, 1, 2, ..., 9
    }

    for (size_t i = 10; i < n + 10; i++) {
        CHECK(static_cast<QubitIdType>(i) == sim->AllocateQubit());
        // 10, 11, ..., 19
    }

    for (size_t i = 10; i < n + 10; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 10, 11, ..., 19
    }

    for (size_t i = 20; i < n + 20; i++) {
        CHECK(static_cast<QubitIdType>(i) == sim->AllocateQubit());
        // 20, 21, ..., 29
    }
}

TEMPLATE_LIST_TEST_CASE("Check re-AllocateQubit", "[Driver]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    sim->AllocateQubit();
    sim->NamedOperation("Hadamard", {}, {0}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);
    CHECK(state[0].real() == Approx(0.707107).epsilon(1e-5));
    CHECK(state[1].real() == Approx(0.707107).epsilon(1e-5));

    sim->AllocateQubit();
    sim->AllocateQubit();
    sim->AllocateQubit();

    state = std::vector<std::complex<double>>(1U << sim->GetNumQubits());
    view = DataView<std::complex<double>, 1>(state);
    sim->State(view);
    CHECK(state[0].real() == Approx(0.707107).epsilon(1e-5));
    CHECK(state[8].real() == Approx(0.707107).epsilon(1e-5));
}
