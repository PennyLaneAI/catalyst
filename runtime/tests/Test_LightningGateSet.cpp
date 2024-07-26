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

#include <cmath>

#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "Utils.hpp"

#include "TestUtils.hpp"

using namespace Pennylane;

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEMPLATE_LIST_TEST_CASE("Identity Gate tests", "[GateSet]", SimTypes)
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

    CHECK(state.at(0) == std::complex<double>{1, 0});

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < state.size(); i++) {
        sum += state[i];
    }

    CHECK(sum == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("PauliX Gate tests num_qubits=1", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 1;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{1, 0});
}

// 1-qubit operations
TEMPLATE_LIST_TEST_CASE("PauliX Gate tests num_qubits=3", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 3;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliX", {}, {Qs[1]}, false);
    sim->NamedOperation("PauliX", {}, {Qs[2]}, false);

    sim->NamedOperation("PauliX", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliX", {}, {Qs[1]}, false);
    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{1, 0});

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < state.size(); i++) {
        sum += state[i];
    }

    CHECK(sum == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("PauliY Gate tests num_qubits=1", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 1;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliY", {}, {Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 1});
}

TEMPLATE_LIST_TEST_CASE("PauliY Gate tests num_qubits=2", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliY", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{0, 0});
    CHECK(state.at(3) == std::complex<double>{-1, 0});
}

TEMPLATE_LIST_TEST_CASE("PauliZ Gate tests num_qubits=2", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliY", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{0, 1});
    CHECK(state.at(3) == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("Hadamard Gate tests num_qubits=2", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.5).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));
    CHECK(state.at(1) == state.at(0));
    CHECK(state.at(2) == state.at(0));
    CHECK(state.at(3) == state.at(0));
}

TEMPLATE_LIST_TEST_CASE("Hadamard Gate tests num_qubits=3", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.5).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));
    CHECK(state.at(1) == state.at(0));
    CHECK(state.at(2) == state.at(0));
    CHECK(state.at(3) == state.at(0));
}

TEMPLATE_LIST_TEST_CASE("MIX Gate test R(X,Y,Z) num_qubits=1,4", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubit = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);

    sim->NamedOperation("RX", {0.123}, {Qs[1]}, false);
    sim->NamedOperation("RY", {0.456}, {Qs[2]}, false);
    sim->NamedOperation("RZ", {0.789}, {Qs[3]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    // calculated by pennylane,
    CHECK(state.at(0) == std::complex<double>{0, 0});

    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{0, 0});
    CHECK(state.at(3) == std::complex<double>{0, 0});

    CHECK(state.at(4) == std::complex<double>{0, 0});
    CHECK(state.at(5) == std::complex<double>{0, 0});
    CHECK(state.at(6) == std::complex<double>{0, 0});
    CHECK(state.at(7) == std::complex<double>{0, 0});

    CHECK(state[8].real() == Approx(0.8975969498074641).epsilon(1e-5));
    CHECK(state[8].imag() == Approx(-0.3736920921192206).epsilon(1e-5));

    CHECK(state.at(9) == std::complex<double>{0, 0});
    CHECK(state[10].real() == Approx(0.20827363966052723).epsilon(1e-5));
    CHECK(state[10].imag() == Approx(-0.08670953277495183).epsilon(1e-5));

    CHECK(state.at(11) == std::complex<double>{0, 0});

    CHECK(state[12].real() == Approx(-0.023011082205037697).epsilon(1e-5));
    CHECK(state[12].imag() == Approx(-0.055271914055973925).epsilon(1e-5));

    CHECK(state.at(13) == std::complex<double>{0, 0});
    CHECK(state[14].real() == Approx(-0.005339369573836912).epsilon(1e-5));
    CHECK(state[14].imag() == Approx(-0.012825002038956146).epsilon(1e-5));
    CHECK(state.at(15) == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("test PhaseShift num_qubits=2", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubit = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    Qs[0] = sim->AllocateQubit();
    Qs[1] = sim->AllocateQubit();

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("RX", {0.123}, {Qs[1]}, false);
    sim->NamedOperation("PhaseShift", {0.456}, {Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    // calculated by pennylane,
    CHECK(state[0].real() == Approx(0.7057699753).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));

    CHECK(state[1].real() == Approx(0).epsilon(1e-5));
    CHECK(state[1].imag() == Approx(-0.04345966).epsilon(1e-5));

    CHECK(state[2].real() == Approx(0.63365519).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0.31079312).epsilon(1e-5));

    CHECK(state[3].real() == Approx(0.01913791).epsilon(1e-5));
    CHECK(state[3].imag() == Approx(-0.039019).epsilon(1e-5));
}

// 2-qubit operations
TEMPLATE_LIST_TEST_CASE("CNOT Gate tests num_qubits=2 [0,1]", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{0, 0});
    CHECK(state.at(3) == std::complex<double>{1, 0});
}

TEMPLATE_LIST_TEST_CASE("CNOT Gate tests num_qubits=2 [1,0]", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[1], Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{1, 0});
    CHECK(state.at(3) == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("MIX Gate test CR(X, Y, Z) num_qubits=1,4", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubit = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("CRX", {0.123}, {Qs[0], Qs[1]}, false);
    sim->NamedOperation("CRY", {0.456}, {Qs[0], Qs[2]}, false);
    sim->NamedOperation("CRZ", {0.789}, {Qs[0], Qs[3]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    // calculated by pennylane,
    CHECK(state[0].real() == Approx(0.7071067811865475).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state.at(2) == std::complex<double>{0, 0});
    CHECK(state.at(3) == std::complex<double>{0, 0});

    CHECK(state.at(4) == std::complex<double>{0, 0});
    CHECK(state.at(5) == std::complex<double>{0, 0});
    CHECK(state.at(6) == std::complex<double>{0, 0});
    CHECK(state.at(7) == std::complex<double>{0, 0});

    CHECK(state[8].real() == Approx(0.6346968899812189).epsilon(1e-5));
    CHECK(state[8].imag() == Approx(-0.2642402124132889).epsilon(1e-5));

    CHECK(state.at(9) == std::complex<double>{0, 0});
    CHECK(state[10].real() == Approx(0.14727170294636227).epsilon(1e-5));
    CHECK(state[10].imag() == Approx(-0.061312898618685635).epsilon(1e-5));
    CHECK(state.at(11) == std::complex<double>{0, 0});

    CHECK(state[12].real() == Approx(-0.016271292269623247).epsilon(1e-5));
    CHECK(state[12].imag() == Approx(-0.03908314523813921).epsilon(1e-5));
    CHECK(state.at(13) == std::complex<double>{0, 0});
    CHECK(state[14].real() == Approx(-0.0037755044329212074).epsilon(1e-5));
    CHECK(state[14].imag() == Approx(-0.009068645910477189).epsilon(1e-5));
    CHECK(state.at(15) == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("CRot", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("CRot", {M_PI, M_PI_2, 0.5}, {Qs[0], Qs[1]}, false);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.7071067812).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));
    CHECK(state[1].real() == Approx(0).epsilon(1e-5));
    CHECK(state[1].imag() == Approx(0).epsilon(1e-5));
    CHECK(state[2].real() == Approx(-0.1237019796).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(-0.4844562109).epsilon(1e-5));
    CHECK(state[3].real() == Approx(0.1237019796).epsilon(1e-5));
    CHECK(state[3].imag() == Approx(-0.4844562109).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("CSWAP test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 3;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("RX", {M_PI}, {Qs[0]}, false);
    sim->NamedOperation("RX", {M_PI}, {Qs[1]}, false);
    sim->NamedOperation("CSWAP", {}, {Qs[0], Qs[1], Qs[2]}, false);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[5].real() == Approx(-1).epsilon(1e-5));
    CHECK(state[5].imag() == Approx(0).epsilon(1e-5));
}

// TODO: Uncomment these tests after `PSWAP` and `ISWAP` are natively supported by Lightning
// simulators.
/*
TEMPLATE_LIST_TEST_CASE("ISWAP test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("ISWAP", {}, {Qs[0], Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 1});
    CHECK(state.at(2) == std::complex<double>{0, 0});
    CHECK(state.at(3) == std::complex<double>{0, 0});
}

TEMPLATE_LIST_TEST_CASE("PSWAP test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("PSWAP", {M_PI_2}, {Qs[0], Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.5).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.5).epsilon(1e-5));
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state[2].real() == Approx(0.5).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(-0.5).epsilon(1e-5));
    CHECK(state.at(3) == std::complex<double>{0, 0});
}
*/

TEMPLATE_LIST_TEST_CASE("IsingXY Gate tests num_qubits=2 [1,0]", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[0]}, false);
    sim->NamedOperation("IsingXY", {0.2}, {Qs[1], Qs[0]}, false);
    sim->NamedOperation("SWAP", {}, {Qs[0], Qs[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.70710678).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0).epsilon(1e-5));
    CHECK(state[1].real() == Approx(-0.70357419).epsilon(1e-5));
    CHECK(state[1].imag() == Approx(0).epsilon(1e-5));
    CHECK(state[2].real() == Approx(0).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(-0.07059289).epsilon(1e-5));
    CHECK(state[3].real() == Approx(0).epsilon(1e-5));
    CHECK(state[3].imag() == Approx(0).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("Toffoli test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 3;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliX", {}, {Qs[1]}, false);
    sim->NamedOperation("Toffoli", {}, {Qs[0], Qs[1], Qs[2]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0) == std::complex<double>{0, 0});
    CHECK(state.at(1) == std::complex<double>{0, 0});
    CHECK(state[2].real() == Approx(0.70710678).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0).epsilon(1e-5));
    CHECK(state.at(3) == std::complex<double>{0, 0});
    CHECK(state.at(4) == std::complex<double>{0, 0});
    CHECK(state.at(5) == std::complex<double>{0, 0});
    CHECK(state.at(6) == std::complex<double>{0, 0});
    CHECK(state[7].real() == Approx(0.70710678).epsilon(1e-5));
    CHECK(state[7].imag() == Approx(0).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("MultiRZ test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("RX", {M_PI}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("MultiRZ", {M_PI}, {Qs[0], Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[2].real() == Approx(-1).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("MatrixOperation test with 2-qubit", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    const std::vector<QubitIdType> wires = {Qs[0]};
    std::vector<std::complex<double>> matrix{
        {-0.6709485262524046, -0.6304426335363695},
        {-0.14885403153998722, 0.3608498832392019},
        {-0.2376311670004963, 0.3096798175687841},
        {-0.8818365947322423, -0.26456390390903695},
    };
    sim->MatrixOperation(matrix, wires, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(-0.474432).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(-0.44579).epsilon(1e-5));

    CHECK(state[1].real() == Approx(-0.105256).epsilon(1e-5));
    CHECK(state[1].imag() == Approx(0.255159).epsilon(1e-5));

    CHECK(state[2].real() == Approx(-0.168031).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0.218977).epsilon(1e-5));

    CHECK(state[3].real() == Approx(-0.623553).epsilon(1e-5));
    CHECK(state[3].imag() == Approx(-0.187075).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("MatrixOperation test with 3-qubit", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 3;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->StartTapeRecording();
    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("Toffoli", {}, {Qs[0], Qs[1], Qs[2]}, false);

    const std::vector<QubitIdType> wires = {Qs[0], Qs[1]};
    std::vector<std::complex<double>> matrix{
        {0.203377341216, 0.132238554262}, {0.216290940442, 0.203109511967},
        {0.290374372568, 0.123095338906}, {0.040762810130, 0.153237600777},
        {0.062445212079, 0.106020046388}, {0.041489260594, 0.149813636657},
        {0.002100244854, 0.099744848045}, {0.281559630427, 0.083376695381},
        {0.073652349575, 0.066811372960}, {0.150797357980, 0.146266222503},
        {0.324043781913, 0.157417591307}, {0.040556496061, 0.254572386140},
        {0.204954964152, 0.098550445557}, {0.056681743348, 0.225803880189},
        {0.327486634260, 0.130699704247}, {0.299805387808, 0.150417378569},
    };
    sim->MatrixOperation(matrix, wires, false);
    sim->StopTapeRecording();

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(0.349135).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.180548).epsilon(1e-5));
    CHECK(state[2].real() == Approx(0.0456405).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0.145498).epsilon(1e-5));
    CHECK(state[4].real() == Approx(0.281214).epsilon(1e-5));
    CHECK(state[4].imag() == Approx(0.158554).epsilon(1e-5));
    CHECK(state[6].real() == Approx(0.376493).epsilon(1e-5));
    CHECK(state[6].imag() == Approx(0.162104).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("MatrixOperation test with 4-qubit", "[GateSet]", LightningSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubit = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("CRX", {0.123}, {Qs[0], Qs[1]}, false);
    sim->NamedOperation("CRY", {0.456}, {Qs[0], Qs[2]}, false);
    sim->NamedOperation("CRZ", {0.789}, {Qs[0], Qs[3]}, false);

    const std::vector<QubitIdType> wires = {Qs[0], Qs[1], Qs[2]};
    std::vector<std::complex<double>> matrix{
        {-0.14601911598243822, -0.18655250647340088},
        {-0.03917826201290317, -0.031161687050443518},
        {0.11497626236175404, 0.38310733543366354},
        {-0.0929691815340695, 0.1219804125497268},
        {0.07306514883467692, 0.017445444816725875},
        {-0.27330866098918355, -0.6007032759764033},
        {0.4530754397715841, -0.08267189625512258},
        {0.32125201986075, -0.036845158875036116},
        {0.032317572838307884, 0.02292755555300329},
        {-0.18775945295623664, -0.060215004737844156},
        {-0.3093351335745536, -0.2061961962889725},
        {0.4216087567144761, 0.010534488410902099},
        {0.2769943541718527, -0.26016137877135465},
        {0.18727884147867532, 0.02830415812286322},
        {0.3367562196770689, -0.5250999173939218},
        {0.05770014289220745, 0.26595514845958573},
        {0.37885720163317027, 0.3110931426403546},
        {0.13436510737129648, -0.4083415934958021},
        {-0.5443665467635203, 0.2458343977310266},
        {-0.050346912365833024, 0.08709833123617361},
        {0.11505259829552131, 0.010155858056939438},
        {-0.2930849061531229, 0.019339259194141145},
        {0.011825409829453282, 0.011597907736881019},
        {-0.10565527258356637, -0.3113689446440079},
        {0.0273191284561944, -0.2479498526173881},
        {-0.5528072425836249, -0.06114469689935285},
        {-0.20560364740746587, -0.3800208994544297},
        {-0.008236143958221483, 0.3017421511504845},
        {0.04817188123334976, 0.08550951191632741},
        {-0.24081054643565586, -0.3412671345149831},
        {-0.38913538197001885, 0.09288402897806938},
        {-0.07937578245883717, 0.013979426755633685},
        {0.22246583652015395, -0.18276674810033927},
        {0.22376666162382491, 0.2995723155125488},
        {-0.1727191441070097, -0.03880522034607489},
        {0.075780203819001, 0.2818783673816625},
        {-0.6161322400651016, 0.26067347179217193},
        {-0.021161519614267765, -0.08430919051054794},
        {0.1676500381348944, -0.30645601624407504},
        {-0.28858251997285883, 0.018089595494883842},
        {-0.19590767481842053, -0.12844366632033652},
        {0.18707834504831794, -0.1363932722670649},
        {-0.07224221779769334, -0.11267803536286894},
        {-0.23897684826459387, -0.39609971967853685},
        {-0.0032110880452929555, -0.29294331305690136},
        {-0.3188741682462722, -0.17338979346647143},
        {0.08194395032821632, -0.002944814673179825},
        {-0.5695791830944521, 0.33299548924055095},
        {-0.4983660307441444, -0.4222358493977972},
        {0.05533914327048402, -0.42575842134560576},
        {-0.2187623521182678, -0.03087596187054778},
        {0.11278255885846857, 0.07075886163492914},
        {-0.3054684775292515, -0.1739796870866232},
        {0.14151567663565712, 0.20399935744127418},
        {0.06720165377364941, 0.07543463072363207},
        {0.08019665306716581, -0.3473013434358584},
        {-0.2600167605995786, -0.08795704036197827},
        {0.125680477777759, 0.266342700305046},
        {-0.1586772594600269, 0.187360909108502},
        {-0.4653314704208982, 0.4048609954619629},
        {0.39992560380733094, -0.10029244177901954},
        {0.2533527906886461, 0.05222114898540775},
        {-0.15840033949128557, -0.2727320427534386},
        {-0.21590866323269536, -0.1191163626522938},
    };
    sim->MatrixOperation(matrix, wires, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(-0.141499).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(-0.230993).epsilon(1e-5));

    CHECK(state[2].real() == Approx(0.135423).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(-0.235563).epsilon(1e-5));

    CHECK(state[4].real() == Approx(0.299458).epsilon(1e-5));
    CHECK(state[4].imag() == Approx(0.218321).epsilon(1e-5));

    CHECK(state[6].real() == Approx(0.0264869).epsilon(1e-5));
    CHECK(state[6].imag() == Approx(-0.154913).epsilon(1e-5));

    CHECK(state[8].real() == Approx(-0.186607).epsilon(1e-5));
    CHECK(state[8].imag() == Approx(0.188884).epsilon(1e-5));

    CHECK(state[10].real() == Approx(-0.271843).epsilon(1e-5));
    CHECK(state[10].imag() == Approx(-0.281136).epsilon(1e-5));

    CHECK(state[12].real() == Approx(-0.560499).epsilon(1e-5));
    CHECK(state[12].imag() == Approx(-0.310176).epsilon(1e-5));

    CHECK(state[14].real() == Approx(0.0756372).epsilon(1e-5));
    CHECK(state[14].imag() == Approx(-0.226334).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("Controlled gates", "[GateSet]", SimTypes)
{
    const size_t N = 3;
    __catalyst__rt__initialize(nullptr);

    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        // TODO: remove when other devices support controlled gates
        if (rtd_name != "lightning.qubit")
            continue;

        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        std::vector<QUBIT *> Q;
        for (size_t i = 0; i < N; i++) {
            Q.push_back(__catalyst__rt__qubit_allocate());
        }

        __catalyst__rt__toggle_recorder(/* activate_cm */ true);

        /* qml.Hadamard(wires=0) */
        /* qml.Hadamard(wires=1) */
        /* qml.Hadamard(wires=2) */
        for (size_t i = 0; i < N; i++) {
            __catalyst__qis__Hadamard(Q[i], NO_MODIFIERS);
        }

        /* qml.ctrl(qml.PauliX, control=(1,2), control_values=(False, False))(wires=0) */
        {
            QUBIT *ctrls[] = {Q[1], Q[2]};
            bool values[] = {false, false};
            Modifiers mod = {false, 2, (QUBIT *)ctrls, (bool *)values};
            __catalyst__qis__PauliX(Q[0], &mod);
        }

        /* qml.ctrl(qml.PauliY, control=(0,1), control_values=(False, False))(wires=2) */
        {
            QUBIT *ctrls[] = {Q[0], Q[1]};
            bool values[] = {false, false};
            Modifiers mod = {false, 2, (QUBIT *)ctrls, (bool *)values};
            __catalyst__qis__PauliY(Q[2], &mod);
        }

        /* qml.ctrl(qml.PauliZ, control=(2,0), control_values=(False, False))(wires=1) */
        {
            QUBIT *ctrls[] = {Q[2], Q[0]};
            bool values[] = {false, false};
            Modifiers mod = {false, 2, (QUBIT *)ctrls, (bool *)values};
            __catalyst__qis__PauliZ(Q[1], &mod);
        }
        __catalyst__rt__toggle_recorder(/* activate_cm */ false);

        {
            MemRefT_CplxT_double_1d state = getState(1 << N);
            __catalyst__qis__State(&state, 0);
            CplxT_double *buffer = state.data_allocated;

            CHECK(buffer[0].real == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[0].imag == Approx(-0.353553).epsilon(1e-5));
            CHECK(buffer[1].real == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[1].imag == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[2].real == Approx(-0.353553).epsilon(1e-5));
            CHECK(buffer[2].imag == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[3].real == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[3].imag == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[4].real == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[4].imag == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[5].real == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[5].imag == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[6].real == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[6].imag == Approx(0.000000).epsilon(1e-5));
            CHECK(buffer[7].real == Approx(0.353553).epsilon(1e-5));
            CHECK(buffer[7].imag == Approx(0.000000).epsilon(1e-5));

            freeState(state);
        }

        __catalyst__rt__qubit_release_array(nullptr);
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEMPLATE_LIST_TEST_CASE("IsingZZ Gate Test", "[GateSet]", SimTypes)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> qubits;
    qubits.reserve(n);

    for (size_t i = 0; i < n; i++) {
        qubits.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {qubits[0]}, false);
    sim->NamedOperation("Hadamard", {}, {qubits[1]}, false);
    sim->NamedOperation("IsingZZ", {M_PI_4}, {qubits[0], qubits[1]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.at(0).real() == Approx(0.4619397663).epsilon(1e-5));
    CHECK(state.at(0).imag() == Approx(-0.1913417162).epsilon(1e-5));
    CHECK(state.at(1).real() == Approx(0.4619397663).epsilon(1e-5));
    CHECK(state.at(1).imag() == Approx(0.1913417162).epsilon(1e-5));
    CHECK(state.at(2).real() == Approx(0.4619397663).epsilon(1e-5));
    CHECK(state.at(2).imag() == Approx(0.1913417162).epsilon(1e-5));
    CHECK(state.at(3).real() == Approx(0.4619397663).epsilon(1e-5));
    CHECK(state.at(3).imag() == Approx(-0.1913417162).epsilon(1e-5));
}
