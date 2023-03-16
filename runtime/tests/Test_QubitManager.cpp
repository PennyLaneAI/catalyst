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

#include "RuntimeCAPI.h"

#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEST_CASE("Simple allocation and release of one qubit", "[QubitManager]")
{
    QubitManager qm = QubitManager();
    QubitIdType idx = qm.Allocate(0);
    CHECK(qm.isValidQubitId(idx));

    qm.Release(idx);
    CHECK(!qm.isValidQubitId(idx));
}

TEST_CASE("Allocation and reallocation of one qubit multiple times", "[QubitManager]")
{
    QubitManager qm = QubitManager();

    QubitIdType q = qm.Allocate(0);
    CHECK(q == 0);
    qm.Release(q);
    QubitIdType q0 = qm.Allocate(0);
    CHECK(q0 == 1);
    CHECK(qm.getDeviceId(q0) == 0);
    qm.Release(q0);
}

TEST_CASE("Allocation and reallocation of two qubit", "[QubitManager]")
{
    QubitManager qm = QubitManager();
    QubitIdType idx0 = qm.Allocate(0);
    QubitIdType idx1 = qm.Allocate(1);

    qm.Release(idx0);

    CHECK((!qm.isValidQubitId(idx0) && qm.isValidQubitId(idx1)));

    CHECK(qm.getDeviceId(idx1) == 0);

    REQUIRE_THROWS_WITH(qm.getDeviceId(idx0), Catch::Contains("Invalid device qubit"));
}

TEST_CASE("multiple release of qubits", "[QubitManager]")
{
    QubitManager qm = QubitManager();

    QubitIdType idx0 = qm.Allocate(0);
    QubitIdType idx1 = qm.Allocate(1);
    QubitIdType idx2 = qm.Allocate(2);
    QubitIdType idx3 = qm.Allocate(3);

    qm.Release(idx2);

    CHECK(idx3 == 3);
    CHECK(!qm.isValidQubitId(idx2));
    CHECK(qm.getDeviceId(idx3) == 2);

    qm.Release(idx0);

    CHECK(!qm.isValidQubitId(idx0));
    CHECK(qm.getDeviceId(idx3) == 1);
    CHECK(qm.getDeviceId(idx1) == 0);

    QubitIdType idx4 = qm.Allocate(2);
    QubitIdType idx5 = qm.Allocate(3);
    QubitIdType idx6 = qm.Allocate(4);

    qm.Release(idx5);

    CHECK(idx4 == 4);
    CHECK(!qm.isValidQubitId(idx5));
    CHECK(qm.isValidQubitId(idx4));
    CHECK(qm.getDeviceId(idx4) == 2);
    CHECK(qm.getDeviceId(idx6) == 3);
}

TEST_CASE("Test isFreeQubitId for a vector of wires", "[QubitManager]")
{
    QubitManager qm = QubitManager();

    QubitIdType idx0 = qm.Allocate(0);
    QubitIdType idx1 = qm.Allocate(1);
    QubitIdType idx2 = qm.Allocate(2);
    QubitIdType idx3 = qm.Allocate(3);

    qm.Release(idx0);
    qm.Release(idx2);

    QubitIdType idx4 = qm.Allocate(2);
    QubitIdType idx5 = qm.Allocate(3);
    QubitIdType idx6 = qm.Allocate(4);

    qm.Release(idx5);
    qm.Release(idx3);

    CHECK(qm.getDeviceId(idx1) == 0);
    CHECK(qm.getDeviceId(idx4) == 1);
    CHECK(qm.getDeviceId(idx6) == 2);

    CHECK(!qm.isValidQubitId({idx0, idx1, idx2}));
    CHECK(qm.isValidQubitId({idx1, idx1, idx4}));
    CHECK(!qm.isValidQubitId({idx0, idx5, idx6, idx2}));
}

TEST_CASE("Test getSimulatorId for a vector of wires", "[QubitManager]")
{
    QubitManager qm = QubitManager();

    QubitIdType idx0 = qm.Allocate(0);
    QubitIdType idx1 = qm.Allocate(1);
    QubitIdType idx2 = qm.Allocate(2);
    QubitIdType idx3 = qm.Allocate(3);

    qm.Release(idx0);
    qm.Release(idx2);

    QubitIdType idx4 = qm.Allocate(2);
    QubitIdType idx5 = qm.Allocate(3);
    QubitIdType idx6 = qm.Allocate(4);

    qm.Release(idx5);
    qm.Release(idx3);

    CHECK(qm.getSimulatorId(0) == idx1);
    CHECK(qm.getSimulatorId(1) == idx4);
    CHECK(qm.getSimulatorId(2) == idx6);

    REQUIRE_THROWS_WITH(qm.getSimulatorId(3), Catch::Contains("Invalid simulator qubit"));
}
