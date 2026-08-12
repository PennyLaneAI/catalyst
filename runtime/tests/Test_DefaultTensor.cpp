// Copyright 2026 Xanadu Quantum Technologies Inc.

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
#include <complex>
#include <cstdint>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "DataView.hpp"
#include "DefaultTensor.hpp"
#include "QuantumDevice.hpp"
#include "Types.h"

using namespace Catch::Matchers;
using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Devices;

namespace {

/// Probabilities over all currently allocated qubits.
auto allProbs(DefaultTensor &sim, size_t num_qubits) -> std::vector<double>
{
    std::vector<double> probs(size_t{1} << num_qubits, 0.0);
    DataView<double, 1> view(probs);
    sim.Probs(view);
    return probs;
}

} // namespace

TEST_CASE("DefaultTensor: Bell state probabilities", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    REQUIRE(wires.size() == 2);
    REQUIRE(sim.GetNumQubits() == 2);

    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[0], WithinAbs(0.5, 1e-12));
    CHECK_THAT(probs[1], WithinAbs(0.0, 1e-12));
    CHECK_THAT(probs[2], WithinAbs(0.0, 1e-12));
    CHECK_THAT(probs[3], WithinAbs(0.5, 1e-12));
}

TEST_CASE("DefaultTensor: wire 0 is the most significant bit", "[DefaultTensor]")
{
    // A device that inverts this passes every self-consistency check while
    // silently disagreeing with every other PennyLane device.
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("PauliX", {}, {wires[0]}, false, {}, {}, {});
    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[2], WithinAbs(1.0, 1e-12)); // |10>
}

TEST_CASE("DefaultTensor: expectation values and variance", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    const double theta = 0.7;
    sim.NamedOperation("RY", {theta}, {wires[0]}, false, {}, {}, {});

    auto z = sim.Observable(PauliZ, {}, {wires[0]});
    CHECK_THAT(sim.Expval(z), WithinAbs(std::cos(theta), 1e-12));
    CHECK_THAT(sim.Var(z), WithinAbs(1.0 - std::cos(theta) * std::cos(theta), 1e-12));
}

TEST_CASE("DefaultTensor: tensor and Hamiltonian observables", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(3);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    auto z0 = sim.Observable(PauliZ, {}, {wires[0]});
    auto z1 = sim.Observable(PauliZ, {}, {wires[1]});
    auto z2 = sim.Observable(PauliZ, {}, {wires[2]});

    CHECK_THAT(sim.Expval(sim.TensorObservable({z0, z1})), WithinAbs(1.0, 1e-12));
    // Summands on disjoint wires must not interfere.
    CHECK_THAT(sim.Expval(sim.HamiltonianObservable({0.5, 0.25}, {z0, z2})),
               WithinAbs(0.25, 1e-12));
}

TEST_CASE("DefaultTensor: adjoint round trip", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    sim.NamedOperation("RX", {0.9}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("RX", {0.9}, {wires[0]}, /*inverse=*/true, {}, {}, {});
    auto z = sim.Observable(PauliZ, {}, {wires[0]});
    CHECK_THAT(sim.Expval(z), WithinAbs(1.0, 1e-12));
}

TEST_CASE("DefaultTensor: multi-controlled gates", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(3);
    sim.NamedOperation("PauliX", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("PauliX", {}, {wires[1]}, false, {}, {}, {});
    // Toffoli expressed as X on wire 2 with two control wires.
    sim.NamedOperation("PauliX", {}, {wires[2]}, false, {wires[0], wires[1]}, {true, true}, {});
    auto probs = allProbs(sim, 3);
    CHECK_THAT(probs[7], WithinAbs(1.0, 1e-12));
}

TEST_CASE("DefaultTensor: dynamic qubit allocation mid-circuit", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto first = sim.AllocateQubits(1);
    sim.NamedOperation("Hadamard", {}, {first[0]}, false, {}, {}, {});

    // Allocate a second qubit *after* gates have already been applied. This is
    // the path Catalyst's automatic qubit management drives.
    const QubitIdType second = sim.AllocateQubit();
    REQUIRE(sim.GetNumQubits() == 2);
    sim.NamedOperation("CNOT", {}, {first[0], second}, false, {}, {}, {});

    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[0], WithinAbs(0.5, 1e-12));
    CHECK_THAT(probs[3], WithinAbs(0.5, 1e-12));
}

TEST_CASE("DefaultTensor: releasing a qubit keeps the state normalised", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    // Freeing an entangled qubit behaves like an unread measurement: the
    // survivor collapses to a definite branch, and probabilities still sum to 1.
    sim.ReleaseQubit(wires[1]);
    REQUIRE(sim.GetNumQubits() == 1);

    auto probs = allProbs(sim, 1);
    CHECK_THAT(probs[0] + probs[1], WithinAbs(1.0, 1e-12));
    const bool collapsed = (probs[0] > 0.999) || (probs[1] > 0.999);
    CHECK(collapsed);
}

TEST_CASE("DefaultTensor: qubit IDs are never reused", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    // Keep one qubit alive so this is not treated as a full teardown.
    const QubitIdType keep = sim.AllocateQubit();
    const QubitIdType scratch = sim.AllocateQubit();
    sim.ReleaseQubit(scratch);
    const QubitIdType fresh = sim.AllocateQubit();

    // Reusing IDs breaks Catalyst's automatic qubit management, which would then
    // hand the same ID to two live wires.
    CHECK(fresh != scratch);
    CHECK(fresh != keep);
    CHECK(sim.GetNumQubits() == 2);
}

TEST_CASE("DefaultTensor: operating on a released qubit raises", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.ReleaseQubit(wires[1]);
    REQUIRE_THROWS_WITH(
        sim.NamedOperation("Hadamard", {}, {wires[1]}, false, {}, {}, {}),
        ContainsSubstring("unallocated or released qubit"));
}

TEST_CASE("DefaultTensor: bulk teardown does not contract", "[DefaultTensor]")
{
    // Regression test: releasing every live qubit used to collapse them one at a
    // time, each requiring a full contraction, making program teardown cost as
    // much as the whole simulation. With a 2^20 intermediate cap this 30-qubit
    // GHZ chain only completes if teardown is free.
    DefaultTensor sim{"{'max_intermediate_log2': 20}"};
    constexpr size_t n = 30;
    auto wires = sim.AllocateQubits(n);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    for (size_t i = 0; i + 1 < n; i++) {
        sim.NamedOperation("CNOT", {}, {wires[i], wires[i + 1]}, false, {}, {}, {});
    }

    auto z0 = sim.Observable(PauliZ, {}, {wires[0]});
    auto zl = sim.Observable(PauliZ, {}, {wires[n - 1]});
    CHECK_THAT(sim.Expval(sim.TensorObservable({z0, zl})), WithinAbs(1.0, 1e-10));

    REQUIRE_NOTHROW(sim.ReleaseQubits(wires));
    CHECK(sim.GetNumQubits() == 0);
}

TEST_CASE("DefaultTensor: partial probabilities avoid the full state", "[DefaultTensor]")
{
    // A marginal is the diagonal of a reduced density matrix, so its cost scales
    // with the number of requested wires, not the register width.
    DefaultTensor sim{"{'max_intermediate_log2': 20}"};
    constexpr size_t n = 30;
    auto wires = sim.AllocateQubits(n);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    for (size_t i = 0; i + 1 < n; i++) {
        sim.NamedOperation("CNOT", {}, {wires[i], wires[i + 1]}, false, {}, {}, {});
    }

    std::vector<double> probs(4, 0.0);
    DataView<double, 1> view(probs);
    sim.PartialProbs(view, {wires[0], wires[n - 1]});
    CHECK_THAT(probs[0], WithinAbs(0.5, 1e-10));
    CHECK_THAT(probs[1], WithinAbs(0.0, 1e-10));
    CHECK_THAT(probs[2], WithinAbs(0.0, 1e-10));
    CHECK_THAT(probs[3], WithinAbs(0.5, 1e-10));
}

TEST_CASE("DefaultTensor: mid-circuit measurement collapses the state", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    Result r = sim.Measure(wires[0], 1); // postselect |1>
    CHECK(*r == true);
    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[3], WithinAbs(1.0, 1e-12)); // collapsed to |11>
}

TEST_CASE("DefaultTensor: shots, sampling and counts", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.SetDeviceShots(1000);
    REQUIRE(sim.GetDeviceShots() == 1000);

    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    std::vector<double> eigvals(4, 0.0);
    std::vector<int64_t> counts(4, 0);
    DataView<double, 1> ev(eigvals);
    DataView<int64_t, 1> cv(counts);
    sim.Counts(ev, cv);

    CHECK(counts[0] + counts[1] + counts[2] + counts[3] == 1000);
    // Only |00> and |11> can occur on a Bell state.
    CHECK(counts[1] == 0);
    CHECK(counts[2] == 0);
}

TEST_CASE("DefaultTensor: unsupported gates raise", "[DefaultTensor]")
{
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    // SWAP/S/T are deliberately absent from the bare gate set; Catalyst is
    // expected to decompose them before they reach the device.
    REQUIRE_THROWS_WITH(sim.NamedOperation("PauliRot", {0.5}, {wires[0]}, false, {}, {}, {}),
                        ContainsSubstring("unsupported gate"));
}

TEST_CASE("DefaultTensor: oversized contraction raises instead of exhausting RAM",
          "[DefaultTensor]")
{
    DefaultTensor sim{"{'max_intermediate_log2': 8}"};
    constexpr size_t n = 20;
    auto wires = sim.AllocateQubits(n);
    for (size_t i = 0; i < n; i++) {
        sim.NamedOperation("Hadamard", {}, {wires[i]}, false, {}, {}, {});
    }

    std::vector<double> probs(size_t{1} << n, 0.0);
    DataView<double, 1> view(probs);
    REQUIRE_THROWS_WITH(sim.Probs(view), ContainsSubstring("exceeding the configured limit"));
}

TEST_CASE("DefaultTensor: device kwargs are parsed", "[DefaultTensor]")
{
    // An out-of-range guard must be rejected at construction.
    REQUIRE_THROWS_WITH(DefaultTensor{"{'max_intermediate_log2': 99}"},
                        ContainsSubstring("max_intermediate_log2"));
}
