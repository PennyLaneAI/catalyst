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
auto allProbs(DefaultTensor &sim, size_t num_qubits) -> std::vector<double> {
    std::vector<double> probs(size_t{1} << num_qubits, 0.0);
    DataView<double, 1> view(probs);
    sim.Probs(view);
    return probs;
}

} // namespace

TEST_CASE("DefaultTensor: Bell state probabilities", "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: wire 0 is the most significant bit", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("PauliX", {}, {wires[0]}, false, {}, {}, {});
    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[2], WithinAbs(1.0, 1e-12)); // |10>
}

TEST_CASE("DefaultTensor: expectation values and variance", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    const double theta = 0.7;
    sim.NamedOperation("RY", {theta}, {wires[0]}, false, {}, {}, {});

    auto z = sim.Observable(PauliZ, {}, {wires[0]});
    CHECK_THAT(sim.Expval(z), WithinAbs(std::cos(theta), 1e-12));
    CHECK_THAT(sim.Var(z), WithinAbs(1.0 - std::cos(theta) * std::cos(theta), 1e-12));
}

TEST_CASE("DefaultTensor: tensor and Hamiltonian observables", "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: adjoint round trip", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    sim.NamedOperation("RX", {0.9}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("RX", {0.9}, {wires[0]}, /*inverse=*/true, {}, {}, {});
    auto z = sim.Observable(PauliZ, {}, {wires[0]});
    CHECK_THAT(sim.Expval(z), WithinAbs(1.0, 1e-12));
}

TEST_CASE("DefaultTensor: multi-controlled gates", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(3);
    sim.NamedOperation("PauliX", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("PauliX", {}, {wires[1]}, false, {}, {}, {});
    // Toffoli expressed as X on wire 2 with two control wires.
    sim.NamedOperation("PauliX", {}, {wires[2]}, false, {wires[0], wires[1]}, {true, true}, {});
    auto probs = allProbs(sim, 3);
    CHECK_THAT(probs[7], WithinAbs(1.0, 1e-12));
}

TEST_CASE("DefaultTensor: dynamic qubit allocation mid-circuit", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto first = sim.AllocateQubits(1);
    sim.NamedOperation("Hadamard", {}, {first[0]}, false, {}, {}, {});

    const QubitIdType second = sim.AllocateQubit();
    REQUIRE(sim.GetNumQubits() == 2);
    sim.NamedOperation("CNOT", {}, {first[0], second}, false, {}, {}, {});

    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[0], WithinAbs(0.5, 1e-12));
    CHECK_THAT(probs[3], WithinAbs(0.5, 1e-12));
}

TEST_CASE("DefaultTensor: releasing a qubit keeps the state normalised", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    sim.ReleaseQubit(wires[1]);
    REQUIRE(sim.GetNumQubits() == 1);

    auto probs = allProbs(sim, 1);
    CHECK_THAT(probs[0] + probs[1], WithinAbs(1.0, 1e-12));
    const bool collapsed = (probs[0] > 0.999) || (probs[1] > 0.999);
    CHECK(collapsed);
}

TEST_CASE("DefaultTensor: qubit IDs are never reused", "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: operating on a released qubit raises", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.ReleaseQubit(wires[1]);
    REQUIRE_THROWS_WITH(sim.NamedOperation("Hadamard", {}, {wires[1]}, false, {}, {}, {}),
                        ContainsSubstring("unallocated or released qubit"));
}

TEST_CASE("DefaultTensor: bulk teardown does not contract", "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: partial probabilities avoid the full state", "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: mid-circuit measurement collapses the state", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    Result r = sim.Measure(wires[0], 1); // postselect |1>
    CHECK(*r == true);
    auto probs = allProbs(sim, 2);
    CHECK_THAT(probs[3], WithinAbs(1.0, 1e-12)); // collapsed to |11>
}

TEST_CASE("DefaultTensor: shots, sampling and counts", "[DefaultTensor]") {
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
    CHECK(counts[1] == 0);
    CHECK(counts[2] == 0);
}

TEST_CASE("DefaultTensor: unsupported gates raise", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    // SWAP/S/T are deliberately absent from the bare gate set; Catalyst is
    // expected to decompose them before they reach the device.
    REQUIRE_THROWS_WITH(sim.NamedOperation("PauliRot", {0.5}, {wires[0]}, false, {}, {}, {}),
                        ContainsSubstring("unsupported gate"));
}

TEST_CASE("DefaultTensor: oversized contraction raises instead of exhausting RAM",
          "[DefaultTensor]") {
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

TEST_CASE("DefaultTensor: device kwargs are parsed", "[DefaultTensor]") {
    REQUIRE_THROWS_WITH(DefaultTensor{"{'max_intermediate_log2': 99}"},
                        ContainsSubstring("max_intermediate_log2"));
}

TEST_CASE("DefaultTensor: probabilities are shot-based when shots are set", "[DefaultTensor]") {
    // Regression test. `Probs`/`PartialProbs` must ESTIMATE from samples when a
    // finite shot count is set, matching LightningSimulator::Probs (which
    // branches on `device_shots != 0`). Returning analytic probabilities under
    // shots made results differ from every other device for the same program.
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    sim.SetDeviceShots(2000);
    bool sawNoise = false;
    for (int rep = 0; rep < 5; rep++) {
        auto probs = allProbs(sim, 2);
        if (std::abs(probs[0] - 0.5) > 1e-12) {
            sawNoise = true;
        }
        // States outside the Bell support must never be sampled.
        CHECK_THAT(probs[1], WithinAbs(0.0, 1e-12));
        CHECK_THAT(probs[2], WithinAbs(0.0, 1e-12));
        CHECK_THAT(probs[0] + probs[3], WithinAbs(1.0, 1e-12));
        CHECK_THAT(probs[0], WithinAbs(0.5, 0.05));
    }
    CHECK(sawNoise);

    // With shots=0 the same call must be exact again.
    sim.SetDeviceShots(0);
    auto exact = allProbs(sim, 2);
    CHECK_THAT(exact[0], WithinAbs(0.5, 1e-12));
    CHECK_THAT(exact[3], WithinAbs(0.5, 1e-12));
}

TEST_CASE("DefaultTensor: expval and var are shot-based when shots are set", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    sim.NamedOperation("RY", {0.7}, {wires[0]}, false, {}, {}, {});
    auto z = sim.Observable(PauliZ, {}, {wires[0]});

    const double exact = std::cos(0.7);
    CHECK_THAT(sim.Expval(z), WithinAbs(exact, 1e-12));

    sim.SetDeviceShots(20000);
    CHECK_THAT(sim.Expval(z), WithinAbs(exact, 0.05));

    // A shot-based expectation value must not disturb the state: switching back
    // to analytic mode has to reproduce the exact value.
    sim.SetDeviceShots(0);
    CHECK_THAT(sim.Expval(z), WithinAbs(exact, 1e-12));
}

TEST_CASE("DefaultTensor: shot-based observables needing a basis rotation", "[DefaultTensor]") {
    // <X> on |+> is deterministic (+1), so sampling in the X eigenbasis must
    // return exactly 1. A device that sampled in the computational basis
    // instead would return ~0 here.
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(1);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    auto x = sim.Observable(PauliX, {}, {wires[0]});

    sim.SetDeviceShots(5000);
    CHECK_THAT(sim.Expval(x), WithinAbs(1.0, 1e-9));
    CHECK_THAT(sim.Var(x), WithinAbs(0.0, 1e-9));
}

TEST_CASE("DefaultTensor: shot-based tensor observable on a Bell state", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("Hadamard", {}, {wires[0]}, false, {}, {}, {});
    sim.NamedOperation("CNOT", {}, {wires[0], wires[1]}, false, {}, {}, {});

    auto z0 = sim.Observable(PauliZ, {}, {wires[0]});
    auto z1 = sim.Observable(PauliZ, {}, {wires[1]});
    auto zz = sim.TensorObservable({z0, z1});

    sim.SetDeviceShots(5000);
    // Z0Z1 is +1 on both |00> and |11>, so the outcome is deterministic.
    CHECK_THAT(sim.Expval(zz), WithinAbs(1.0, 1e-9));
    CHECK_THAT(sim.Var(zz), WithinAbs(0.0, 1e-9));
}

TEST_CASE("DefaultTensor: shot-based Hamiltonian expectation", "[DefaultTensor]") {
    DefaultTensor sim{"{}"};
    auto wires = sim.AllocateQubits(2);
    sim.NamedOperation("RY", {0.6}, {wires[0]}, false, {}, {}, {});
    auto z0 = sim.Observable(PauliZ, {}, {wires[0]});
    auto z1 = sim.Observable(PauliZ, {}, {wires[1]});
    auto ham = sim.HamiltonianObservable({0.5, 0.25}, {z0, z1});

    const double exact = sim.Expval(ham);
    sim.SetDeviceShots(40000);
    CHECK_THAT(sim.Expval(ham), WithinAbs(exact, 0.05));
}
