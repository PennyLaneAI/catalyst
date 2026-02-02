// Copyright 2023-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <fstream>

#include "ExecutionContext.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "RuntimeCAPI.h"

#include "ResourceTracker.hpp"
#include "TestUtils.hpp"
#include "Types.h"

using namespace Catch::Matchers;
using namespace Catalyst::Runtime;

namespace {
// A list of the expected resources that will be created by ApplyGates,
//   precisely 1 of each of these will be created
static const std::string RESOURCE_NAMES[] = {"PauliX",
                                             "C(Adjoint(T))",
                                             "Adjoint(T)",
                                             "C(S)",
                                             "2C(S)",
                                             "CNOT",
                                             "Adjoint(ControlledQubitUnitary)",
                                             "ControlledQubitUnitary",
                                             "Adjoint(QubitUnitary)",
                                             "QubitUnitary"};

static const std::string MEASUREMENT_NAMES[] = {
    "expval(PauliX)",
    "expval(PauliZ)",
    "expval(Prod(num_terms=2))",
    "expval(Hamiltonian(num_terms=2))",
    "probs(all wires)",
    "probs(1 wires)",
    "probs(2 wires)",
};

static void ApplyGates(ResourceTracker &tracker)
{
    if (tracker.GetMaxWires() < 3) {
        FAIL_CHECK("ApplyGates requires at least 3 wires to operate");
    }

    // Apply named gates to test all possible name modifiers (adj, controlled, etc.)
    tracker.NamedOperation("PauliX", false, {0});
    tracker.NamedOperation("T", true, {0});
    tracker.NamedOperation("S", false, {0}, {2});
    tracker.NamedOperation("S", false, {0}, {1, 2});
    tracker.NamedOperation("T", true, {0}, {2});
    tracker.NamedOperation("CNOT", false, {0, 1});

    // Apply matrix gates
    tracker.MatrixOperation(false, {0});
    tracker.MatrixOperation(false, {0}, {1});
    tracker.MatrixOperation(true, {0});
    tracker.MatrixOperation(true, {0}, {1, 2});
}

static void ApplyMeasurements(ResourceTracker &tracker)
{
    // Apply named gates to test all possible name modifiers (adj, controlled, etc.)
    auto pauli_x = tracker.Observable(ObsId::PauliX);
    auto pauli_z = tracker.Observable(ObsId::PauliZ);
    auto prod_obs = tracker.CombinedObservable("Prod", 2);
    auto ham_obs = tracker.CombinedObservable("Hamiltonian", 2);

    tracker.ObsMeasurement("expval", pauli_x);
    tracker.ObsMeasurement("expval", pauli_z);
    tracker.ObsMeasurement("expval", prod_obs);
    tracker.ObsMeasurement("expval", ham_obs);

    tracker.AnalyticalMeasurement("probs", "all");
    tracker.AnalyticalMeasurement("probs", "1");
    tracker.AnalyticalMeasurement("probs", "2");
}

} // namespace

TEST_CASE("Test Resource Tracker Reset", "[resourcetracking]")
{
    // Constructor should call Reset
    ResourceTracker tracker;

    CHECK(tracker.GetFilename() == "");
    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);
    CHECK(tracker.GetDepth() == 0);
    CHECK(tracker.GetComputeDepth() == false);

    tracker.SetResourcesFilename("foo.json");
    tracker.SetComputeDepth(true);
    for (size_t i = 0; i < 10; i++) {
        tracker.AllocateQubit(i);
    }
    tracker.NamedOperation("PauliX", false, {0}, {});
    tracker.NamedOperation("CNOT", false, {0, 1}, {});

    // Check that values are truly dirty
    CHECK(tracker.GetFilename() == "foo.json");
    CHECK(tracker.GetNumGates() != 0);
    CHECK(tracker.GetMaxWires() != 0);
    CHECK(tracker.GetTotalAllocations() != 0);
    CHECK(tracker.GetDepth() != 0);
    CHECK(tracker.GetComputeDepth() == true);

    tracker.Reset();

    // Ensure reset cleans values back to original
    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);
    CHECK(tracker.GetDepth() == 0);
    // Filename and whether to compute depth should not be reset
    CHECK(tracker.GetFilename() == "foo.json");
    CHECK(tracker.GetComputeDepth() == true);
}

TEST_CASE("Test Resource Tracker Wires", "[resourcetracking]")
{
    ResourceTracker tracker;
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);

    // Call SetMaxWires with various different values
    tracker.AllocateQubit(0);
    CHECK(tracker.GetMaxWires() == 1);
    CHECK(tracker.GetTotalAllocations() == 1);
    tracker.ReleaseQubit(0);
    CHECK(tracker.GetMaxWires() == 1);
    CHECK(tracker.GetTotalAllocations() == 1);

    for (size_t i = 1; i <= 4; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetMaxWires() == 4);
    CHECK(tracker.GetTotalAllocations() == 5);
    for (size_t i = 5; i <= 9; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetMaxWires() == 9);
    CHECK(tracker.GetTotalAllocations() == 10);
}

TEST_CASE("Test Resource Tracker Gate Types", "[resourcetracking]")
{
    ResourceTracker tracker;
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetNumGates() == 0);
    for (const auto &name : RESOURCE_NAMES) {
        CHECK(tracker.GetNumGates(name) == 0);
    }

    ApplyGates(tracker);
    CHECK(tracker.GetNumGates() == 10);
    for (const auto &name : RESOURCE_NAMES) {
        CHECK(tracker.GetNumGates(name) == 1);
    }

    // Apply the same gates again, should double the count
    ApplyGates(tracker);
    CHECK(tracker.GetNumGates() == 20);
    for (const auto &name : RESOURCE_NAMES) {
        CHECK(tracker.GetNumGates(name) == 2);
    }

    CHECK(tracker.GetNumGates("NonExistentGate") == 0); // should not exist
}

TEST_CASE("Test Resource Tracker Gate Sizes", "[resourcetracking]")
{
    ResourceTracker tracker;
    CHECK(tracker.GetNumGates() == 0);

    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }

    CHECK(tracker.GetNumGatesBySize(0) == 0);
    CHECK(tracker.GetNumGatesBySize(1) == 0);
    CHECK(tracker.GetNumGatesBySize(2) == 0);
    CHECK(tracker.GetNumGatesBySize(3) == 0);
    CHECK(tracker.GetNumGatesBySize(4) == 0);

    ApplyGates(tracker);
    CHECK(tracker.GetNumGates() == 10);

    CHECK(tracker.GetNumGatesBySize(0) == 0);
    CHECK(tracker.GetNumGatesBySize(1) == 4);
    CHECK(tracker.GetNumGatesBySize(2) == 4);
    CHECK(tracker.GetNumGatesBySize(3) == 2);
    CHECK(tracker.GetNumGatesBySize(4) == 0);
}

TEST_CASE("Test Resource Tracker Depth", "[resourcetracking]")
{
    ResourceTracker tracker;
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }

    CHECK(tracker.GetComputeDepth() == false);
    CHECK(tracker.GetDepth() == 0);

    tracker.NamedOperation("PauliX", false, {0});
    CHECK(tracker.GetDepth() == 0); // Should not change if depth tracking disabled

    // Enable depth tracking and test
    tracker.Reset();
    tracker.SetComputeDepth(true);
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetComputeDepth() == true);
    CHECK(tracker.GetDepth() == 0);

    tracker.NamedOperation("PauliX", false, {0});
    CHECK(tracker.GetDepth() == 1);

    tracker.NamedOperation("CNOT", false, {0, 1});
    tracker.NamedOperation("CNOT", false, {1, 2});
    CHECK(tracker.GetDepth() == 3);

    for (size_t i = 0; i < 5; i++) {
        tracker.NamedOperation("PauliX", false, {3});
    }
    CHECK(tracker.GetDepth() == 5);

    tracker.NamedOperation("CNOT", false, {0, 1});
    CHECK(tracker.GetDepth() == 5);

    tracker.NamedOperation("CNOT", false, {2, 3});
    CHECK(tracker.GetDepth() == 6);
    tracker.NamedOperation("CNOT", false, {1, 2});
    CHECK(tracker.GetDepth() == 7);

    // Check that deallocating qubits doesn't erase depth
    for (size_t i = 0; i < 5; i++) {
        tracker.ReleaseQubit(i);
    }
    CHECK(tracker.GetDepth() == 7);

    // Check that reset works as expected
    tracker.Reset();
    CHECK(tracker.GetDepth() == 0); // Should be reset

    tracker.SetComputeDepth(false);
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }

    tracker.NamedOperation("PauliX", false, {0});
    CHECK(tracker.GetDepth() == 0); // Should not change if depth tracking disabled
}

TEST_CASE("Test Resource Tracker Runtime Checks", "[resourcetracking]")
{
    ResourceTracker tracker;
    tracker.SetComputeDepth(true);
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }

    CHECK(tracker.GetNumGatesBySize(0) == 0);
    CHECK(tracker.GetNumGatesBySize(1) == 0);
    CHECK(tracker.GetNumGatesBySize(2) == 0);
    CHECK(tracker.GetNumGatesBySize(3) == 0);
    CHECK(tracker.GetNumGatesBySize(4) == 0);
    CHECK(tracker.GetDepth() == 0);

    // Exceeds max wires, should throw and fail to increase depth, gate counts
    CHECK_THROWS(tracker.NamedOperation("PauliX", false, {10}));

    CHECK(tracker.GetNumGatesBySize(0) == 0);
    CHECK(tracker.GetNumGatesBySize(1) == 0);
    CHECK(tracker.GetNumGatesBySize(2) == 0);
    CHECK(tracker.GetNumGatesBySize(3) == 0);
    CHECK(tracker.GetNumGatesBySize(4) == 0);
    CHECK(tracker.GetDepth() == 0);
}

TEST_CASE("Test Resource Tracker Printing", "[resourcetracking]")
{
    // The name of the file where the resource usage data is stored
    constexpr char RESOURCES_FILENAME[] = "__pennylane_resources_data.json";

    // Open a file for writing the resources JSON
    FILE *resource_file_w = fopen(RESOURCES_FILENAME, "wx");
    if (resource_file_w == nullptr) {                            // LCOV_EXCL_LINE
        FAIL("Failed to open resource usage file for writing."); // LCOV_EXCL_LINE
    }

    ResourceTracker tracker;
    tracker.SetComputeDepth(true);
    CHECK(tracker.GetFilename() == "");
    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);

    for (size_t i = 0; i < 4; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetMaxWires() == 4);
    CHECK(tracker.GetTotalAllocations() == 4);

    ApplyGates(tracker);

    CHECK(tracker.GetNumGates() == 10);
    CHECK(tracker.GetMaxWires() == 4);
    CHECK(tracker.GetTotalAllocations() == 4);

    // Capture resources usage
    tracker.PrintResourceUsageToFile(resource_file_w);
    fclose(resource_file_w);

    // Open the file of resource data
    std::ifstream resource_file_r(RESOURCES_FILENAME);
    CHECK(resource_file_r.is_open()); // fail-fast if file failed to create

    std::string full_json;
    std::string line;
    // Check all fields have the correct value
    while (std::getline(resource_file_r, line)) {
        full_json += line + "\n";

        // For general data lines, check that the correct number is found
        if (line.find("num_wires") != std::string::npos) {
            CHECK(line.find("4") != std::string::npos);
        }
        if (line.find("num_gates") != std::string::npos) {
            CHECK(line.find("10") != std::string::npos);
        }
        if (line.find("depth") != std::string::npos) {
            CHECK(line.find("10") != std::string::npos);
        }

        // If one of the resource names is in the line, check that there is precisely 1
        for (const auto &name : RESOURCE_NAMES) {
            if (line.find(name) != std::string::npos) {
                CHECK(line.find("1") != std::string::npos);
                break;
            }
        }
    }
    resource_file_r.close();
    std::remove(RESOURCES_FILENAME);

    // Ensure all expected fields were present
    CHECK(full_json.find("num_wires") != std::string::npos);
    CHECK(full_json.find("num_gates") != std::string::npos);
    CHECK(full_json.find("depth") != std::string::npos);
    CHECK(full_json.find("gate_types") != std::string::npos);
    CHECK(full_json.find("gate_sizes") != std::string::npos);
    for (const auto &name : RESOURCE_NAMES) {
        CHECK(full_json.find(name) != std::string::npos);
    }
}

TEST_CASE("Test Resource Tracker WriteOut", "[resourcetracking]")
{
    ResourceTracker tracker;
    CHECK(tracker.GetFilename() == "");
    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);

    for (size_t i = 0; i < 4; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetMaxWires() == 4);
    CHECK(tracker.GetTotalAllocations() == 4);

    ApplyGates(tracker);

    CHECK(tracker.GetNumGates() == 10);
    CHECK(tracker.GetMaxWires() == 4);
    CHECK(tracker.GetTotalAllocations() == 4);

    // Check that writing out the normal way resets
    tracker.WriteOut();

    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetMaxWires() == 0);
    CHECK(tracker.GetTotalAllocations() == 0);

    std::string filename = tracker.GetFilename();
    CHECK(filename != "");
    CHECK(filename.find("__pennylane_resources_data_") != std::string::npos);
    CHECK(filename.find(".json") != std::string::npos);

    // Open the file of resource data
    std::ifstream resource_file_r(filename);
    CHECK(resource_file_r.is_open()); // fail-fast if file failed to create

    std::string full_json;
    std::string line;
    // Check all fields have the correct value
    while (std::getline(resource_file_r, line)) {
        full_json += line + "\n";

        // For general data lines, check that the correct number is found
        if (line.find("num_wires") != std::string::npos) {
            CHECK(line.find("4") != std::string::npos);
        }
        if (line.find("total_allocations") != std::string::npos) {
            CHECK(line.find("4") != std::string::npos);
        }
        if (line.find("num_gates") != std::string::npos) {
            CHECK(line.find("10") != std::string::npos);
        }
        if (line.find("depth") != std::string::npos) {
            CHECK(line.find("null") != std::string::npos);
        }

        // If one of the resource names is in the line, check that there is precisely 1
        for (const auto &name : RESOURCE_NAMES) {
            if (line.find(name) != std::string::npos) {
                CHECK(line.find("1") != std::string::npos);
                break;
            }
        }
    }

    resource_file_r.close();
    std::remove(filename.c_str());

    // Ensure all expected fields were present
    CHECK(full_json.find("num_wires") != std::string::npos);
    CHECK(full_json.find("total_allocations") != std::string::npos);
    CHECK(full_json.find("num_gates") != std::string::npos);
    CHECK(full_json.find("depth") != std::string::npos);
    CHECK(full_json.find("gate_types") != std::string::npos);
    CHECK(full_json.find("gate_sizes") != std::string::npos);
    for (const auto &name : RESOURCE_NAMES) {
        CHECK(full_json.find(name) != std::string::npos);
    }
}

TEST_CASE("Test Resource Tracker SetState Operations", "[resourcetracking]")
{
    ResourceTracker tracker;
    tracker.SetComputeDepth(true);
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetNumGates() == 0);
    CHECK(tracker.GetNumGates("StatePrep") == 0);
    CHECK(tracker.GetNumGates("BasisState") == 0);

    tracker.SetState({0});
    tracker.SetBasisState({0, 1, 2});
    CHECK(tracker.GetNumGates() == 2);
    CHECK(tracker.GetNumGates("StatePrep") == 1);
    CHECK(tracker.GetNumGates("BasisState") == 1);

    CHECK(tracker.GetNumGatesBySize(1) == 1);
    CHECK(tracker.GetNumGatesBySize(3) == 1);
    CHECK(tracker.GetDepth() == 2);
}

TEST_CASE("Test Resource Tracker Measurements", "[resourcetracking]")
{
    ResourceTracker tracker;
    for (size_t i = 0; i < 5; i++) {
        tracker.AllocateQubit(i);
    }
    CHECK(tracker.GetNumMeasurements() == 0);
    for (const auto &name : MEASUREMENT_NAMES) {
        CHECK(tracker.GetNumMeasurements(name) == 0);
    }

    ApplyMeasurements(tracker);
    CHECK(tracker.GetNumMeasurements() == 7);
    for (const auto &name : MEASUREMENT_NAMES) {
        CHECK(tracker.GetNumMeasurements(name) == 1);
    }

    // Apply the same gates again, should double the count
    ApplyMeasurements(tracker);
    CHECK(tracker.GetNumMeasurements() == 14);
    for (const auto &name : MEASUREMENT_NAMES) {
        CHECK(tracker.GetNumMeasurements(name) == 2);
    }

    CHECK(tracker.GetNumMeasurements("NonExistentMeasurement") == 0); // should not exist
}
