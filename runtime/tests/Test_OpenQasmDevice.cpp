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

#include "MemRefUtils.hpp"

#include "ExecutionContext.hpp"
#include "OpenQasmBuilder.hpp"
#include "OpenQasmDevice.hpp"
#include "OpenQasmRunner.hpp"
#include "RuntimeCAPI.h"

Catalyst::Runtime::PythonInterpreterGuard guard{};

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime::Device;
using BType = OpenQasm::BuilderType;

TEST_CASE("Test OpenQasmRunner base class", "[openqasm]")
{
    // check the coverage support
    OpenQasm::OpenQasmRunner runner{};
    REQUIRE_THROWS_WITH(runner.runCircuit("", "", 0),
                        Catch::Contains("[Function:runCircuit] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Probs("", "", 0, 0),
                        Catch::Contains("[Function:Probs] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Sample("", "", 0, 0),
                        Catch::Contains("[Function:Sample] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Expval("", "", 0),
                        Catch::Contains("[Function:Expval] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Var("", "", 0),
                        Catch::Contains("[Function:Var] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.State("", "", 0, 0),
                        Catch::Contains("[Function:State] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Gradient("", "", 0, 0),
                        Catch::Contains("[Function:Gradient] Error in Catalyst Runtime: "
                                        "Not implemented method"));
}

TEST_CASE("Test BraketRunner::runCircuit()", "[openqasm]")
{
    OpenQasm::BraketBuilder builder{};

    builder.Register(OpenQasm::RegisterType::Qubit, "q", 2);

    builder.Gate("Hadamard", {}, {}, {0}, false);
    builder.Gate("CNOT", {}, {}, {0, 1}, false);

    auto &&circuit = builder.toOpenQasm();

    OpenQasm::BraketRunner runner{};
    auto &&results = runner.runCircuit(circuit, "default", 100);
    CHECK(results.find("GateModelQuantumTaskResult") != std::string::npos);
}

TEST_CASE("Test the OpenQasmDevice constructor", "[openqasm]")
{
    SECTION("Common")
    {
        auto device = OpenQasmDevice("{shots : 100}");
        CHECK(device.GetNumQubits() == 0);

        REQUIRE_THROWS_WITH(device.Circuit(),
                            Catch::Contains("[Function:toOpenQasm] Error in Catalyst Runtime: "
                                            "Invalid number of quantum register"));
    }

    SECTION("Braket SV1")
    {
        auto device =
            OpenQasmDevice("{shots: 100, device_type : braket.local.qubit, backend : default}");
        CHECK(device.GetNumQubits() == 0);

        REQUIRE_THROWS_WITH(device.Circuit(),
                            Catch::Contains("[Function:toOpenQasm] Error in Catalyst Runtime: "
                                            "Invalid number of quantum register"));
    }
}

TEST_CASE("Test qubits allocation OpenQasmDevice", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>("{shots : 100}");

    constexpr size_t n = 3;
    device->AllocateQubits(1);
    CHECK(device->GetNumQubits() == 1);

    REQUIRE_THROWS_WITH(
        device->AllocateQubits(n - 1),
        Catch::Contains("[Function:AllocateQubits] Error in Catalyst Runtime: Partial qubits "
                        "allocation is not supported by OpenQasmDevice"));
}

TEST_CASE("Test the bell pair circuit with BuilderType::Common", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>("{shots : 100}");

    constexpr size_t n = 2;
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[1]}, false);

    device->Measure(wires[1]);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[2] qubits;\n"
                         "bit[2] bits;\n"
                         "h qubits[0];\n"
                         "cnot qubits[0], qubits[1];\n"
                         "bits[1] = measure qubits[1];\n"
                         "reset qubits;\n";

    CHECK(device->Circuit() == toqasm);
}

TEST_CASE("Test measurement processes, the bell pair circuit with BuilderType::Braket",
          "[openqasm]")
{
    constexpr size_t shots{1000};
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>(
        "{device_type : braket.local.qubit, backend : default, shots : 1000}");

    constexpr size_t n{2};
    constexpr size_t size{1UL << n};
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[1]}, false);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[2] qubits;\n"
                         "bit[2] bits;\n"
                         "h qubits[0];\n"
                         "cnot qubits[0], qubits[1];\n"
                         "bits = measure qubits;\n";

    CHECK(device->Circuit() == toqasm);

    SECTION("Probs")
    {
        std::vector<double> probs(size);
        DataView<double, 1> view(probs);
        device->Probs(view);

        CHECK(probs[1] == probs[2]);
        CHECK(probs[0] + probs[3] == Approx(1.f).margin(1e-5));
    }

    SECTION("PartialProbs")
    {
        std::vector<double> probs(size);
        DataView<double, 1> view(probs);
        device->PartialProbs(view, std::vector<QubitIdType>{0, 1});

        CHECK(probs[0] + probs[3] == Approx(1.f).margin(1e-5));
    }

    SECTION("Samples")
    {
        std::vector<double> samples(shots * n);
        MemRefT<double, 2> buffer{samples.data(), samples.data(), 0, {shots, n}, {1, 1}};
        DataView<double, 2> view(buffer.data_aligned, buffer.offset, buffer.sizes, buffer.strides);
        device->Sample(view, shots);

        for (size_t i = 0; i < shots * n; i++) {
            CHECK((samples[i] == 0.f || samples[i] == 1.f));
        }
    }

    SECTION("PartialSamples")
    {
        std::vector<double> samples(shots);
        MemRefT<double, 2> buffer{samples.data(), samples.data(), 0, {shots, 1}, {1, 1}};
        DataView<double, 2> view(buffer.data_aligned, buffer.offset, buffer.sizes, buffer.strides);
        device->PartialSample(view, std::vector<QubitIdType>{0}, shots);

        for (size_t i = 0; i < shots; i++) {
            CHECK((samples[i] == 0.f || samples[i] == 1.f));
        }
    }

    SECTION("Counts")
    {
        std::vector<double> eigvals(size);
        std::vector<int64_t> counts(size);
        DataView<double, 1> eview(eigvals);
        DataView<int64_t, 1> cview(counts);
        device->Counts(eview, cview, shots);

        size_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            CHECK(eigvals[i] == static_cast<double>(i));
            sum += counts[i];
        }
        CHECK(sum == shots);
    }

    SECTION("PartialCounts")
    {
        size_t size = (1UL << 1);
        std::vector<double> eigvals(size);
        std::vector<int64_t> counts(size);
        DataView<double, 1> eview(eigvals);
        DataView<int64_t, 1> cview(counts);
        device->PartialCounts(eview, cview, std::vector<QubitIdType>{1}, shots);

        size_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            CHECK(eigvals[i] == static_cast<double>(i));
            sum += counts[i];
        }
        CHECK(sum == shots);
    }

    SECTION("Expval(h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto expval = device->Expval(obs);
        CHECK(expval == Approx(0.0).margin(1e-5));
    }

    SECTION("Expval(x(0) @ h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs_x = device->Observable(ObsId::PauliX, {}, std::vector<QubitIdType>{0});
        auto obs_h = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto obs = device->TensorObservable({obs_x, obs_h});
        auto expval = device->Expval(obs);
        CHECK(expval == Approx(0.7071067812).margin(1e-5));
    }

    SECTION("Var(h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto expval = device->Var(obs);
        CHECK(expval == Approx(1.0).margin(1e-5));
    }

    SECTION("Var(x(0) @ h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs_x = device->Observable(ObsId::PauliX, {}, std::vector<QubitIdType>{0});
        auto obs_h = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto obs = device->TensorObservable({obs_x, obs_h});
        auto expval = device->Var(obs);
        CHECK(expval == Approx(0.5).margin(1e-5));
    }
}

TEST_CASE("Test measurement processes, a simple circuit with BuilderType::Braket", "[openqasm]")
{
    constexpr size_t shots{1000};
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>(
        "{device_type : braket.local.qubit, backend : default, shots : 1000}");

    constexpr size_t n{5};
    constexpr size_t size{1UL << n};
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("PauliX", {}, {wires[0]}, false);
    device->NamedOperation("PauliY", {}, {wires[1]}, false);
    device->NamedOperation("PauliZ", {}, {wires[2]}, false);
    device->NamedOperation("RX", {0.6}, {wires[4]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[3]}, false);
    device->NamedOperation("Toffoli", {}, {wires[0], wires[3], wires[4]}, false);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[5] qubits;\n"
                         "bit[5] bits;\n"
                         "x qubits[0];\n"
                         "y qubits[1];\n"
                         "z qubits[2];\n"
                         "rx(0.6) qubits[4];\n"
                         "cnot qubits[0], qubits[3];\n"
                         "ccnot qubits[0], qubits[3], qubits[4];\n"
                         "bits = measure qubits;\n";

    CHECK(device->Circuit() == toqasm);

    SECTION("Probs")
    {
        std::vector<double> probs(size);
        DataView<double, 1> view(probs);
        device->Probs(view);

        CHECK(probs[27] + probs[26] == Approx(1.f).margin(1e-5));
    }

    SECTION("PartialProbs")
    {
        std::vector<double> probs(size);
        DataView<double, 1> view(probs);
        device->PartialProbs(view, std::vector<QubitIdType>{0, 1, 2, 3, 4});

        CHECK(probs[27] + probs[26] == Approx(1.f).margin(1e-5));
    }

    SECTION("Samples")
    {
        std::vector<double> samples(shots * n);
        MemRefT<double, 2> buffer{samples.data(), samples.data(), 0, {shots, n}, {1, 1}};
        DataView<double, 2> view(buffer.data_aligned, buffer.offset, buffer.sizes, buffer.strides);
        device->Sample(view, shots);

        for (size_t i = 0; i < shots * n; i++) {
            CHECK((samples[i] == 0.f || samples[i] == 1.f));
        }
    }

    SECTION("PartialSamples")
    {
        std::vector<double> samples(shots);
        MemRefT<double, 2> buffer{samples.data(), samples.data(), 0, {shots, 1}, {1, 1}};
        DataView<double, 2> view(buffer.data_aligned, buffer.offset, buffer.sizes, buffer.strides);
        device->PartialSample(view, std::vector<QubitIdType>{0}, shots);

        for (size_t i = 0; i < shots; i++) {
            CHECK((samples[i] == 0.f || samples[i] == 1.f));
        }
    }

    SECTION("Counts")
    {
        std::vector<double> eigvals(size);
        std::vector<int64_t> counts(size);
        DataView<double, 1> eview(eigvals);
        DataView<int64_t, 1> cview(counts);
        device->Counts(eview, cview, shots);

        size_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            CHECK(eigvals[i] == static_cast<double>(i));
            sum += counts[i];
        }
        CHECK(sum == shots);
    }

    SECTION("PartialCounts")
    {
        size_t size = (1UL << 1);
        std::vector<double> eigvals(size);
        std::vector<int64_t> counts(size);
        DataView<double, 1> eview(eigvals);
        DataView<int64_t, 1> cview(counts);
        device->PartialCounts(eview, cview, std::vector<QubitIdType>{1}, shots);

        size_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            CHECK(eigvals[i] == static_cast<double>(i));
            sum += counts[i];
        }
        CHECK(sum == shots);
    }

    SECTION("Expval(h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto expval = device->Expval(obs);
        CHECK(expval == Approx(-0.7071067812).margin(1e-5));
    }

    SECTION("Expval(hermitian(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        std::vector<std::complex<double>> matrix{
            {0, 0},
            {0, -1},
            {0, 1},
            {0, 0},
        };
        auto obs = device->Observable(ObsId::Hermitian, matrix, std::vector<QubitIdType>{1});
        auto expval = device->Expval(obs);
        CHECK(expval == Approx(0).margin(1e-5));
    }

    SECTION("Expval(x(0) @ h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs_z = device->Observable(ObsId::PauliZ, {}, std::vector<QubitIdType>{0});
        auto obs_h = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto tp = device->TensorObservable({obs_z, obs_h});
        auto expval = device->Expval(tp);
        CHECK(expval == Approx(0.7071067812).margin(1e-5));

        auto obs = device->HamiltonianObservable({0.2}, {tp});
        REQUIRE_THROWS_WITH(device->Expval(obs),
                            Catch::Contains("Unsupported observable: QasmHamiltonianObs"));
    }

    SECTION("Var(h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto var = device->Var(obs);
        CHECK(var == Approx(0.5).margin(1e-5));
    }

    SECTION("Var(hermitian(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        std::vector<std::complex<double>> matrix{
            {0, 0},
            {0, -1},
            {0, 1},
            {0, 0},
        };
        auto obs = device->Observable(ObsId::Hermitian, matrix, std::vector<QubitIdType>{1});
        auto var = device->Var(obs);
        CHECK(var == Approx(1).margin(1e-5));
    }

    SECTION("Var(x(0) @ h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs_z = device->Observable(ObsId::PauliZ, {}, std::vector<QubitIdType>{0});
        auto obs_h = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto tp = device->TensorObservable({obs_z, obs_h});
        auto var = device->Var(tp);
        CHECK(var == Approx(0.5).margin(1e-5));

        auto obs = device->HamiltonianObservable({0.2}, {tp});
        REQUIRE_THROWS_WITH(device->Var(obs),
                            Catch::Contains("Unsupported observable: QasmHamiltonianObs"));
    }
}

TEST_CASE("Test MatrixOperation with BuilderType::Braket", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>(
        "{device_type : braket.local.qubit, backend : default, shots : 1000}");

    constexpr size_t n{2};
    constexpr size_t size{1UL << n};
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("PauliX", {}, {wires[0]}, false);
    device->NamedOperation("PauliY", {}, {wires[1]}, false);
    std::vector<std::complex<double>> matrix{
        {0, 0},
        {0, -1},
        {0, 1},
        {0, 0},
    };
    device->MatrixOperation(matrix, {wires[0]}, false);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[2] qubits;\n"
                         "bit[2] bits;\n"
                         "x qubits[0];\n"
                         "y qubits[1];\n"
                         "#pragma braket unitary([[0, 0-1im], [0+1im, 0]]) qubits[0]\n"
                         "bits = measure qubits;\n";

    CHECK(device->Circuit() == toqasm);

    SECTION("Probs")
    {
        std::vector<double> probs(size);
        DataView<double, 1> view(probs);
        device->Probs(view);
        CHECK(probs[1] == Approx(1.f).margin(1e-5));
    }

    SECTION("Expval(h(1))")
    {
        device->SetDeviceShots(0); // to get deterministic results
        auto obs = device->Observable(ObsId::Hadamard, {}, std::vector<QubitIdType>{1});
        auto expval = device->Expval(obs);
        CHECK(expval == Approx(-0.7071067812).margin(1e-5));
    }
}

TEST_CASE("Test PSWAP and ISWAP with BuilderType::Braket", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>(
        "{device_type : braket.local.qubit, backend : default, shots : 1000}");

    constexpr size_t n{2};
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);
    device->NamedOperation("ISWAP", {}, {wires[0], wires[1]}, false);
    device->NamedOperation("PSWAP", {0}, {wires[0], wires[1]}, false);

    auto obs = device->Observable(ObsId::PauliZ, {}, std::vector<QubitIdType>{1});
    auto expval = device->Expval(obs);
    CHECK(expval == Approx(1).margin(1e-5));
}

TEST_CASE("Test MatrixOperation with OpenQasmDevice and BuilderType::Common", "[openqasm]")
{
    auto device = OpenQasmDevice("{shots : 100}");
    auto wires = device.AllocateQubits(2);
    std::vector<std::complex<double>> matrix{
        {0, 0},
        {0, -1},
        {0, 1},
        {0, 0},
    };

    REQUIRE_THROWS_WITH(device.MatrixOperation(matrix, {wires[0]}, false),
                        Catch::Contains("Unsupported functionality"));
}

TEST_CASE("Test __catalyst__rt__device_init registering the OpenQasm device", "[CoreQIS]")
{
    __catalyst__rt__initialize();

    char device_aws[30] = "braket.aws.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __catalyst__rt__device_init((int8_t *)device_aws, nullptr, nullptr);
#else
    REQUIRE_THROWS_WITH(__catalyst__rt__device_init((int8_t *)device_aws, nullptr, nullptr),
                        Catch::Contains("cannot open shared object file"));
#endif

    __catalyst__rt__finalize();

    __catalyst__rt__initialize();

    char device_local[30] = "braket.local.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __catalyst__rt__device_init((int8_t *)device_local, nullptr, nullptr);
#else
    REQUIRE_THROWS_WITH(__catalyst__rt__device_init((int8_t *)(int8_t *), nullptr, nullptr),
                        Catch::Contains("cannot open shared object file"));
#endif

    __catalyst__rt__finalize();
}
