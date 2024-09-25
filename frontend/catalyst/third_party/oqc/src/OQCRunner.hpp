
// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <complex>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

#include "Exception.hpp"
#include "Python.hpp"

#include <pybind11/embed.h>

namespace Catalyst::Runtime::Device {

/**
 * The OpenQasm circuit runner interface.
 */
struct OQCRunnerBase {
    explicit OQCRunnerBase() = default;
    virtual ~OQCRunnerBase() = default;

    [[nodiscard]] virtual auto runCircuit([[maybe_unused]] const std::string &circuit,
                                          [[maybe_unused]] const std::string &device,
                                          [[maybe_unused]] size_t shots,
                                          [[maybe_unused]] const std::string &kwargs = "") const
        -> std::string
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Probs([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
          [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
          [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<double>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Sample([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
           [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Counts([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
           [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Expval([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] const std::string &kwargs = "") const
        -> double
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto Var([[maybe_unused]] const std::string &circuit,
                                   [[maybe_unused]] const std::string &device,
                                   [[maybe_unused]] size_t shots,
                                   [[maybe_unused]] const std::string &kwargs = "") const -> double
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    State([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
          [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
          [[maybe_unused]] const std::string &kwargs = "") const
        -> std::vector<std::complex<double>>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto Gradient([[maybe_unused]] const std::string &circuit,
                                        [[maybe_unused]] const std::string &device,
                                        [[maybe_unused]] size_t shots,
                                        [[maybe_unused]] size_t num_qubits,
                                        [[maybe_unused]] const std::string &kwargs = "") const
        -> std::vector<double>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
};

/**
 * The OpenQASM circuit runner to execute an OpenQASM circuit on OQC devices thanks to
 * OQC qcaas client.
 */
struct OQCRunner : public OQCRunnerBase {

    [[nodiscard]] auto Counts(const std::string &circuit, const std::string &device, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t>
    {
        namespace py = pybind11;
        using namespace py::literals;

        RT_FAIL_IF(!Py_IsInitialized(), "The Python interpreter is not initialized");

        auto locals = py::dict("circuit"_a = circuit, "device"_a = device, "kwargs"_a = kwargs,
                               "shots"_a = shots, "msg"_a = "");

        py::exec(
            R"(
            import os
            from qcaas_client.client import OQCClient, QPUTask, CompilerConfig
            from qcaas_client.config import QuantumResultsFormat, Tket, TketOptimizations
            optimisations = Tket()
            optimisations.tket_optimizations = TketOptimizations.DefaultMappingPass

            RES_FORMAT = QuantumResultsFormat().binary_count()

            try:
                email = os.environ.get("OQC_EMAIL")
                password = os.environ.get("OQC_PASSWORD")
                url = os.environ.get("OQC_URL")
                client = OQCClient(url=url, email=email, password=password)
                client.authenticate()
                oqc_config = CompilerConfig(repeats=shots, results_format=RES_FORMAT, optimizations=optimisations)
                oqc_task = QPUTask(circuit, oqc_config)
                res = client.execute_tasks(oqc_task)
                counts = res[0].result["cbits"]

            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
            )",
            py::globals(), locals);

        auto &&msg = locals["msg"].cast<std::string>();
        RT_FAIL_IF(!msg.empty(), msg.c_str());

        py::dict results = locals["counts"];

        std::vector<size_t> counts_value;
        for (auto item : results) {
            auto key = item.first;
            auto value = item.second;
            counts_value.push_back(value.cast<size_t>());
        }
        return counts_value;
    }
};

} // namespace Catalyst::Runtime::Device
