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

#pragma once

#include <complex>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Exception.hpp"

#include <pybind11/embed.h>

namespace Catalyst::Runtime::Device::OQC{

// To protect the py::exec calls concurrently
std::mutex runner_mu;

/**
 * The OQC circuit runner interface.
 */
struct OQCRunner {
    explicit OQCRunner() = default;
    virtual ~OQCRunner() = default;
    [[nodiscard]] virtual auto
    Counts([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
           [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
};

/**
 * The OQC circuit runner to execute an OpenQasm circuit on OQC cloud
 */
struct OQCRunner : public OQCRunner {

    [[nodiscard]] auto Counts(const std::string &circuit, const std::string &device, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t> override
    {
        std::lock_guard<std::mutex> lock(runner_mu);
        namespace py = pybind11;
        using namespace py::literals;

        RT_FAIL_IF(!Py_IsInitialized(), "The Python interpreter is not initialized");

        auto locals = py::dict("circuit"_a = circuit, "device"_a = device,
                               "kwargs"_a = kwargs, "shots"_a = shots, "msg"_a = "");

        py::exec(
            R"(
            from qcaas_client.client import OQCClient, QPUTask
            from scc.compiler.config import CompilerConfig, QuantumResultsFormat, Tket, TketOptimizations, OptimizationConfig


            try:
                # Set compiler configuration options (if desired, default settings will be used if no user input provided)
                shot_count = shots
                res_format = QuantumResultsFormat().binary_count()
                optimisations = Tket()
                optimisations.tket_optimizations = TketOptimizations.Two

                config = CompilerConfig(repeats=shot_count, results_format=res_format, optimizations=optimisations)

                QPUTask(program=circuit, config=config)

                #TODO: IMPLEMENT WAIT LOGIC

            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
            py::globals(), locals);

        auto &&msg = locals["msg"].cast<std::string>();
        RT_FAIL_IF(!msg.empty(), msg.c_str());

        py::list results = locals["counts"];

        // TODO: UPDATE
        std::vector<size_t> counts;
        counts.reserve(shots * num_qubits);
        for (py::handle item : results) {
            counts.push_back(item.cast<size_t>());
        }

        return counts;
    }
};

} // namespace Catalyst::Runtime::Device::OQC
