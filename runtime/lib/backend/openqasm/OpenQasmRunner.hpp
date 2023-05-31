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

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Exception.hpp"

#include <pybind11/embed.h>

namespace Catalyst::Runtime::Device::OpenQasm {

/**
 * A (RAII) class for `pybind11::initialize_interpreter` and `pybind11::finalize_interpreter`.
 *
 * @note This is not copiable or movable and used in C++ tests and the ExecutionContext manager
 * of the runtime to solve the issue with re-initialization of the Python interpreter in `catch2`
 * tests which also enables the runtime to reuse the same interpreter in the scope of the global
 * quantum device unique pointer.
 */
struct PythonInterpreterGuard {
    PythonInterpreterGuard() { pybind11::initialize_interpreter(); }
    ~PythonInterpreterGuard() { pybind11::finalize_interpreter(); }

    PythonInterpreterGuard(const PythonInterpreterGuard &) = delete;
    PythonInterpreterGuard(PythonInterpreterGuard &&) = delete;
    PythonInterpreterGuard &operator=(const PythonInterpreterGuard &) = delete;
    PythonInterpreterGuard &operator=(PythonInterpreterGuard &&) = delete;
};

/**
 * The OpenQasm circuit runner interface.
 */
class OpenQasmRunner {
  protected:
    using ResultType = pybind11::list;

  public:
    explicit OpenQasmRunner() = default;
    virtual ~OpenQasmRunner() = default;
    virtual auto runCircuit([[maybe_unused]] const std::string &circuit,
                            [[maybe_unused]] const std::string &hw_name,
                            [[maybe_unused]] size_t shots) -> std::optional<ResultType>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    virtual auto Probs([[maybe_unused]] const std::string &circuit,
                       [[maybe_unused]] const std::string &hw_name, [[maybe_unused]] size_t shots,
                       [[maybe_unused]] size_t num_qubits) -> std::vector<double>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    virtual auto Sample([[maybe_unused]] const std::string &circuit,
                        [[maybe_unused]] const std::string &hw_name, [[maybe_unused]] size_t shots,
                        [[maybe_unused]] size_t num_qubits) -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
};

/**
 * The OpenQasm circuit runner to execute an OpenQasm circuit on Braket Devices backed by
 * Amazon Braket Python SDK.
 */
class BraketRunner : public OpenQasmRunner {
    auto runCircuit(const std::string &circuit, const std::string &device, size_t shots)
        -> std::optional<ResultType> override
    {
        namespace py = pybind11;
        using namespace py::literals;

        RT_FAIL_IF(!Py_IsInitialized(), "The Python interpreter is not initialized");

        auto locals = py::dict("braket_device"_a = device, "circuit"_a = circuit, "shots"_a = shots,
                               "msg"_a = "");

        py::exec(
            R"(
                  from braket.aws import AwsDevice
                  from braket.ir.openqasm import Program as OpenQasmProgram

                  device = AwsDevice(braket_device)
                  try:
                      result = device.run(OpenQasmProgram(source=circuit), shots=int(shots)).result()
                      measure_counts = dict(result.measurement_counts) # Counter -> dict
                      measures = [dict(result.measurement_counts), dict(result.measurement_probabilities)]
                  except Exception as e:
                      msg = str(e)
              )",
            py::globals(), locals);

        auto &&msg = locals["msg"].cast<std::string>();
        RT_FAIL_IF(!msg.empty(), msg.c_str());

        return static_cast<ResultType>(locals["measures"]);
    }

    auto Probs(const std::string &circuit, const std::string &device, size_t shots,
               size_t num_qubits) -> std::vector<double> override
    {
        namespace py = pybind11;
        using namespace py::literals;

        RT_FAIL_IF(!Py_IsInitialized(), "The Python interpreter is not initialized");

        auto locals = py::dict("braket_device"_a = device, "circuit"_a = circuit, "shots"_a = shots,
                               "num_qubits"_a = num_qubits, "msg"_a = "");

        py::exec(
            R"(
                  from braket.aws import AwsDevice
                  from braket.ir.openqasm import Program as OpenQasmProgram

                  device = AwsDevice(braket_device)
                  try:
                      result = device.run(OpenQasmProgram(source=circuit), shots=int(shots)).result()
                      probs_dict = {int(s, 2): p for s, p in result.measurement_probabilities.items()}
                      probs_list = []
                      for i in range(2 ** int(num_qubits)):
                          probs_list.append(probs_dict[i] if i in probs_dict else 0)
                  except Exception as e:
                      msg = str(e)
              )",
            py::globals(), locals);

        auto &&msg = locals["msg"].cast<std::string>();
        RT_FAIL_IF(!msg.empty(), msg.c_str());

        py::list results = locals["probs_list"];

        std::vector<double> probs;
        probs.reserve(std::pow(2, num_qubits));
        for (py::handle item : results) {
            probs.push_back(item.cast<double>());
        }

        return probs;
    }

    auto Sample(const std::string &circuit, const std::string &device, size_t shots,
                size_t num_qubits) -> std::vector<size_t> override
    {
        namespace py = pybind11;
        using namespace py::literals;

        RT_FAIL_IF(!Py_IsInitialized(), "The Python interpreter is not initialized");

        auto locals = py::dict("braket_device"_a = device, "circuit"_a = circuit, "shots"_a = shots,
                               "msg"_a = "");

        py::exec(
            R"(
                  import numpy as np
                  from braket.aws import AwsDevice
                  from braket.ir.openqasm import Program as OpenQasmProgram

                  device = AwsDevice(braket_device)
                  try:
                      result = device.run(OpenQasmProgram(source=circuit), shots=int(shots)).result()
                      samples = np.array(result.measurements).flatten()
                  except Exception as e:
                      msg = str(e)
              )",
            py::globals(), locals);

        auto &&msg = locals["msg"].cast<std::string>();
        RT_FAIL_IF(!msg.empty(), msg.c_str());

        py::list results = locals["samples"];

        std::vector<size_t> samples;
        samples.reserve(shots * num_qubits);
        for (py::handle item : results) {
            samples.push_back(item.cast<size_t>());
        }

        return samples;
    }
};

} // namespace Catalyst::Runtime::Device::OpenQasm
