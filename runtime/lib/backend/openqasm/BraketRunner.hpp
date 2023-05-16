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

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include "Exception.hpp"

#include <pybind11/embed.h>

namespace Catalyst::Runtime::Device::OpenQasm {

/**
 * This class provides methods to execute OpenQasm circuits using Amazon Braket
 * Python SDK backed by `pybind11::embed`
 *
 * @param circuit An OpenQasm supported circuit by Braket
 * @param result The serialized version of the result
 */
class BraketRunner {
  private:
    std::string circuit;
    std::string result;

  public:
    explicit BraketRunner(std::string _circuit) : circuit(std::move(_circuit)) {}
    explicit BraketRunner() {}
    ~BraketRunner() = default;

    void addCircuit(const std::string &_circuit) { this->circuit = _circuit; }

    auto getCircuit() const -> std::string { return this->circuit; }

    auto getResult() const -> std::string { return this->result; }

    void runCircuit([[maybe_unused]] const std::string &device, size_t shots)
    {
        using namespace pybind11::literals;

        pybind11::scoped_interpreter gaurd{};
        pybind11::print(this->circuit);
        auto locals = pybind11::dict("braket_device"_a = device, "circuit"_a = this->circuit,
                                     "num_shots"_a = shots, "results"_a = "");

        pybind11::exec(
            R"(
                from braket.aws import AwsDevice
                from braket.ir.openqasm import Program as OpenQasmProgram

                device = AwsDevice(braket_device)
                result = device.run(OpenQasmProgram(source=circuit), shots=int(num_shots)).result()
                results = str(result)
            )",
            pybind11::globals(), locals);

        this->result = locals["results"].cast<std::string>();
    }
};

} // namespace Catalyst::Runtime::Device::OpenQasm
