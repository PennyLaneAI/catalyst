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

#include <string>
#include <vector>

#include "nanobind/nanobind.h"

#include "PythonDriverUtils.hpp"

#define DEBUG_TYPE "[QPC] "

namespace nb = nanobind;

std::string pythonLowerPauliRot(double theta, const std::string &pauliWord, std::vector<int> wires)
{
    QuantumPythonCallbacks::PyInterpreterGuard guard;
    std::string mlirText = guard.withGil([&] -> std::string {
        const char *moduleName = "catalyst.python_callbacks";
        const char *functionName = "paulirot_callback_wrapper";

        try {
            nb::module_ wrapperModule = nb::module_::import_(moduleName);
            nb::object wrapperFunction = wrapperModule.attr(functionName);

            nb::list pyWires;
            for (int w : wires) {
                pyWires.append(w);
            }

            nb::object pythonResult = wrapperFunction(theta, pauliWord.c_str(), pyWires);

            return nb::borrow<nb::str>(pythonResult).c_str();
        }
        catch (const nb::python_error &error) {
            throw QuantumPythonCallbacks::TracingError(moduleName, functionName, pauliWord,
                                                       error.what());
        }
        catch (const std::exception &error) {
            throw;
        }
    });

    return mlirText;
}

extern "C" __attribute__((visibility("default"))) void *getPythonLowerPauliRot()
{
    return reinterpret_cast<void *>(pythonLowerPauliRot);
}
