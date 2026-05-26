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
#include <string>
#include <vector>

#include "DynamicLibraryLoader.hpp"
#include "Exception.hpp"

namespace Catalyst::Runtime::Device::PLTape {

/**
 * @brief Runs serialized PennyLane tapes against a Python PennyLane device
 *        by calling into the companion nanobind shared library.
 *
 * This follows the exact same pattern as `BraketRunner`: the runner dlopen's
 * the companion `pennylane_python_module.so` shared library at each measurement
 * call, calls an extern "C" function, which acquires the Python GIL and executes
 * the tape on the specified PennyLane device.
 *
 * The shared library name is injected at compile time via the PLPYTHON_PY macro
 * (see CMakeLists.txt).
 */
struct PLPythonRunner {
    explicit PLPythonRunner() = default;
    ~PLPythonRunner() = default;

    [[nodiscard]] auto Execute(const std::string &tape_json, const std::string &obs_json,
                               const std::string &meas_json, const std::string &device_kwargs, 
                               size_t shots) const -> std::vector<double>
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);
        using fn_t = void (*)(const char *, const char *, const char *, const char *, size_t, void *);
        auto impl = libLoader.getSymbol<fn_t>("pl_execute");

        std::vector<double> result;
        impl(tape_json.c_str(), obs_json.c_str(), meas_json.c_str(), device_kwargs.c_str(), shots, &result);
        return result;
    }
};

} // namespace Catalyst::Runtime::Device::PLTape
