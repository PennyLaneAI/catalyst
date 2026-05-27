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

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace QuantumPythonCallbacks {

class QPCError : public std::runtime_error {
  public:
    explicit QPCError(std::string message) : std::runtime_error(std::move(message)) {}
};

class TracingError : public QPCError {
  public:
    TracingError(std::string moduleName, std::string functionName, std::string args,
                 std::string error)
        : QPCError("An error occurred while tracing " + functionName + " from module " +
                   moduleName + " with args " + args + ": " + error)
    {
    }
};

class PyInterpreterGuard {
  public:
    static PyInterpreterGuard &ensure();

    template <class T> decltype(auto) withGil(T &&func)
    {
        static thread_local int depth = 0;
        if (depth > 0) {
            throw QPCError("Recursive call to withGil detected.");
        }
        ++depth;
        struct DepthGuard {
            int &d;
            ~DepthGuard() { --d; }
        } guard{depth};

        nb::gil_scoped_acquire acquire;
        try {
            return std::invoke(std::forward<T>(func));
        }
        catch (const nb::python_error &e) {
            throw QPCError(e.what());
        }
    }

    PyInterpreterGuard(const PyInterpreterGuard &) = delete;
    PyInterpreterGuard &operator=(const PyInterpreterGuard &) = delete;

  private:
    PyInterpreterGuard();
    ~PyInterpreterGuard();

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace QuantumPythonCallbacks
