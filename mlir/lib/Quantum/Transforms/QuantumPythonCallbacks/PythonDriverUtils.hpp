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

#include <memory>
#include <string>

namespace QuantumPythonCallbacks {

class PyInterpreterWrapper {
  public:
    PyInterpreterWrapper();
    ~PyInterpreterWrapper();

    PyInterpreterWrapper(const PyInterpreterWrapper &) = delete;
    PyInterpreterWrapper &operator=(const PyInterpreterWrapper &) = delete;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    void extracted();
    void syncSitePackages();
};

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

} // namespace QuantumPythonCallbacks
