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
#include <iostream>
#include <stdexcept>
#include <string>

#include "nanobind/eval.h"
#include "nanobind/nanobind.h"

constexpr const char sitePackagesScript[] = R"(
import os
import sys
import site

venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(venv_path, "lib", f"python{py_version}", "site-packages")

    if os.path.exists(site_packages):
        site.addsitedir(site_packages)
)";

namespace nb = nanobind;

namespace QuantumPythonDecompositions {

class QPDError : public std::runtime_error {
  public:
    explicit QPDError(std::string message) : std::runtime_error(std::move(message)) {}
};

class TracingError : public QPDError {
  public:
    TracingError(std::string moduleName, std::string functionName, std::string args,
                 std::string error)
        : QPDError("An error occurred while tracing " + functionName + " from module " +
                   moduleName + " with args " + args + ": " + error)
    {
    }
};

class PyInterpreterGuard {
  public:
    PyInterpreterGuard(const PyInterpreterGuard &) = delete;
    PyInterpreterGuard &operator=(const PyInterpreterGuard &) = delete;

    PyInterpreterGuard()
    {
        if (!Py_IsInitialized()) {
            Py_Initialize();
            PyEval_SaveThread(); // release the GIL
        }

        nb::gil_scoped_acquire acquire;
        syncSitePackages();
    };

    template <class T> decltype(auto) withGil(T &&func)
    {
        nb::gil_scoped_acquire acquire;
        try {
            return std::invoke(std::forward<T>(func));
        }
        catch (const nb::python_error &e) {
            throw QPDError(e.what());
        }
    }

  private:
    /**
     * @brief Ensure that site-packages are synced for the spawned interpreter.
     */
    void syncSitePackages()
    {
        try {
            nb::object scope = nb::module_::import_("__main__").attr("__dict__");

            nb::exec(sitePackagesScript, scope);
        }
        catch (const nb::python_error &e) {
            std::cerr << "Failed to load site-packages: " << e.what();
            return;
        }
    }
};

} // namespace QuantumPythonDecompositions
