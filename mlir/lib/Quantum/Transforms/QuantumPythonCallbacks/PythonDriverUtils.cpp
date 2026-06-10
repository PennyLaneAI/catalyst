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

#include "PythonDriverUtils.hpp"

#include <iostream>
#include <optional>

#include "Python.h"
#include "nanobind/eval.h"
#include "nanobind/nanobind.h"

#define DEBUG_TYPE "[QPC] "

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

namespace QuantumPythonCallbacks {

struct __attribute__((visibility("hidden"))) PyInterpreterGuard::Impl {
    bool createdInterpreter{false};
    std::optional<nb::gil_scoped_release> release;
};

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
        std::cout << "Failed to load site-packages: " << e.what();
        return;
    }
}

/**
 * @brief Construct a PyInterpreterGuard, initializing the Python interpreter if it is not already
 * initialized and releasing the GIL if it owns the interpreter.
 */
PyInterpreterGuard::PyInterpreterGuard() : impl(std::make_unique<Impl>())
{
    if (!Py_IsInitialized()) {
        Py_Initialize();
        impl->createdInterpreter = true;

        syncSitePackages();

        impl->release.emplace();
    }
    else {
        nb::gil_scoped_acquire acquire;
        syncSitePackages();
    }
}

/**
 * @brief Destroy a PyInterpreterGuard, ensuring that the GIL is released and the interpreter is
 * destroyed if it was created by the PyInterpreterGuard.
 */
PyInterpreterGuard::~PyInterpreterGuard()
{
    if (impl && impl->createdInterpreter) {
        impl->release.reset();
        Py_Finalize();
    }
}

/**
 * @brief Ensure the Python interpreter is initialized and the GIL is released,
 * returning a guard object that manages their lifetimes.
 *
 * If the interpreter is already initialized (e.g. by qjit),
 * this will simply return a guard that does not own the interpreter
 * and does not release the GIL.
 */
PyInterpreterGuard &PyInterpreterGuard::ensure()
{
    static PyInterpreterGuard *inst = new PyInterpreterGuard();
    return *inst;
}

} // namespace QuantumPythonCallbacks
