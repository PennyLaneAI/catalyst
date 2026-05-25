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

#include <optional>

constexpr const char *sitePackagesScript = R"(
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

namespace py = pybind11;

namespace QuantumPythonCallbacks {

struct __attribute__((visibility("hidden"))) PyInterpreterGuard::Impl {
    std::optional<py::scoped_interpreter> owned;
    std::optional<py::gil_scoped_release> release;
};

/**
 * @brief Execute a function while holding the GIL,
 * ensuring the interpreter is initialized.
 */
void syncSitePackages()
{
    py::gil_scoped_acquire acquire;

    try {
        py::exec(sitePackagesScript);
    }
    catch (const py::error_already_set &e) {
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
        impl->owned.emplace();
        syncSitePackages();
        impl->release.emplace();
    }
    // else the interpreter is already initialized (e.g. by qjit),
    // so we do not take ownership or release the GIL
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
