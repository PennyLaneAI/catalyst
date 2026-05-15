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

#include "pybind11/embed.h"

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

struct __attribute__((visibility("hidden"))) PyInterpreterWrapper::Impl {
    py::scoped_interpreter interpreter;
};

PyInterpreterWrapper::PyInterpreterWrapper() : impl(std::make_unique<Impl>())
{
    syncSitePackages();
}

PyInterpreterWrapper::~PyInterpreterWrapper() = default;

void PyInterpreterWrapper::syncSitePackages()
{
    py::gil_scoped_acquire acquire;

    try {
        py::exec(sitePackagesScript);
    }
    catch (const py::error_already_set &e) {
        return;
    }
}
} // namespace QuantumPythonCallbacks
