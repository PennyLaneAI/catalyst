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

#include "Driver/PythonDriverUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "pybind11/embed.h"

namespace py = pybind11;

namespace catalyst {
namespace driver {

constexpr const char *site_packages = R"(
import os
import sys
import site

# Check if the user is running from an active virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    # Construct the path: e.g., .venv/lib/python3.14/site-packages
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(venv_path, "lib", f"python{py_version}", "site-packages")
    
    # Adds the directory AND processes any editable install .pth files
    if os.path.exists(site_packages):
        site.addsitedir(site_packages)
)";

// We use a pointer to explicitly control the lifetime of the interpreter
static py::scoped_interpreter *global_interpreter = nullptr;

void initialize_python_runtime()
{
    if (global_interpreter) {
        return;
    }

    // llvm::errs() << "initializing interpreter...\n";

    // TODO: optimize this to use the PL interpreter if available
    global_interpreter = new py::scoped_interpreter();

    try {
        // llvm::errs() << "\tupdating site packages...\n";

        py::exec(site_packages);
    }
    catch (const py::error_already_set &e) {
        llvm::errs() << "Failed to link virtual environment: " << e.what() << "\n";
    }
    // llvm::errs() << "interpreter initialization complete\n";
}

void finalize_python_runtime()
{
    if (driver::global_interpreter) {
        delete driver::global_interpreter;
        driver::global_interpreter = nullptr;
    }
}

} // namespace driver
} // namespace catalyst
