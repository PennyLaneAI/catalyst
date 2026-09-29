// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This nanobind module is dlopen'd by the PLPythonDevice runtime backend (see PLPythonRunner.hpp).
// It forwards the instructions of a running program to the Python side of the PennyLane Python
// device bridge, ``catalyst.device.python_device.runtime_dispatch``, which records them on a tape
// and executes that tape on the PennyLane device.

#include <string>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

extern "C" NB_EXPORT bool pl_dispatch(const char *method, const char *payload, void *result,
                                      void *error)
{
    namespace nb = nanobind;
    auto *results = reinterpret_cast<std::vector<double> *>(result);
    auto *message = reinterpret_cast<std::string *>(error);

    nb::gil_scoped_acquire lock;
    try {
        // Intentionally leaked: it must outlive the interpreter's module teardown.
        static auto *dispatch = new nb::object(
            nb::module_::import_("catalyst.device.python_device").attr("runtime_dispatch"));
        nb::object values = (*dispatch)(std::string(method), std::string(payload));
        for (nb::handle item : values) {
            results->push_back(nb::cast<double>(item));
        }
        return true;
    }
    catch (nb::python_error &e) {
        // Report the exception without its traceback. The exception itself is kept on the Python
        // side and re-raised once the program returns (see ``python_device.reraise_pending``).
        std::string type = nb::cast<std::string>(nb::handle(e.type()).attr("__name__"));
        *message = "[PennyLane Python device] " + type + ": " +
                   nb::cast<std::string>(nb::str(e.value()));
        return false;
    }
    catch (std::exception &e) {
        *message = std::string("[PennyLane Python device] ") + e.what();
        return false;
    }
}

NB_MODULE(pennylane_python_module, m) { m.doc() = "PennyLane Python device bridge"; }
