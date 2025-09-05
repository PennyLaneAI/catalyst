// Copyright 2024 Xanadu Quantum Technologies Inc.
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

#include <string>
#include <vector>

#include <nanobind/eval.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "Exception.hpp"

std::string program = R"(
# check if Python execution works
import os
msg = ""
print("Python code executed successfully")
)";

extern "C" {
[[gnu::visibility("default")]] void counts(const char *_circuit, const char *_device, size_t shots,
                                           size_t num_qubits, const char *_kwargs, void *_vector)
{
    namespace nb = nanobind;
    using namespace nb::literals;

    nb::gil_scoped_acquire lock;

    nb::dict locals;
    locals["circuit"] = _circuit;
    locals["device"] = _device;
    locals["kwargs"] = _kwargs;
    locals["shots"] = shots;
    locals["num_qubits"] = num_qubits;
    locals["msg"] = "";

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope, locals);

    // auto msg = nb::cast<std::string>(locals["msg"]);
    // RT_FAIL_IF(!msg.empty(), msg.c_str());

    // nb::dict results = locals["counts"];

    // std::vector<size_t> *counts_value = reinterpret_cast<std::vector<size_t> *>(_vector);
    // for (auto item : results) {
    //     auto key = item.first;
    //     auto value = item.second;
    //     counts_value->push_back(nb::cast<size_t>(value));
    // }
    // return;
    return;
}
} // extern "C"

NB_MODULE(oqc_python_module, m) { m.doc() = "oqc"; }
