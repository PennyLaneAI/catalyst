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

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <unordered_map>

namespace py = pybind11;

// From PyBind11's documentation:
//
//     Do you have any global variables that are pybind11 objects or invoke pybind11 functions in
//     either their constructor or destructor? You are generally not allowed to invoke any Python
//     function in a global static context. We recommend using lazy initialization and then
//     intentionally leaking at the end of the program.
//
// https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
std::unordered_map<int64_t, py::function> *references;

void sanitizeResult(py::object obj) {
    
}

void sanitizeResults(py::list results) {
    for (py::object obj : results) {
        sanitizeResult(obj);
    }
}

extern "C" {
[[gnu::visibility("default")]] void callbackCall(int64_t identifier, int64_t count, int64_t retc, va_list args)
{
    auto it = references->find(identifier);
    if (it == references->end()) {
        throw std::invalid_argument("Callback called with invalid identifier");
    }
    auto lambda = it->second;

    py::list flat_args;
    for (int i = 0; i < count; i++) {
        int64_t ptr = va_arg(args, int64_t);
        flat_args.append(ptr);
    }

    py::list flat_results = lambda(flat_args);

    // We have a flat list of return values.
    // These returns **may** be array views to
    // the very same memrefs that we passed as inputs.
    // As a first prototype, let's copy these values.
    // I think it is best to always copy them because
    // of aliasing. Let's just copy them to guarantee
    // no aliasing issues. We can revisit this as an optimization
    // and allowing these to alias.
    for (py::handle obj : flat_results) {
        
    }
}
}

auto registerImpl(py::function f)
{
    // Do we need to see if it is already present or can we just override it? Just override is fine.
    // Does python reuse id's? Yes.
    // But only after they have been garbaged collected.
    // So as long as we maintain a reference to it, then they won't be garbage collected.
    // Inserting the function into the unordered map increases the reference by one.
    int64_t id = (int64_t)f.ptr();
    references->insert({id, f});
    return id;
}

PYBIND11_MODULE(catalyst_callback_registry, m)
{
    if (references == nullptr) {
        references = new std::unordered_map<int64_t, py::function>();
    }
    m.doc() = "Callbacks";
    m.def("register", &registerImpl, "Call a python function registered in a map.");
}
