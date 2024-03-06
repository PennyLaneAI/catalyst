#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Types.h"

namespace py = pybind11;

// From PyBind11's documentation:
//
//     Do you have any global variables that are pybind11 objects or invoke pybind11 functions in
//     either their constructor or destructor? You are generally not allowed to invoke any Python
//     function in a global static context. We recommend using lazy initialization and then
//     intentionally leaking at the end of the program.
//
// https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
std::unordered_map<uintptr_t, py::function> *references;

extern "C" {
[[gnu::visibility("default")]] void _registerImpl(uintptr_t id, py::function f)
{
    if (references == nullptr)
        references = new std::unordered_map<uintptr_t, py::function>();
    // TODO: memory de-reference?
    references->insert({id, f});
}

[[gnu::visibility("default")]] void callbackCall(uintptr_t identifier)
{
    auto it = references->find(identifier);
    auto lambda = it->second;
    lambda();
}
}
