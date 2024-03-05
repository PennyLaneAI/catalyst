#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Types.h"

namespace py = pybind11;

std::unordered_map<uintptr_t, py::function> references;

extern "C" {
__attribute__((visibility("default"))) void _registerImpl(uintptr_t id, py::function f)
{
    references.insert({id, f});
}

__attribute__((visibility("default"))) void callbackCall(uintptr_t identifier)
{
    auto it = references.find(identifier);
    auto lambda = it->second();
}
}
