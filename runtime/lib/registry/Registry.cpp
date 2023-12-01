#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Types.h"

namespace py = pybind11;

std::unordered_map<uintptr_t, py::function> references;

__attribute__((visibility("default"))) void _registerImpl(uintptr_t id, py::function f)
{
    fprintf(stderr, "%ld\n", id);
    references.insert({id, f});
}

__attribute__((visibility("default"))) void callbackCall(uintptr_t identifier)
{
    fprintf(stderr, "%ld\n", identifier);
    auto it = references.find(identifier);
    auto lambda = it->second();
    // lambda();
}
