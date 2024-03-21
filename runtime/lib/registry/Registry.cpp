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

extern "C" {
[[gnu::visibility("default")]] void callbackCall(int64_t identifier)
{
    py::gil_scoped_acquire lock;
    auto it = references->find(identifier);
    if (it == references->end()) {
        throw std::invalid_argument("Callback called with invalid identifier");
    }
    auto lambda = it->second;
    lambda();
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
