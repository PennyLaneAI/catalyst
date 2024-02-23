#include <cstdint>
#include <pybind11/pybind11.h>
#include <unordered_map>

namespace py = pybind11;

// Global object that will keep references to functions alive.
// std::unordered_map<uintptr_t, py::object> references;
//extern void _registerImpl(uintptr_t, py::function);
void (*_registerImpl)(uint_ptr_t, py::function);
auto registerImpl(py::function f)
{
    // TODO: Do we need to see if it is already present
    // or can we just override it?
    // Asking in terms of memory management.
    uintptr_t id = (uintptr_t)f.ptr();
    void* handle = dlopen("/home/ubuntu/code/catalyst/runtime/build/lib/registry.cpython-310-x86_64-linux-gnu.so", RTLD_LAZY);
    _registerImpl = dlsym(handle, "_registerImpl");
    _registerImpl(id, f);
    dlclose(handle);
    // references.insert({id, f});
    return id;
}

PYBIND11_MODULE(pyregistry, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring.
    m.def("register", &registerImpl, "Call a function...");
}
