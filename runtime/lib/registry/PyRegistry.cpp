#include <cstdint>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <dlfcn.h>

namespace py = pybind11;

// Global object that will keep references to functions alive.
// std::unordered_map<uintptr_t, py::object> references;
//extern void _registerImpl(uintptr_t, py::function);
typedef void (*fptr_t)(uintptr_t, py::function);
void (*_registerImpl)(uintptr_t, py::function);
auto registerImpl(py::function f)
{
    // TODO: Do we need to see if it is already present
    // or can we just override it?
    // Asking in terms of memory management.
    uintptr_t id = (uintptr_t)f.ptr();
    // registry.cpython-310-x86_64-linux-gnu.so
    // is in the same rpath.
    // therefore this succeeds.
    // Let's make a small test and also test the other operating systems.
    void* handle = dlopen("registry.cpython-310-x86_64-linux-gnu.so", RTLD_LAZY | RTLD_NODELETE);
    if (!handle) { fprintf(stderr, "handle is null"); fflush(stderr); }
    _registerImpl = (fptr_t) dlsym(handle, "_registerImpl");
    if (!_registerImpl) { fprintf(stderr, "registerImpl is null"); fflush(stderr); }
    _registerImpl(id, f);
    dlclose(handle);
    return id;
}

PYBIND11_MODULE(pyregistry, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring.
    m.def("register", &registerImpl, "Call a function...");
}
