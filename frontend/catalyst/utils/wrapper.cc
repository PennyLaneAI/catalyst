#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void wrap(py::object func, py::tuple py_args)
{
    size_t length = py_args.attr("__len__")().cast<size_t>();
    auto ctypes = py::module::import("ctypes");
    if (length != 2) {
        throw std::runtime_error("Invalid number of arguments.");
    }

    typedef void (*f_ptr_t)(void *, void *);
    f_ptr_t f_ptr = *((f_ptr_t *)ctypes.attr("addressof")(func).cast<size_t>());

    auto value0 = py_args.attr("__getitem__")(0);
    void *value0_ptr = *(void **)ctypes.attr("addressof")(value0).cast<size_t>();
    auto value1 = py_args.attr("__getitem__")(1);
    void *value1_ptr = *(void **)ctypes.attr("addressof")(value1).cast<size_t>();

    f_ptr(value0_ptr, value1_ptr);
}

PYBIND11_MODULE(wrapper, m)
{
    m.doc() = "wrapper module";
    m.def("wrap", &wrap, "A wrapper function.");
}
