#include <pybind11/pybind11.h>

namespace py = pybind11;

auto registerImpl(py::function f, py::args args, py::kwargs kwargs) {
	return f(args, kwargs);
}

PYBIND11_MODULE(pyregistry, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring.
	m.def("register", &registerImpl, "Call a function...");
}
