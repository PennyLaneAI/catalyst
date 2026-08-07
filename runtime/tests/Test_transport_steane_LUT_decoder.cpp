// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pybind11/embed.h"
#include "pybind11/numpy.h"

#include <cstdint>

#include "catch2/catch_test_macros.hpp"

#include "SteaneDecoderTable.hpp"

namespace py = pybind11;

namespace {

// pybind11 (nanobind does not support embedding). Setting PyConfig.program_name
// to CMake's Python_EXECUTABLE routes Python's venv detection to that binary,
// so sys.prefix and site-packages match the interpreter CMake found.
void ensurePythonInterpreter() {
    if (Py_IsInitialized()) {
        return;
    }
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyStatus status = PyConfig_SetBytesString(&config, &config.program_name,
                                              EMBEDDED_PYTHON_EXECUTABLE);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        throw std::runtime_error("PyConfig_SetBytesString(program_name) failed");
    }
    py::initialize_interpreter(&config);
}

} // namespace

// Cross-check STEANE_SYNDROME_TO_QUBIT (transport/common/SteaneDecoderTable.hpp)
// against the frontend's Steane parity check matrix. Loads the numpy-only leaf
// _code_registry.py by path so catalyst's package __init__ chain never runs.
TEST_CASE("STEANE_SYNDROME_TO_QUBIT decodes the Steane code from qec_code_lib.py",
          "[steane_decoder][frontend]") {
    ensurePythonInterpreter();

    py::module_ importlib_util = py::module_::import("importlib.util");
    py::object spec = importlib_util.attr("spec_from_file_location")(
        "_steane_code_registry", STEANE_CODE_REGISTRY_PATH);
    REQUIRE(!spec.is_none());
    py::module_ code_registry = importlib_util.attr("module_from_spec")(spec);
    spec.attr("loader").attr("exec_module")(code_registry);

    REQUIRE(py::hasattr(code_registry, "_CODE_REGISTRY"));
    py::dict registry = code_registry.attr("_CODE_REGISTRY").cast<py::dict>();
    REQUIRE(registry.contains("Steane"));

    // _CODE_REGISTRY["Steane"] layout: (n, k, d, x_tanner, z_tanner, ...); Hx == Hz.
    py::tuple steane = registry["Steane"].cast<py::tuple>();
    auto H_arr = steane[3].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
    auto H = H_arr.unchecked<2>();
    const auto n_aux = H.shape(0);
    const auto n_data = H.shape(1);

    using catalyst::transport::common::STEANE_CHECKS;
    using catalyst::transport::common::STEANE_SYNDROME_TO_QUBIT;
    REQUIRE(n_aux == STEANE_CHECKS);
    REQUIRE(STEANE_SYNDROME_TO_QUBIT[0] == -1);

    // Pack check 0 as the MSB, matching the transport backends' decoder path.
    for (py::ssize_t i = 0; i < n_data; ++i) {
        int syndrome_int = 0;
        for (py::ssize_t c = 0; c < n_aux; ++c) {
            syndrome_int = (syndrome_int << 1) | static_cast<int>(H(c, i) & 1);
        }
        REQUIRE(STEANE_SYNDROME_TO_QUBIT[syndrome_int] == i);
    }
}
