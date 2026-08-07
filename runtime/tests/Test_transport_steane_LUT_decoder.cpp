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

// pybind11 rather than nanobind: nanobind does not support embedding an
// interpreter (see Test_OpenQasmDevice.cpp).
void ensurePythonInterpreter() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
}

} // namespace

// Cross-check STEANE_SYNDROME_TO_QUBIT (transport/common/SteaneDecoderTable.hpp)
// against the frontend's Steane parity check matrix in qec_code_lib.py.
TEST_CASE("STEANE_SYNDROME_TO_QUBIT decodes the Steane code from qec_code_lib.py",
          "[steane_decoder][frontend]") {
    ensurePythonInterpreter();

    py::module_ code_lib =
        py::module_::import("catalyst.python_interface.transforms.qecp.qec_code_lib");
    REQUIRE(py::hasattr(code_lib, "_CODE_REGISTRY"));
    py::dict registry = code_lib.attr("_CODE_REGISTRY").cast<py::dict>();
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
