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

#include <cstdint>
#include <string>

#include "catch2/catch_test_macros.hpp"

#include "pybind11/embed.h"
#include "pybind11/numpy.h"

// steane_decoder.hpp is a GPU-side header that marks its decode helper with
// `__device__`. The constexpr STEANE_SYNDROME_TO_QUBIT table is usable from the
// host, so we shim `__device__` to an empty token to make the header parse in a
// host-only test translation unit.
#ifndef __device__
#define __device__
#define CATALYST_TEST_DEFINED_DEVICE_SHIM
#endif
#include "steane_decoder.hpp"
#ifdef CATALYST_TEST_DEFINED_DEVICE_SHIM
#undef __device__
#undef CATALYST_TEST_DEFINED_DEVICE_SHIM
#endif

namespace py = pybind11;

namespace {

// pybind11 owns interpreter startup; nanobind explicitly does not support
// embedding one, per Test_OpenQasmDevice.cpp.
void ensurePythonInterpreter()
{
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
}

} // namespace

// STEANE_SYNDROME_TO_QUBIT is a hard-coded [[7,1,3]] Steane decoder living in
// the transport backend (steane_decoder.hpp, mirrored in steane_decoder_fn.cpp).
// The Steane code itself is defined in the frontend at qec_code_lib.py via the
// stabilizer matrix ``H = _CODE_REGISTRY["Steane"][3]`` (Hx == Hz for Steane).
// A correct single-error decoder must send:
//   - syndrome 000 -> "no error" (-1), and
//   - the syndrome ``H[:, i]`` -> qubit ``i``, for each data qubit ``i``.
// This test enforces exactly that against the frontend definition, and SKIPs
// when the frontend module is not importable (envs without catalyst installed).
TEST_CASE("STEANE_SYNDROME_TO_QUBIT decodes the Steane code from qec_code_lib.py",
          "[steane_decoder][frontend]")
{
    ensurePythonInterpreter();

    py::object registry_obj;
    try {
        py::module_ code_lib = py::module_::import(
            "catalyst.python_interface.transforms.qecp.qec_code_lib");
        registry_obj = code_lib.attr("_CODE_REGISTRY");
    }
    catch (const py::error_already_set &e) {
        SKIP(std::string("catalyst frontend not importable in this environment: ") + e.what());
    }

    py::tuple steane = registry_obj.cast<py::dict>()["Steane"].cast<py::tuple>();
    // Layout of _CODE_REGISTRY["Steane"]: (n, k, d, x_tanner, z_tanner, ...).
    auto H_arr =
        steane[3].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
    auto H = H_arr.unchecked<2>();
    const auto n_aux = H.shape(0);
    const auto n_data = H.shape(1);

    using catalyst::transport::gpu_verbs::STEANE_CHECKS;
    using catalyst::transport::gpu_verbs::STEANE_SYNDROME_TO_QUBIT;
    REQUIRE(n_aux == STEANE_CHECKS);

    REQUIRE(STEANE_SYNDROME_TO_QUBIT[0] == -1);

    // steane_decode packs check 0 as the MSB of the 3-bit index (see
    // steane_decoder.hpp:steane_decode); mirror that packing here.
    for (py::ssize_t i = 0; i < n_data; ++i) {
        int syndrome_int = 0;
        for (py::ssize_t c = 0; c < n_aux; ++c) {
            syndrome_int = (syndrome_int << 1) | static_cast<int>(H(c, i) & 1);
        }
        REQUIRE(STEANE_SYNDROME_TO_QUBIT[syndrome_int] == i);
    }
}
