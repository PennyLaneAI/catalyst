// Copyright 2024 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Exception.hpp"

std::string program = R"(

try:
    import os
    email = os.environ.get("OQC_EMAIL")
    password = os.environ.get("OQC_PASSWORD")
    url = os.environ.get("OQC_URL")
    if not all([email, password, url]):
        raise ValueError(
            """
            OQC credentials not found in environment variables.
            Please set the environment variables `OQC_EMAIL`, `OQC_PASSWORD` and `OQC_URL`.
            """
        )
    import qcaas_client
    from qcaas_client.client import OQCClient, QPUTask, CompilerConfig, QuantumResultsFormat
    from qcaas_client.compiler_config import Tket, TketOptimizations
    optimisations = Tket()
    optimisations.tket_optimizations = TketOptimizations.DefaultMappingPass
    RES_FORMAT = QuantumResultsFormat().binary_count()
    client = OQCClient(url=url, email=email, password=password)
    client.authenticate()
    oqc_config = CompilerConfig(repeats=shots, results_format=RES_FORMAT, optimizations=optimisations)
    oqc_task = QPUTask(program=circuit, config=oqc_config, qpu_id=qpu_id)
    res = client.execute_tasks(oqc_task, qpu_id=qpu_id)
    counts = res[0].result["cbits"]
except Exception as e:
    print(f"circuit: {circuit}")
    msg = str(e)
)";

extern "C" {
[[gnu::visibility("default")]] int counts(const char *_circuit, const char *_qpu_id, size_t shots,
                                          size_t num_qubits, const char *_kwargs, void *_vector,
                                          char *error_msg, size_t error_msg_size)
{
    namespace py = pybind11;
    using namespace py::literals;

    py::gil_scoped_acquire lock;

    py::dict locals;
    locals["circuit"] = _circuit;
    locals["qpu_id"] = _qpu_id;
    locals["kwargs"] = _kwargs;
    locals["shots"] = shots;
    locals["num_qubits"] = num_qubits;
    locals["msg"] = "";

    // Evaluate in scope of main module
    py::object scope = py::module_::import("__main__").attr("__dict__");
    py::exec(py::str(program.c_str()), scope, locals);

    auto msg = py::cast<std::string>(locals["msg"]);

    if (!msg.empty()) {
        size_t copy_len = std::min(msg.length(), error_msg_size - 1);
        memcpy(error_msg, msg.c_str(), copy_len);
        error_msg[copy_len] = '\0';
        return -1;
    }

    // Process counts only if we didn't have credential issues
    py::dict results = locals["counts"];

    std::vector<size_t> *counts_value = reinterpret_cast<std::vector<size_t> *>(_vector);
    for (auto item : results) {
        auto key = item.first;
        auto value = item.second;
        counts_value->push_back(py::cast<size_t>(value));
    }

    return 0; // Success
}
} // extern "C"

PYBIND11_MODULE(oqc_python_module, m) { m.doc() = "oqc"; }
