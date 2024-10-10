// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <csignal>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "Driver/CompilerDriver.h"

namespace py = pybind11;
using namespace catalyst::driver;

std::vector<Pipeline> parseCompilerSpec(const py::list &pipelines)
{
    std::vector<Pipeline> out;
    for (py::handle obj : pipelines) {
        py::tuple t = obj.cast<py::tuple>();
        auto i = t.begin();
        auto py_name = i++;
        auto py_passes = i++;
        assert(i == t.end());
        std::string name = py_name->attr("__str__")().cast<std::string>();
        Pipeline::PassList passes;
        std::transform(py_passes->begin(), py_passes->end(), std::back_inserter(passes),
                       [](py::handle p) { return p.attr("__str__")().cast<std::string>(); });
        out.push_back(Pipeline({name, passes}));
    }
    return out;
}

PYBIND11_MODULE(compiler_driver, m)
{
    //===--------------------------------------------------------------------===//
    // Catalyst Compiler Driver
    //===--------------------------------------------------------------------===//
    py::class_<CompilerOutput> compout_class(m, "CompilerOutput");
    compout_class.def(py::init<>())
        .def("get_output_ir", [](const CompilerOutput &co) -> std::string { return co.outIR; })
        .def("get_object_filename",
             [](const CompilerOutput &co) -> std::string { return co.objectFilename; })
        .def("get_diagnostic_messages",
             [](const CompilerOutput &co) -> std::string { return co.diagnosticMessages; });

    m.def(
        "run_compiler_driver",
        [](const char *source, const char *workspace, const char *moduleName, bool keepIntermediate,
           bool asyncQnodes, bool verbose, py::list pipelines, bool lower_to_llvm,
           const char *checkpointStage) -> std::unique_ptr<CompilerOutput> {
            // Install signal handler to catch user interrupts (e.g. CTRL-C).
            signal(SIGINT,
                   [](int code) { throw std::runtime_error("KeyboardInterrupt (SIGINT)"); });

            std::unique_ptr<CompilerOutput> output(new CompilerOutput());
            assert(output);

            if (QuantumDriverMainFromArgs(source, workspace, moduleName, keepIntermediate,
                                          asyncQnodes, verbose, lower_to_llvm,
                                          parseCompilerSpec(pipelines), checkpointStage, *output)) {
                throw std::runtime_error("Compilation failed:\n" + output->diagnosticMessages);
            }
            return output;
        },
        py::arg("source"), py::arg("workspace"), py::arg("module_name") = "jit source",
        py::arg("keep_intermediate") = false, py::arg("async_qnodes") = false,
        py::arg("verbose") = false, py::arg("pipelines") = py::list(),
        py::arg("lower_to_llvm") = true, py::arg("checkpoint_stage") = "");
}
