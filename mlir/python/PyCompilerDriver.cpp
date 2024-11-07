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
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "Driver/CompilerDriver.h"

namespace nb = nanobind;
using namespace catalyst::driver;

std::vector<Pipeline> parseCompilerSpec(const nb::list &pipelines)
{
    std::vector<Pipeline> out;
    for (nb::handle obj : pipelines) {
        nb::tuple t = nb::cast<nb::tuple>(obj);
        assert(nb::len(t) == 2);
        std::string name = nb::cast<std::string>(t[0]);
        nb::list py_passes = t[1];
        llvm::SmallVector<std::string> passes;
        std::transform(py_passes.begin(), py_passes.end(), std::back_inserter(passes),
                       [](nb::handle p) { return nb::cast<std::string>(p.attr("__str__")()); });
        Pipeline pipeline;
        pipeline.setName(name);
        pipeline.setPasses(passes);
        out.push_back(pipeline);
    }
    return out;
}

NB_MODULE(compiler_driver, m)
{
    //===--------------------------------------------------------------------===//
    // Catalyst Compiler Driver
    //===--------------------------------------------------------------------===//
    nb::class_<CompilerOutput> compout_class(m, "CompilerOutput");
    compout_class.def(nb::init<>())
        .def("get_output_ir", [](const CompilerOutput &co) -> std::string { return co.outIR; })
        .def("get_object_filename",
             [](const CompilerOutput &co) -> std::string { return co.outputFilename; })
        .def("get_diagnostic_messages",
             [](const CompilerOutput &co) -> std::string { return co.diagnosticMessages; });

    m.def(
        "run_compiler_driver",
        [](const char *source, const char *workspace, const char *moduleName, bool keepIntermediate,
           bool asyncQnodes, bool verbose, nb::list pipelines, bool lower_to_llvm,
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
        nb::arg("source"), nb::arg("workspace"), nb::arg("module_name") = "jit source",
        nb::arg("keep_intermediate") = false, nb::arg("async_qnodes") = false,
        nb::arg("verbose") = false, nb::arg("pipelines") = nb::list(),
        nb::arg("lower_to_llvm") = true, nb::arg("checkpoint_stage") = "");
}
