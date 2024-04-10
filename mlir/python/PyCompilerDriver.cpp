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

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

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

/// Count the number of files in a workspace. Return the next available counter.
size_t initDumpCounter(const char *workspace)
{
    using namespace std;
    try {
        size_t maxDump = 0;
        auto dirIter = std::filesystem::directory_iterator(workspace);
        for (const filesystem::directory_entry& entry : dirIter) {
            if (entry.is_regular_file()) {
                maxDump = std::max(maxDump, size_t(stoul(entry.path().filename().string())));
            }
        }
        return maxDump > 0 ? maxDump + 1 : 0;
    }
    catch(std::filesystem::filesystem_error &e) {
        return 0;
    }
}

PYBIND11_MODULE(compiler_driver, m)
{
    //===--------------------------------------------------------------------===//
    // Catalyst Compiler Driver
    //===--------------------------------------------------------------------===//
    py::class_<FunctionAttributes> funcattrs_class(m, "FunctionAttributes");
    funcattrs_class.def(py::init<>())
        .def("get_function_name",
             [](const FunctionAttributes &fa) -> std::string { return fa.functionName; })
        .def("get_return_type",
             [](const FunctionAttributes &fa) -> std::string { return fa.returnType; });

    py::class_<CompilerOutput> compout_class(m, "CompilerOutput");
    compout_class.def(py::init<>())
        .def("get_pipeline_output",
             [](const CompilerOutput &co, const std::string &name) -> std::optional<std::string> {
                 auto res = co.pipelineOutputs.find(name);
                 return res != co.pipelineOutputs.end() ? res->second
                                                        : std::optional<std::string>();
             })
        .def("get_output_ir", [](const CompilerOutput &co) -> std::string { return co.outIR; })
        .def("get_object_filename",
             [](const CompilerOutput &co) -> std::string { return co.objectFilename; })
        .def("get_function_attributes",
             [](const CompilerOutput &co) -> FunctionAttributes { return co.inferredAttributes; })
        .def("get_diagnostic_messages",
             [](const CompilerOutput &co) -> std::string { return co.diagnosticMessages; });

    m.def(
        "run_compiler_driver",
        [](const char *source, const char *workspace, const char *moduleName, bool keepIntermediate,
           bool verbose, py::list pipelines,
           bool lower_to_llvm) -> std::unique_ptr<CompilerOutput> {
            std::unique_ptr<CompilerOutput> output(new CompilerOutput(initDumpCounter(workspace)));
            assert(output);

            llvm::raw_string_ostream errStream{output->diagnosticMessages};

            CompilerOptions options{.source = source,
                                    .workspace = workspace,
                                    .moduleName = moduleName,
                                    .diagnosticStream = errStream,
                                    .keepIntermediate = keepIntermediate,
                                    .verbosity = verbose ? Verbosity::All : Verbosity::Urgent,
                                    .pipelinesCfg = parseCompilerSpec(pipelines),
                                    .lowerToLLVM = lower_to_llvm};

            errStream.flush();

            if (mlir::failed(QuantumDriverMain(options, *output))) {
                throw std::runtime_error("Compilation failed:\n" + output->diagnosticMessages);
            }
            return output;
        },
        py::arg("source"), py::arg("workspace"), py::arg("module_name") = "jit source",
        py::arg("keep_intermediate") = false, py::arg("verbose") = false,
        py::arg("pipelines") = py::list(), py::arg("lower_to_llvm") = true);
}
