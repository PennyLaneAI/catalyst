// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Catalyst/Driver/CompilerDriver.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

namespace py = pybind11;
using namespace mlir::python::adaptors;


std::vector<Pipeline>
parseCompilerSpec(const py::list &pipelines) {
    std::vector< Pipeline > out;
    for (py::handle obj : pipelines) {
        py::tuple t = obj.cast<py::tuple>();
        auto i = t.begin();
        auto py_name = i; i++;
        auto py_passes = i; i++;
        assert(i==t.end());
        std::string name = py_name->attr("__str__")().cast<std::string>();
        Pipeline::PassList passes;
        std::transform(py_passes->begin(), py_passes->end(), std::back_inserter(passes),
            [](py::handle p){ return p.attr("__str__")().cast<std::string>();});
        out.push_back(Pipeline({name, passes}));
    }
    return out;
}


PYBIND11_MODULE(_catalystDriver, m)
{
    //===--------------------------------------------------------------------===//
    // Catalyst Compiler Driver
    //===--------------------------------------------------------------------===//
    py::class_<FunctionAttributes> funcattrs_class(m, "FunctionAttributes");
    funcattrs_class.def(py::init<>())
        .def("getFunctionName", [](const FunctionAttributes &fa) -> std::string {
            return fa.functionName;
        })
        .def("getReturnType", [](const FunctionAttributes &fa) -> std::string {
            return fa.returnType;
        })
        ;

    py::class_<CompilerOutput> compout_class(m, "CompilerOutput");
    compout_class.def(py::init<>())
        .def("getPipelineOutput", [](const CompilerOutput &co, const std::string &name) -> std::string {
            auto res = co.pipelineOutputs.find(name);
            return res != co.pipelineOutputs.end() ? res->second : "";
        })
        .def("getOutputIR", [](const CompilerOutput &co) -> std::string {
            return co.outIR;
        })
        .def("getObjectFilename", [](const CompilerOutput &co) -> std::string {
            return co.objectFilename;
        })
        .def("getFunctionAttributes", [](const CompilerOutput &co) -> FunctionAttributes {
            return co.inferredAttributes;
        })
        ;

    m.def(
        "compile_asm",
        [](const char *source, const char *workspace, const char *moduleName,
           bool inferFunctionAttrs, bool keepIntermediate, bool verbose,
           py::list pipelines, bool attemptLLVMLowering) //-> CompilerOutput *
        {
            FunctionAttributes inferredAttributes;
            mlir::MLIRContext ctx;
            std::string errors;
            Verbosity verbosity = verbose ? CO_VERB_ALL : CO_VERB_SILENT;
            llvm::raw_string_ostream errStream{errors};

            CompilerSpec spec{.pipelinesCfg = parseCompilerSpec(pipelines),
                              .attemptLLVMLowering = attemptLLVMLowering };
            CompilerOptions options{.ctx = &ctx,
                                    .source = source,
                                    .workspace = workspace,
                                    .moduleName = moduleName,
                                    .diagnosticStream = errStream,
                                    .keepIntermediate = keepIntermediate,
                                    .verbosity = verbosity};

            CompilerOutput *output = new CompilerOutput();
            assert(output);

            if (mlir::failed(QuantumDriverMain(spec, options, *output))) {
                throw std::runtime_error("Compilation failed:\n" + errors);
            }
            if (verbosity > CO_VERB_SILENT && !errors.empty()) {
                py::print(errors);
            }

            return output;
        },
        py::arg("source"), py::arg("workspace"), py::arg("module_name") = "jit source",
        py::arg("infer_function_attrs") = false, py::arg("keep_intermediate") = false,
        py::arg("verbose") = false, py::arg("pipelines") = py::list(),
        py::arg("attemptLLVMLowering") = true);
}
