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

#include "PythonFunction.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h" // for automatic vector + variant conversion

#include "PythonDriverUtils.hpp"

namespace py = pybind11;

namespace QuantumPythonCallbacks {

std::string tracePauliRotDecomp(double theta, std::string pauliWord, PyWires wires)
{
    py::gil_scoped_acquire acquire;
    const char *moduleName = "catalyst.python_callbacks";
    const char *functionName = "paulirot_callback_wrapper";

    try {
        py::module_ wrapperModule = py::module_::import(moduleName);
        py::object wrapperFunction = wrapperModule.attr(functionName);
        py::object pythonResult = wrapperFunction(theta, pauliWord, wires);
        return pythonResult.cast<std::string>();
    }
    catch (const py::error_already_set &error) {
        throw TracingError(moduleName, functionName, pauliWord, error.what());
    }
}

mlir::OwningOpRef<mlir::func::FuncOp> lowerPauliRotDecomp(mlir::ModuleOp module, double theta,
                                                          std::string pauliWord, PyWires wires)
{
    std::string result = tracePauliRotDecomp(theta, pauliWord, wires);

    llvm::StringRef resultRef = result;

    mlir::ParserConfig config = mlir::ParserConfig(module.getContext());
    auto moduleOp = mlir::parseSourceString(resultRef, config);

    if (!moduleOp) {
        return nullptr;
    }

    // get the lowered funcop from the python result
    mlir::OwningOpRef<mlir::func::FuncOp> funcOp;
    moduleOp->walk([&](mlir::func::FuncOp func) {
        if (func.getName() == "_pauli_rot_decomposition") {
            func->remove();
            funcOp = mlir::OwningOpRef<mlir::func::FuncOp>(func);
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    return funcOp;
}

} // namespace QuantumPythonCallbacks
