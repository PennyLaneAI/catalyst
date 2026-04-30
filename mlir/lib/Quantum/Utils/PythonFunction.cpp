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

#include "Quantum/Utils/PythonFunction.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h" // for automatic vector + variant conversion

namespace py = pybind11;
using namespace pybind11::literals;

namespace catalyst {
namespace quantum {

/**
 * @brief trace and lower a python-defined circuit, returning the textual MLIR.
 *
 * @param moduleName: the name of the module containing the python function
 * @param functionName: the name of the python function to execute
 * @param args: the arguments to the python function
 * @return textual MLIR of the traced function
 */
std::string tracePauliRotDecomp(llvm::StringRef moduleName, llvm::StringRef functionName,
                                std::vector<PyArg> args, PyWires wires)
{
    py::gil_scoped_acquire acquire;

    try {
        py::module_ userModule = py::module_::import(moduleName.str().c_str());
        py::object userFunction = userModule.attr(functionName.str().c_str());

        py::module_ wrapperModule = py::module_::import("catalyst.python_callbacks");
        py::object wrapperFunction = wrapperModule.attr("paulirot_callback_wrapper");

        py::list pyArgs;
        for (auto arg : args) {
            pyArgs.append(py::cast(arg));
        }

        py::object pythonResult = wrapperFunction(*pyArgs, "wires"_a = py::cast(wires));
        return pythonResult.cast<std::string>();
    }
    catch (const py::error_already_set &error) {
        llvm::errs() << "Python error occurred: " << error.what() << "\n";
        return "";
    }
}

mlir::OwningOpRef<mlir::func::FuncOp> lowerPauliRotDecomp(mlir::ModuleOp module,
                                                          llvm::StringRef moduleName,
                                                          llvm::StringRef functionName,
                                                          std::vector<PyArg> args, PyWires wires)
{
    std::string result = tracePauliRotDecomp(moduleName, functionName, args, wires);

    llvm::StringRef resultRef = result;

    mlir::ParserConfig config = mlir::ParserConfig(module.getContext());
    auto moduleOp = mlir::parseSourceString(resultRef, config);

    if (!moduleOp) {
        llvm::errs() << "failed to parse python output, returning null\n";
        return nullptr;
    }

    // get the lowered funcop from the python result
    mlir::OwningOpRef<mlir::func::FuncOp> funcOp;
    moduleOp->walk([&](mlir::func::FuncOp func) {
        if (func.getName() == functionName) {
            func->remove();
            funcOp = mlir::OwningOpRef<mlir::func::FuncOp>(func);
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    return funcOp;
}

} // namespace quantum
} // namespace catalyst
