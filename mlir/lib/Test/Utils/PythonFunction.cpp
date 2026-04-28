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

#include "Test/Utils/PythonFunction.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h" // for automatic vector + variant conversion

namespace py = pybind11;
using namespace pybind11::literals;

namespace catalyst {
namespace test {

/*
 * @brief execute a python function and return a string result.
 *
 * @param module_name: the name of the module containing the python function.
 * @param function_name: the name of the python function to execute.
 * @param args: the arguments to the python function.
 */
std::string python_circuit_execution(llvm::StringRef module_name, llvm::StringRef function_name,
                                     std::vector<PyArg> args, PyWires wires)
{
    py::gil_scoped_acquire acquire;

    try {
        py::module_ user_module = py::module_::import(module_name.str().c_str());
        py::object user_function = user_module.attr(function_name.str().c_str());

        py::module_ wrapper_module = py::module_::import("catalyst.utils.python_callbacks");
        py::object wrapper_function = wrapper_module.attr("callback_wrapper");

        llvm::errs() << "got python function and module\n";
        py::list py_args;
        for (auto arg : args) {
            py_args.append(py::cast(arg));
        }
        llvm::errs() << "cast args\n";

        llvm::errs() << "calling " << module_name << "." << function_name << "(";
        py::print(*py_args, wires, user_function, 1, "sep"_a = ", ", "end"_a = ")\n");

        py::object python_result =
            wrapper_function(*py_args, "op_wires"_a = py::cast(wires),
                             "decomp_rule"_a = user_function, "static_argnums"_a = 1);
        llvm::errs() << "got result from python\n";
        return python_result.cast<std::string>();
    }
    catch (const py::error_already_set &error) {
        llvm::errs() << "Python error occurred: " << error.what() << "\n";
        return "";
    }
}

mlir::OwningOpRef<mlir::Operation *> get_op_from_python(mlir::ModuleOp module,
                                                        llvm::StringRef module_name,
                                                        llvm::StringRef function_name,
                                                        std::vector<PyArg> args, PyWires wires)
{
    std::string result = python_circuit_execution(module_name, function_name, args, wires);
    llvm::errs() << "got python result: " << result << "\n";

    llvm::StringRef resultRef = result;

    // parse and insert the returned IR
    mlir::ParserConfig config = mlir::ParserConfig(module.getContext());
    auto outOp = mlir::parseSourceString(resultRef, config);

    // if parsing failed, return the null op from the parser
    if (!outOp) {
        llvm::errs() << "failed to parse python output, returning null\n";
        return outOp;
    }

    llvm::errs() << "successfully parsed python result\n";

    // get the lowered function from the python result
    outOp->walk([&](mlir::func::FuncOp func) {
        if (func.getName() == function_name) {
            llvm::errs() << "found matching function!\n";
            func->remove();
            outOp = mlir::OwningOpRef<mlir::func::FuncOp>(func);
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    return outOp;
}

} // namespace test
} // namespace catalyst
