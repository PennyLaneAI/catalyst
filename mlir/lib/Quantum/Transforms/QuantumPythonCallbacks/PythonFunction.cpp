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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser/Parser.h"
#include "nanobind/nanobind.h"

#include "Quantum/Transforms/DecompCallbacks.h"

#include "PythonDriverUtils.hpp"

#define DEBUG_TYPE "[QPC] "

namespace nb = nanobind;

namespace {

mlir::OwningOpRef<mlir::func::FuncOp> lowerPauliRotImpl(mlir::MLIRContext *ctx, double theta,
                                                        const std::string &pauliWord,
                                                        llvm::ArrayRef<int> wires)
{
    std::vector<int> wiresVec(wires.begin(), wires.end());

    // Invoke Python and parse the returned MLIR text.
    std::string mlirText = QuantumPythonCallbacks::PyInterpreterGuard::ensure().withGil([&] {
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "lowering paulirot with pauliword " << pauliWord
                                << "\n");
        const char *moduleName = "catalyst.python_callbacks";
        const char *functionName = "paulirot_callback_wrapper";

        try {

            LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Importing the python module...\n");
            nb::module_ wrapperModule = nb::module_::import_(moduleName);

            LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Getting the python function...\n");
            // nb::object wrapperFunction = wrapperModule.attr("test_function");
            nb::object wrapperFunction = wrapperModule.attr(functionName);

            nb::list pyWires;
            for (int w : wires) {
                pyWires.append(w);
            }

            LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "executing python function...\n");
            nb::object pythonResult = wrapperFunction(theta, pauliWord.c_str(), pyWires);

            // LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Executing test function...\n");
            // nb::object pythonResult = wrapperFunction();

            LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Got function output, casting...\n");

            std::string output = nb::cast<const char *>(pythonResult);
            // llvm::errs() << "[TEST OUTPUT] " << output << "\n";

            return std::string(output);

            // LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "casting result...\n");
            // return nb::cast<std::string>(pythonResult);
        }
        catch (const nb::python_error &error) {
            throw QuantumPythonCallbacks::TracingError(moduleName, functionName, pauliWord,
                                                       error.what());
        }
    });

    mlir::ParserConfig config(ctx);
    auto moduleOp = mlir::parseSourceString(llvm::StringRef(mlirText), config);
    if (!moduleOp) {
        return nullptr;
    }

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

// Exception-safe entry point installed in the registry. Converts any failure
// to an MLIR diagnostic + nullptr return so the pass can signalPassFailure.
mlir::OwningOpRef<mlir::func::FuncOp> pythonLowerPauliRot(mlir::MLIRContext *ctx, double theta,
                                                          const std::string &pauliWord,
                                                          llvm::ArrayRef<int> wires)
{
    try {
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "lowering paulirot with pauliword " << pauliWord
                                << "\n");
        return lowerPauliRotImpl(ctx, theta, pauliWord, wires);
    }
    catch (const std::exception &e) {
        mlir::emitError(mlir::UnknownLoc::get(ctx))
            << "PauliRot decomposition callback failed for pauli_word='" << pauliWord
            << "': " << e.what();
        return nullptr;
    }
    catch (...) {
        mlir::emitError(mlir::UnknownLoc::get(ctx))
            << "PauliRot decomposition callback failed for pauli_word='" << pauliWord
            << "': unknown exception";
        return nullptr;
    }
}

} // namespace

// The default visibility keeps the symbol in the .so's dynamic table even under
// -fvisibility=hidden. C linkage so the loader can resolve by plain name.
extern "C" __attribute__((visibility("default"))) void registerPythonDecompCallback()
{
    catalyst::quantum::registerLowerPauliRot(pythonLowerPauliRot);
}
