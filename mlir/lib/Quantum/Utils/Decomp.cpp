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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "Quantum/Utils/Decomp.h"

namespace catalyst {
namespace quantum {

std::vector<mlir::OwningOpRef<mlir::func::FuncOp>> getRulesFromBytecode(llvm::StringRef filename,
                                                                        mlir::MLIRContext *context)
{
    llvm::errs() << "getting rules from bytecode\n";
    mlir::ParserConfig config(context);
    llvm::errs() << "got parser config from context\n";
    llvm::errs() << filename << "\n";
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
    llvm::errs() << "parsed file\n";

    std::vector<mlir::OwningOpRef<mlir::func::FuncOp>> funcOps;

    if (!moduleOp) {
        llvm::errs() << "failed to find module\n";
        return funcOps;
    }

    llvm::errs() << "collecting funcops\n";
    for (auto func : moduleOp->getOps<mlir::func::FuncOp>()) {
        func->remove();
        funcOps.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(func));
    }

    llvm::errs() << "got funcops\n";

    return funcOps;
}

} // namespace quantum
} // namespace catalyst
