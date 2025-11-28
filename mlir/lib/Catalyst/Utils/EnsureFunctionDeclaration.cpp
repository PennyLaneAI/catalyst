// Copyright 2025 Xanadu Quantum Technologies Inc.

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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace catalyst {

LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

func::FuncOp ensurefuncOrDeclare(PatternRewriter &rewriter, Operation *op, StringRef fnSymbol,
                                 FunctionType fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);

        // FIX: Check if 'op' is already a Module, otherwise find the parent.
        ModuleOp mod = dyn_cast<ModuleOp>(op);
        if (!mod) {
            mod = op->getParentOfType<ModuleOp>();
        }

        // Fallback if we somehow have no module (shouldn't happen in valid IR)
        assert(mod && "Could not find a valid ModuleOp to insert function declaration");

        rewriter.setInsertionPointToStart(mod.getBody());

        auto funcOp = rewriter.create<func::FuncOp>(op->getLoc(), fnSymbol, fnType);
        funcOp.setPrivate();
        fnDecl = funcOp;
    }
    else {
        assert(isa<func::FuncOp>(fnDecl) && "Function declaration is not a func::FuncOp");
    }

    return cast<func::FuncOp>(fnDecl);
}

} // namespace catalyst
