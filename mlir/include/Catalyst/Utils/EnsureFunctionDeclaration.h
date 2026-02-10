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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
using namespace mlir;

namespace catalyst {

// When lowering custom dialects, we often need to generate calls to runtime CAPI functions.
// This utility function generates declarations for these functions if they do not exist.
//
// It supports both:
// 1. LLVM::LLVMFuncOp (for LLVM dialect lowering)
// 2. func::FuncOp (for standard MLIR lowering, marks visibility as private)
template <typename OpT, typename TypeT>
OpT ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op, StringRef fnSymbol,
                              TypeT fnType)
{
    // Lookup the symbol to see if it already exists
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        // If not found, insert it at the start of the Module
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        // Create the specific function operation (LLVMFuncOp or FuncOp)
        auto newFunc = OpT::create(rewriter, op->getLoc(), fnSymbol, fnType);

        // Handle visibility differences:
        // func::FuncOp usually requires explicit private visibility for runtime decls.
        if (isa<func::FuncOp>(newFunc)) {
            newFunc.setPrivate();
        }

        fnDecl = newFunc;
    }
    else {
        // Verify the existing symbol is the correct type
        assert(isa<OpT>(fnDecl) && "Existing symbol is not the expected operation type");
    }

    return cast<OpT>(fnDecl);
}

} // namespace catalyst
