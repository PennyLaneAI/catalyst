// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace catalyst;

bool hasQnodeAttribute(func::FuncOp funcOp)
{
    static constexpr llvm::StringRef qnodeAttr = "qnode";
    return funcOp->hasAttr(qnodeAttr);
}

bool hasOutlinedAttribute(func::FuncOp funcOp)
{
    static constexpr llvm::StringRef qnodeAttr = "outlined";
    return funcOp->hasAttr(qnodeAttr);
}

void addOutlinedAttribute(func::FuncOp funcOp, PatternRewriter &rewriter)
{
    static constexpr llvm::StringRef qnodeAttr = "outlined";
    rewriter.updateRootInPlace(funcOp, [&] { funcOp->setAttr(qnodeAttr, rewriter.getUnitAttr()); });
}

namespace {
struct OutlineQuantumModuleRewritePattern : public mlir::OpRewritePattern<func::FuncOp> {
    using mlir::OpRewritePattern<func::FuncOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {

        ModuleOp parent = op->getParentOfType<ModuleOp>();
        ModuleOp grandparent = parent->getParentOfType<ModuleOp>();
        bool isValid = hasQnodeAttribute(op) && !hasOutlinedAttribute(op) && !grandparent;
        if (!isValid)
            return failure();

        auto deviceMod = rewriter.create<mlir::ModuleOp>(op.getLoc());
        IRMapping map;
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(deviceMod.getBody(), deviceMod.getBody()->end());
            Operation *cloneOp = op->clone(map);
            rewriter.insert(cloneOp);
        }

        /*
            SmallVector<Operation *, 8> symbolDefWorklist = {cloneOp};
            while (!symbolDefWorklist.empty()) {
              if (std::optional<SymbolTable::UseRange> symbolUses =
                      SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
                for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
                  StringRef symbolName =
                      cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
                  if (symbolTable.lookup(symbolName))
                    continue;

                  Operation *symbolDefClone =
                      parentSymbolTable.lookup(symbolName)->clone();
                  symbolDefWorklist.push_back(symbolDefClone);
                  symbolTable.insert(symbolDefClone);
                }
              }
            }
        */

        addOutlinedAttribute(op, rewriter);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateOutlineQuantumModulePatterns(RewritePatternSet &patterns)
{
    patterns.add<OutlineQuantumModuleRewritePattern>(patterns.getContext());
}

} // namespace catalyst
