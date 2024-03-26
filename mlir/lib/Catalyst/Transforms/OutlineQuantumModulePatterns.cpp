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

bool hasOutlinedAttribute(func::CallOp op)
{
    static constexpr llvm::StringRef payload = "payload";
    return op->hasAttr(payload);
}

void addOutlinedAttribute(func::CallOp op, ModuleOp module, PatternRewriter &rewriter)
{
    auto payload = rewriter.getStringAttr("payload");
    auto symRefAttr = FlatSymbolRefAttr::get(rewriter.getStringAttr(module.getSymName().value()));
    auto namedAttribute = NamedAttribute(payload, symRefAttr);
    rewriter.updateRootInPlace(op, [&] { op->setAttrs({namedAttribute}); });
}

namespace {
struct OutlineQuantumModuleRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::CallOp callOp,
                                        mlir::PatternRewriter &rewriter) const override
    {

        auto callee = callOp.getCalleeAttr();
        auto op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callee);
        ModuleOp parent = op->getParentOfType<ModuleOp>();
        ModuleOp grandparent = parent->getParentOfType<ModuleOp>();
        bool isValid = hasQnodeAttribute(op) && !hasOutlinedAttribute(callOp) && !grandparent;
        if (!isValid)
            return failure();

        SymbolTable parentSymbolTable(parent);
        auto newName = std::string("payload.") + std::string(op.getSymName().data());
        auto deviceMod = rewriter.create<mlir::ModuleOp>(op.getLoc(), newName);
        SymbolTable symbolTable(deviceMod);

        IRMapping map;
        Operation *cloneOp = op->clone(map);
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(deviceMod.getBody(), deviceMod.getBody()->end());
            rewriter.insert(cloneOp);
        }

        SmallVector<Operation *, 8> symbolDefWorklist = {cloneOp};
        while (!symbolDefWorklist.empty()) {
            if (std::optional<SymbolTable::UseRange> symbolUses =
                    SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
                for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
                    StringRef symbolName =
                        cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
                    if (symbolTable.lookup(symbolName))
                        continue;

                    Operation *symbolDefClone = parentSymbolTable.lookup(symbolName)->clone();
                    symbolDefWorklist.push_back(symbolDefClone);
                    symbolTable.insert(symbolDefClone);
                }
            }
        }

        auto exec =
            rewriter.create<DeviceExecuteOp>(op.getLoc(), callOp.getResultTypes(),
                                             deviceMod.getSymName().value(), SmallVector<Value>());
        rewriter.replaceOp(callOp, exec);
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
