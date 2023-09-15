// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "scatter"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mhlo/IR/hlo_ops.h"

using namespace mlir;

namespace {

struct ScatterOpRewritePattern : public mlir::OpRewritePattern<mhlo::ScatterOp> {
    using mlir::OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // Correct Block extract and make a function out of it.
        auto &reg = op.getUpdateComputation();
        if (!reg.hasOneBlock())
            return failure();
        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr updateFn =
            getOrInsertUpdateFunction(op.getLoc(), moduleOp, rewriter, reg);
        // Replace the results with the updated inputs.
        rewriter.replaceOp(op, op.getInputs());

        // Create the function
        return success();
    }

    FlatSymbolRefAttr getOrInsertUpdateFunction(Location loc, ModuleOp moduleOp, OpBuilder &b,
                                                Region &updateRegion) const
    {
        MLIRContext *ctx = b.getContext();
        std::string funcName = "__catalyst_update_scatter";

        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(moduleOp.getBody());

        Block *originalBlock = &updateRegion.front();
        Operation *originalTerminator = originalBlock->getTerminator();
        ValueRange originalArguments = originalBlock->getArguments();

        auto updateFnType = FunctionType::get(ctx, /*inputs=*/
                                              originalArguments.getTypes(),
                                              /*outputs=*/originalTerminator->getOperandTypes());

        auto updateFn = b.create<func::FuncOp>(loc, funcName, updateFnType);
        updateFn.setPrivate();

        Block *funcBody = updateFn.addEntryBlock();
        int64_t numOriginalBlockArguments = originalBlock->getNumArguments(); // it is 2

        auto outlinedFuncBlockArgs = funcBody->getArguments();
        IRRewriter rewriter(b);
        b.setInsertionPointToEnd(funcBody);
        rewriter.mergeBlocks(
            originalBlock, funcBody,
            outlinedFuncBlockArgs);

        b.setInsertionPointToEnd(funcBody);
        b.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                        originalTerminator->getOperands());
        rewriter.eraseOp(originalTerminator);
        return SymbolRefAttr::get(ctx, funcName);
    }

};

} // namespace

namespace catalyst {
namespace quantum {

void populateScatterPatterns(RewritePatternSet &patterns)
{
    patterns.add<ScatterOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
