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

        auto block = updateRegion.getBlocks().begin();
        ValueRange arguments = block->getArguments();
        Operation *originalTerminator = block->getTerminator();

        auto updateFnType = FunctionType::get(ctx, /*inputs=*/
                                              arguments.getTypes(),
                                              /*outputs=*/originalTerminator->getOperandTypes());

        auto updateFn = b.create<func::FuncOp>(loc, funcName, updateFnType);
        updateFn.setPrivate();

        Block *entryBlock = updateFn.addEntryBlock();
        b.setInsertionPointToStart(entryBlock);
        BlockArgument inputs = updateFn.getArgument(0);
        BlockArgument update = updateFn.getArgument(1);

        block->getArgument(0).replaceAllUsesWith(inputs);
        block->getArgument(1).replaceAllUsesWith(update);
        
        for (Operation &op : block->without_terminator()) {
            op.print(llvm::outs());
            op.clone();
        }

        b.create<func::ReturnOp>(loc, inputs);
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
