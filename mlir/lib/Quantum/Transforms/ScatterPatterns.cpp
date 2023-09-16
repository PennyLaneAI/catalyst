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
#include <iostream>
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
        // Add checks for supported cases
        if (!op.getUniqueIndices() || !op.getIndicesAreSorted() ||
            !op.getScatterDimensionNumbers().getUpdateWindowDims().empty()) {
            return failure();
        }

        op.getScatterDimensionNumbers();
        // Extract the block responsible for update
        auto &region = op.getUpdateComputation();

        if (!region.hasOneBlock())
            return failure();

        auto moduleOp = op->getParentOfType<ModuleOp>();

        // We create a function from the update block
        FlatSymbolRefAttr updateFn =
            getOrInsertUpdateFunction(op.getLoc(), moduleOp, rewriter, region);

        auto results = op.getInputs();
        auto resultsOperand = results.front();
        ArrayRef<int64_t> resultsShape = resultsOperand.getType().cast<TensorType>().getShape();
        std::vector<int> resultShapeVector(resultsShape.begin(), resultsShape.end());

        auto updates = op.getUpdates();
        auto scatterIndices = op.getScatterIndices();

        // for loop over the update indices
        std::vector<int> indices;

        // Start generating indices from dimension 0
        std::cout << "before";
        std::vector<std::vector<int>> configurations;
        generateIndicesRecursive(resultShapeVector, indices, 0, configurations);
        std::cout << "after";
        // Replace the results with the updated inputs.
        rewriter.replaceOp(op, results);

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

        auto outlinedFuncBlockArgs = funcBody->getArguments();
        IRRewriter rewriter(b);
        b.setInsertionPointToEnd(funcBody);
        rewriter.mergeBlocks(originalBlock, funcBody, outlinedFuncBlockArgs);

        b.setInsertionPointToEnd(funcBody);
        b.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                 originalTerminator->getOperands());
        rewriter.eraseOp(originalTerminator);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void generateIndicesRecursive(const std::vector<int> &shape, std::vector<int> &currentIndex,
                                  int dimension,  std::vector<std::vector<int>>& configurations) const
    {
        if (dimension == shape.size()) {
            // Base case: Print the current index
            configurations.push_back(currentIndex);
        }
        else {
            for (int i = 0; i < shape[dimension]; i++) {
                currentIndex.push_back(i);
                generateIndicesRecursive(shape, currentIndex, dimension + 1, configurations);
                currentIndex.pop_back();
            }
        }
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
