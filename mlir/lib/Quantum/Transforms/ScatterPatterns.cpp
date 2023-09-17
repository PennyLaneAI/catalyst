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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
        Location loc = op.getLoc();
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
        auto resultsValue = results.front();
        ArrayRef<int64_t> resultsShape = resultsValue.getType().cast<TensorType>().getShape();
        std::vector<int64_t> resultShapeVector(resultsShape.begin(), resultsShape.end());

        auto updates = op.getUpdates();
        auto updatesValue = updates.front();

        auto scatterIndices = op.getScatterIndices();
        auto indexVectorDim = op.getScatterDimensionNumbers().getIndexVectorDim();
        auto scatterDimsToOperandDims =
            op.getScatterDimensionNumbers().getScatterDimsToOperandDims();

        // Start generating indices from dimension 0
        std::vector<int64_t> indices;
        std::vector<std::vector<int64_t>> allUpdatesIndices;
        generateIndicesRecursive(resultShapeVector, indices, 0, allUpdatesIndices);

        // Replace the results with the updated inputs.
        for (auto updatesIndices : allUpdatesIndices) {
            SmallVector<Value> updatesIndicesValue;
            for (auto index : updatesIndices) {
                Value value = rewriter.create<index::ConstantOp>(loc, index);
                updatesIndicesValue.push_back(value);
            }
            // Get results indices from update indices
            ValueRange resultsIndices =
                getResultsIndices(updatesIndices, scatterIndices, indexVectorDim,
                                  scatterDimsToOperandDims, rewriter, loc);
            // Set Args (Value range)
            Value updateValue =
                rewriter.create<tensor::ExtractOp>(loc, updatesValue, updatesIndicesValue);
            Value resultValue =
                rewriter.create<tensor::ExtractOp>(loc, resultsValue, resultsIndices);
            ValueRange args{resultValue, updateValue};

            // Call the function that computes the update
            Value updated = rewriter.create<func::CallOp>(loc, updateFn, args).getResult(0);
            // Insert the update in the results
            rewriter.create<tensor::InsertOp>(loc, updated, resultsValue, resultsIndices);
        }
        // Replace the results with the updated one
        rewriter.replaceOp(op, results);
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

        auto funcBlockArgs = funcBody->getArguments();
        IRRewriter rewriter(b);
        b.setInsertionPointToEnd(funcBody);
        rewriter.mergeBlocks(originalBlock, funcBody, funcBlockArgs);

        b.setInsertionPointToEnd(funcBody);
        b.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                 originalTerminator->getOperands());
        rewriter.eraseOp(originalTerminator);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void generateIndicesRecursive(const std::vector<int64_t> &shape,
                                  std::vector<int64_t> &currentIndex, int64_t dimension,
                                  std::vector<std::vector<int64_t>> &configurations) const
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

    SmallVector<Value> getResultsIndices(std::vector<int64_t> updatesIndices, Value scatterIndices,
                                         int64_t indexVectorDim,
                                         ArrayRef<int64_t> scatterDimsToOperandDims,
                                         mlir::PatternRewriter &rewriter, Location loc) const
    {

        // Check index vector dim is the last dimension
        // Rank
        auto scatterIndicesTensorType = scatterIndices.getType().cast<RankedTensorType>();
        int64_t rank = scatterIndicesTensorType.getRank();
        auto shape = scatterIndicesTensorType.getShape();

        // Offset
        std::vector<int64_t> offsets = updatesIndices;
        offsets.push_back(0);

        std::vector<Value> dynOffsets = {};

        // Size
        std::vector<int64_t> sizes(rank, 0);
        sizes[-1] = shape[-1];
        std::vector<Value> dynSizes = {};

        // Stides
        std::vector<int64_t> strides(rank, 1);
        std::vector<Value> dynStrides = {};

        // Deduce result types
        auto resultType = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            1, scatterIndicesTensorType, offsets, sizes, strides);

        Value scatterIndicesExtracted =
            rewriter.create<tensor::ExtractSliceOp>(loc, resultType, scatterIndices, dynOffsets,
                                                    dynSizes, dynStrides, offsets, sizes, strides);

        // Sort
        SmallVector<Value> results(rank);
        for (auto index : scatterDimsToOperandDims) {
            Value indexValue = rewriter.create<index::ConstantOp>(loc, index);
            Value order = rewriter.create<tensor::ExtractOp>(loc, scatterIndicesExtracted, indexValue);
            // Add to tensor
            results[index] = order;
        }

        // Results indices
        return results;
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
