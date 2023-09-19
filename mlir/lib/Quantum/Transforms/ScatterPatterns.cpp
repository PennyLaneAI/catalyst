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
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mhlo/IR/hlo_ops.h"

using namespace mlir;

namespace {

struct ScatterOpRewritePattern : public mlir::OpRewritePattern<mhlo::ScatterOp> {
    using mlir::OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        // Add checks for supported cases (assumptions: no update windows dim, unique indices and
        // sorted indices)
        if (!op.getUniqueIndices() || !op.getIndicesAreSorted() ||
            !op.getScatterDimensionNumbers().getUpdateWindowDims().empty()) {
            return failure();
        }

        op.getScatterDimensionNumbers();
        // Extract the block responsible for update
        Region &region = op.getUpdateComputation();

        if (!region.hasOneBlock())
            return failure();

        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

        // We create a function from the update block
        FlatSymbolRefAttr updateFn =
            getOrInsertUpdateFunction(op.getLoc(), moduleOp, rewriter, region);
        func::FuncOp updateFnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, updateFn);

        // Get the inputs and updates values
        Value resultsValue = op.getInputs().front();
        Value updatesValue = op.getUpdates().front();

        // Get the shape of the updates
        ArrayRef<int64_t> updatesShape = updatesValue.getType().cast<TensorType>().getShape();
        std::vector<int64_t> updatesShapeVector(updatesShape.begin(), updatesShape.end());

        // Get the scatter indices
        auto scatterIndices = op.getScatterIndices();
        int64_t indexVectorDim = op.getScatterDimensionNumbers().getIndexVectorDim();
        ArrayRef<int64_t> scatterDimsToOperandDims =
            op.getScatterDimensionNumbers().getScatterDimsToOperandDims();

        // Generate all possible indices given the shape of updates
        std::vector<int64_t> indices;
        std::vector<std::vector<int64_t>> allUpdatesIndices;
        generateIndicesRecursive(updatesShapeVector, indices, 0, allUpdatesIndices);

        // Replace the results with the updated inputs.
        for (auto updatesIndices : allUpdatesIndices) {
            SmallVector<Value> updatesIndicesValue;
            for (int64_t index : updatesIndices) {
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

            // f64 -> tensor<f64> if necessary
            if (!isa<RankedTensorType>(updateValue.getType())) {
                Type resultTy = RankedTensorType::get({}, updateValue.getType());
                updateValue = rewriter.create<tensor::FromElementsOp>(loc, resultTy, updateValue);
            }
            if (!isa<RankedTensorType>(resultValue.getType())) {
                Type resultTy = RankedTensorType::get({}, resultValue.getType());
                resultValue = rewriter.create<tensor::FromElementsOp>(loc, resultTy, resultValue);
            }

            // Set the arguments for the call op
            ValueRange args{resultValue, updateValue};

            // Call the function that computes the update
            Value updated = rewriter.create<func::CallOp>(loc, updateFnOp, args).getResult(0);

            // Make it a scalar if necessary tensor<f64> -> f64
            Value updatedExtracted;
            if (isa<RankedTensorType>(updated.getType())) {
                updatedExtracted = rewriter.create<tensor::ExtractOp>(loc, updated);
            }
            else {
                updatedExtracted = updated;
            }
            // Insert the update in the results and replace the previous value
            Value res = rewriter.create<tensor::InsertOp>(loc, updatedExtracted, resultsValue,
                                                          resultsIndices);
            resultsValue = res;
        }
        // Replace the results with the updated one
        rewriter.replaceOp(op, resultsValue);
        return success();
    }

    FlatSymbolRefAttr getOrInsertUpdateFunction(Location loc, ModuleOp moduleOp, OpBuilder &builder,
                                                Region &updateRegion) const
    {
        MLIRContext *ctx = builder.getContext();

        // Create the function to replace the update block from scatter
        std::string funcName = "__catalyst_update_scatter";

        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        Block *originalBlock = &updateRegion.front();
        Operation *originalTerminator = originalBlock->getTerminator();
        ValueRange originalArguments = originalBlock->getArguments();

        // Get the arguments and outputs types from the original block
        FunctionType updateFnType =
            FunctionType::get(ctx, /*inputs=*/
                              originalArguments.getTypes(),
                              /*outputs=*/originalTerminator->getOperandTypes());

        func::FuncOp updateFn = builder.create<func::FuncOp>(loc, funcName, updateFnType);
        updateFn.setPrivate();

        // Create the block of the function
        Block *funcBody = updateFn.addEntryBlock();

        auto funcBlockArgs = funcBody->getArguments();
        IRRewriter rewriter(builder);
        builder.setInsertionPointToEnd(funcBody);

        // Merge the two blocks and delete the first one
        rewriter.mergeBlocks(originalBlock, funcBody, funcBlockArgs);

        builder.setInsertionPointToEnd(funcBody);
        builder.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                       originalTerminator->getOperands());
        rewriter.eraseOp(originalTerminator);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void generateIndicesRecursive(const std::vector<int64_t> &shape,
                                  std::vector<int64_t> &currentIndex, int64_t dimension,
                                  std::vector<std::vector<int64_t>> &configurations) const
    {
        if (dimension == shape.size()) {
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

        if (rank != 1) {
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

            scatterIndices = rewriter.create<tensor::ExtractSliceOp>(
                loc, resultType, scatterIndices, dynOffsets, dynSizes, dynStrides, offsets, sizes,
                strides);
        }
        // Sort
        SmallVector<Value> results(rank);
        for (auto index : scatterDimsToOperandDims) {
            Value indexValue = rewriter.create<index::ConstantOp>(loc, index);
            Value value = rewriter.create<tensor::ExtractOp>(loc, scatterIndices, indexValue);
            Value valueCasted =
                rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), value);
            // Add to tensor
            results[index] = valueCasted;
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
