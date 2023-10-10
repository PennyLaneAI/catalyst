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
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mhlo/IR/hlo_ops.h"

using namespace mlir;

namespace {

struct ScatterOpRewritePattern : public mlir::OpRewritePattern<mhlo::ScatterOp> {
    using mlir::OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // Compute operation hash in case they are more than one scatter and they have different
        // update function
        auto opHash = OperationEquivalence::computeHash(op);
        // Create the function to replace the update block from scatter
        std::string funcName = "__catalyst_update_scatter" + std::to_string(opHash);

        Location loc = op.getLoc();
        // Add checks for supported cases (assumptions: no update windows dim, unique indices and
        // sorted indices)
        if (!op.getUniqueIndices() || !op.getIndicesAreSorted()) {
            op.emitError() << "Indices are not unique and/or not sorted, unique boolean: "
                           << op.getUniqueIndices()
                           << ", sorted boolean :" << op.getIndicesAreSorted();
            return failure();
        }

        // Extract the block responsible for update
        Region &region = op.getUpdateComputation();

        if (!region.hasOneBlock())
            return failure();

        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

        // We create a function from the update block
        FlatSymbolRefAttr updateFn =
            getOrInsertUpdateFunction(op.getLoc(), moduleOp, rewriter, region, funcName);
        func::FuncOp updateFnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, updateFn);

        // Get the inputs and updates values
        Value resultsValue = op.getInputs().front();
        auto inputsShape = resultsValue.getType().cast<RankedTensorType>().getShape();

        Value updatesValue = op.getUpdates().front();

        // Get the shape of the updates
        ArrayRef<int64_t> updatesShape = updatesValue.getType().cast<TensorType>().getShape();
        int64_t updatesSize = updatesShape.size();
        std::vector<int64_t> updatesShapeVector(updatesShape.begin(), updatesShape.end());

        // Get the scatter indices
        auto scatterIndices = op.getScatterIndices();
        int64_t indexVectorDim = op.getScatterDimensionNumbers().getIndexVectorDim();
        ArrayRef<int64_t> scatterDimsToOperandDims =
            op.getScatterDimensionNumbers().getScatterDimsToOperandDims();

        // Get updated windows dims
        ArrayRef<int64_t> updatedWindowsDims =
            op.getScatterDimensionNumbers().getUpdateWindowDims();

        // Get updated inserted windows dims
        ArrayRef<int64_t> insertedWindowsDims =
            op.getScatterDimensionNumbers().getInsertedWindowDims();

        std::vector<int64_t> updatedScatterDims;
        // Separate updated windows dims
        if (!updatedWindowsDims.empty()) {
            int64_t start = 0;
            std::vector<int64_t> dimensions;

            for (int64_t i = start; i <= updatesSize; ++i) {
                dimensions.push_back(i);
            }

            std::copy_if(updatedWindowsDims.begin(), updatedWindowsDims.end(),
                         std::back_inserter(updatedScatterDims), [&dimensions](int64_t element) {
                             return std::find(dimensions.begin(), dimensions.end(), element) !=
                                    dimensions.end();
                         });
        }

        // Generate all possible indices given the shape of updates
        std::vector<int64_t> indices;
        SmallVector<Value> allUpdatesIndices;
        generateIndicesRecursive(updatesShapeVector, indices, 0, allUpdatesIndices, rewriter, loc);
        // All updates indices -> tensor
        // get shape and type
        std::vector<int64_t> totalShape;
        if (allUpdatesIndices.size() != 0) {
            totalShape.push_back(allUpdatesIndices.size() / updatesShapeVector.size());
            totalShape.push_back(updatesShapeVector.size());
        }

        Type resultTy = RankedTensorType::get(totalShape, rewriter.getIndexType());
        auto allUpdatesIndicesTensor =
            rewriter.create<tensor::FromElementsOp>(loc, resultTy, allUpdatesIndices);

        // Create the loop values
        Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
        Value sizeAllUpdatesIndices =
            rewriter.create<index::ConstantOp>(loc, allUpdatesIndices.size());
        Value c1 = rewriter.create<index::ConstantOp>(loc, 1);

        Value resultValue =
            rewriter
                .create<scf::ForOp>(
                    loc, c0, sizeAllUpdatesIndices, c1, /*iterArgsInit=*/resultsValue,
                    [&](OpBuilder &rewriter, Location loc, Value i, ValueRange iterArgs) {
                        Value updatesIndices =
                            rewriter.create<tensor::ExtractOp>(loc, allUpdatesIndicesTensor, i);

                        // Scatter update
                        SmallVector<Value> updateScatterIndices;
                        for (int64_t index : updatedScatterDims) {
                            Value indexValue = rewriter.create<index::ConstantOp>(loc, index);
                            Value updateScatterIndex =
                                rewriter.create<tensor::ExtractOp>(loc, updatesIndices, indexValue);
                            updateScatterIndices.push_back(updateScatterIndex);
                        }
                        // Windows update
                        SmallVector<Value> updateWindowsIndices;
                        for (int64_t index : updatedWindowsDims) {
                            Value indexValue = rewriter.create<index::ConstantOp>(loc, index);
                            Value updateWindowsIndex =
                                rewriter.create<tensor::ExtractOp>(loc, updatesIndices, indexValue);
                            updateWindowsIndices.push_back(updateWindowsIndex);
                        }
                        // Get results indices from update indices
                        std::cout << updateScatterIndices.empty() << std::endl;
                        std::cout << updateWindowsIndices.empty() << std::endl;
                        SmallVector<Value> resultsIndicesValue = getResultsIndices(
                            updateScatterIndices, updateWindowsIndices, inputsShape,
                            insertedWindowsDims, scatterIndices, indexVectorDim,
                            scatterDimsToOperandDims, rewriter, loc);

                        // Tensor to vector rework
                        SmallVector<Value> updatesIndicesValue;
                        if (!isa<RankedTensorType>(updatesIndices.getType())) {
                            RankedTensorType updateType =
                                updatesIndices.getType().cast<RankedTensorType>();
                            for (int64_t index = 0; index <= updateType.getShape().size();
                                 ++index) {
                                Value indexValue = rewriter.create<index::ConstantOp>(loc, index);
                                Value value = rewriter.create<tensor::ExtractOp>(
                                    loc, updatesIndices, indexValue);
                                updatesIndicesValue.push_back(value);
                            }
                        }
                        // Set Args (Value range)
                        Value updateValue = rewriter.create<tensor::ExtractOp>(loc, updatesValue,
                                                                               updatesIndicesValue);
                        Value resultValue = rewriter.create<tensor::ExtractOp>(loc, resultsValue,
                                                                               resultsIndicesValue);
                        // f64 -> tensor<f64> if necessary
                        if (!isa<RankedTensorType>(updateValue.getType())) {
                            Type resultTy = RankedTensorType::get({}, updateValue.getType());
                            updateValue =
                                rewriter.create<tensor::FromElementsOp>(loc, resultTy, updateValue);
                        }
                        if (!isa<RankedTensorType>(resultValue.getType())) {
                            Type resultTy = RankedTensorType::get({}, resultValue.getType());
                            resultValue =
                                rewriter.create<tensor::FromElementsOp>(loc, resultTy, resultValue);
                        }

                        // Set the arguments for the call op
                        std::vector<Value> args{resultValue, updateValue};

                        // Call the function that computes the update
                        Value updated =
                            rewriter.create<func::CallOp>(loc, updateFnOp, args).getResult(0);
                        // Make it a scalar if necessary tensor<f64> -> f64
                        Value updatedExtracted;
                        if (isa<RankedTensorType>(updated.getType())) {
                            updatedExtracted = rewriter.create<tensor::ExtractOp>(loc, updated);
                        }
                        else {
                            updatedExtracted = updated;
                        }
                        // Insert the update in the results and replace the previous value
                        Value res = rewriter.create<tensor::InsertOp>(
                            loc, updatedExtracted, resultsValue, resultsIndicesValue);
                        rewriter.create<scf::YieldOp>(loc, res);
                    })
                .getResult(0);
        // Replace the results with the updated one
        rewriter.replaceOp(op, resultsValue);
        return success();
    }

    FlatSymbolRefAttr getOrInsertUpdateFunction(Location loc, ModuleOp moduleOp, OpBuilder &builder,
                                                Region &updateRegion, std::string funcName) const
    {
        MLIRContext *ctx = builder.getContext();

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
                                  SmallVector<Value> &configurations,
                                  mlir::PatternRewriter &rewriter, Location loc) const
    {
        if (dimension == shape.size()) {
            for (auto elem : currentIndex) {
                // integer to Value
                auto valueCurrentIndex = rewriter.create<index::ConstantOp>(loc, elem);
                // Add to configuration
                configurations.push_back(valueCurrentIndex);
            }
        }
        else {
            for (int i = 0; i < shape[dimension]; i++) {
                currentIndex.push_back(i);
                generateIndicesRecursive(shape, currentIndex, dimension + 1, configurations,
                                         rewriter, loc);
                currentIndex.pop_back();
            }
        }
    }

    SmallVector<Value> getResultsIndices(SmallVector<Value> updateScatterIndices,
                                         SmallVector<Value> updateWindowsIndices,
                                         ArrayRef<int64_t> inputsShape,
                                         ArrayRef<int64_t> insertedWindowsDims,
                                         Value scatterIndices, int64_t indexVectorDim,
                                         ArrayRef<int64_t> scatterDimsToOperandDims,
                                         OpBuilder &rewriter, Location loc) const
    {
        // Check index vector dim is the last dimension
        // Rank
        auto scatterIndicesTensorType = scatterIndices.getType().cast<RankedTensorType>();
        int64_t rank = scatterIndicesTensorType.getRank();
        auto shape = scatterIndicesTensorType.getShape();

        // Get the scatter indices
        if (!updateScatterIndices.empty()) {
            // Offset
            std::vector<int64_t> offsets(rank, 0);
            std::vector<Value> dynOffsets = {};

            // Size
            std::vector<int64_t> sizes(rank, 1);
            sizes[indexVectorDim] = shape[indexVectorDim];
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

        if (insertedWindowsDims.empty()) {
            // Add Scatter in Windows Indices
            SmallVector<Value> results;
            for (size_t i = 0; i < updateWindowsIndices.size(); ++i) {
                auto indexUpdate = updateWindowsIndices[i];
                auto itScatter =
                    std::find(scatterDimsToOperandDims.begin(), scatterDimsToOperandDims.end(), i);
                if (itScatter != scatterDimsToOperandDims.end()) {
                    Value index = rewriter.create<index::ConstantOp>(loc, i);
                    auto indexScatter =
                        rewriter.create<tensor::ExtractOp>(loc, scatterIndices, index);

                    Value addValue = rewriter.create<arith::AddIOp>(loc, indexScatter, indexUpdate);
                    Value addValueCasted =
                        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), addValue);
                    results.push_back(addValueCasted);
                }
                else {
                    Value indexValue = rewriter.create<index::ConstantOp>(loc, indexUpdate);
                    results.push_back(indexValue);
                }
            }
            return results;
        }
        else {
            SmallVector<Value> fullStartIndex;
            for (size_t i = 0; i < inputsShape.size(); ++i) {
                // Full start indices (use scatter dims op)
                auto itScatter =
                    std::find(scatterDimsToOperandDims.begin(), scatterDimsToOperandDims.end(), i);
                if (itScatter != scatterDimsToOperandDims.end()) {
                    Value index = rewriter.create<index::ConstantOp>(loc, i);
                    auto indexScatter =
                        rewriter.create<tensor::ExtractOp>(loc, scatterIndices, index);
                    fullStartIndex.push_back(indexScatter);
                }
                else {
                    TypedAttr indexAttr = rewriter.getI32IntegerAttr(0);
                    Value index = rewriter.create<arith::ConstantOp>(loc, indexAttr);
                    fullStartIndex.push_back(index);
                }
            }
            SmallVector<Value> fullWindowIndex;
            // Full windows indices
            for (auto insertedDim : insertedWindowsDims) {
                auto c0 = rewriter.create<index::ConstantOp>(loc, 0);
                updateWindowsIndices.insert(updateWindowsIndices.begin() + insertedDim, c0);
            }
            // Add
            SmallVector<Value> results;
            for (size_t i = 0; i < updateWindowsIndices.size(); ++i) {
                Value indexScatter = fullStartIndex[i];
                auto indexUpdate = updateWindowsIndices[i];
                Value addValue = rewriter.create<arith::AddIOp>(loc, indexScatter, indexUpdate);
                Value addValueCasted =
                    rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), addValue);
                results.push_back(addValueCasted);
            }
            return results;
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
