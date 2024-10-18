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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mhlo/IR/hlo_ops.h"

using namespace mlir;

namespace catalyst {

struct ScatterOpRewritePattern : public mlir::OpRewritePattern<mhlo::ScatterOp> {
    using mlir::OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

    mlir::LogicalResult onlyOneInputUpdateAndResult(mhlo::ScatterOp op) const
    {
        // Assumption 1: only one input, one update, and one result
        // * size(inputs) == 1
        // * size(updates) == 1
        // * size(results) == 1
        // All ScatterOp ops with N inputs, N updates, and N results can be split
        // into N ScatterOp ops with 1 input, 1 update and 1 result.
        // This simplifies the analysis of the update_computation.

        // From:
        // C5: 0 < size(inputs) = size(updates) = N
        // C24: element_type(results[i]) = Ei for all i in [0,N).
        // It follows that:
        // 0 < size(inputs) = size(updates) = size(results) = N
        return op.getResults().size() == 1 ? success() : failure();
    }

    mlir::LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        if (failed(onlyOneInputUpdateAndResult(op))) {
            // Otherwise it will segfault.
            op.emitError() << "Only one input, update, and result";
            return failure();
        }
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
            getOrInsertUpdateFunction(loc, moduleOp, rewriter, region, funcName);
        func::FuncOp updateFnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, updateFn);

        // Get all the variables necessary
        UpdateData variables = getUpdateData(op, rewriter, loc);

        // Create the loop values (start, end and increment)
        Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
        Value sizeAllUpdatesIndices = rewriter.create<index::ConstantOp>(loc, variables.size);
        Value c1 = rewriter.create<index::ConstantOp>(loc, 1);

        // Create a SCF for op, the initial value for args is the results
        Value resultValue =
            rewriter
                .create<scf::ForOp>(
                    loc, c0, sizeAllUpdatesIndices, c1, /*iterArgsInit=*/variables.resultsValue,
                    [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
                        // Get the results
                        auto results = iterArgs.front();

                        // Extract from the all indices tensor the right configuration
                        // with the value i as index: allUpdatesIndices[i]
                        Value updatesIndices;
                        if (variables.allUpdatesIndicesTensor) {
                            updatesIndices = extractUpdateIndices(variables.allUpdatesIndicesTensor,
                                                                  i, loc, builder);
                        }

                        // Scatter update
                        SmallVector<Value> updateScatterIndices;
                        if (variables.allUpdatesIndicesTensor) {
                            for (int64_t index : variables.updatedScatterDims) {
                                Value indexValue = builder.create<index::ConstantOp>(loc, index);
                                Value updateScatterIndex = builder.create<tensor::ExtractOp>(
                                    loc, updatesIndices, indexValue);
                                updateScatterIndices.push_back(updateScatterIndex);
                            }
                        }

                        // Windows update
                        SmallVector<Value> updateWindowsIndices;
                        if (variables.allUpdatesIndicesTensor) {
                            for (int64_t index : variables.updatedWindowsDims) {
                                Value indexValue = builder.create<index::ConstantOp>(loc, index);
                                Value updateWindowsIndex = builder.create<tensor::ExtractOp>(
                                    loc, updatesIndices, indexValue);
                                updateWindowsIndices.push_back(updateWindowsIndex);
                            }
                        }

                        // Get results indices from update indices.
                        // The results indices are used to store the computed update of one element.
                        SmallVector<Value> resultsIndicesValue =
                            getResultsIndices(updateScatterIndices, updateWindowsIndices,
                                              variables.inputsShape, variables.insertedWindowsDims,
                                              variables.scatterIndices, variables.indexVectorDim,
                                              variables.scatterDimsToOperandDims, builder, loc);

                        // Right now the indices are stored in an IR tensor.
                        // We need to extract them all to pass them to the tensor.extract op.
                        SmallVector<Value> updatesIndicesValue;
                        if (updatesIndices) {
                            if (isa<RankedTensorType>(updatesIndices.getType())) {
                                RankedTensorType updateType =
                                    cast<RankedTensorType>(updatesIndices.getType());

                                for (int64_t index = 0; index < updateType.getShape()[0]; ++index) {
                                    Value indexValue =
                                        builder.create<index::ConstantOp>(loc, index);
                                    Value value = builder.create<tensor::ExtractOp>(
                                        loc, updatesIndices, indexValue);
                                    updatesIndicesValue.push_back(value);
                                }
                            }
                        }
                        // Set the arguments of the update function
                        Value updateValue = builder.create<tensor::ExtractOp>(
                            loc, variables.updatesValue, updatesIndicesValue);
                        Value resultValue =
                            builder.create<tensor::ExtractOp>(loc, results, resultsIndicesValue);
                        // The update function from JAX always expects tensors.
                        // Convert f64 -> tensor<f64> if necessary
                        if (!isa<RankedTensorType>(updateValue.getType())) {
                            Type resultTy = RankedTensorType::get({}, updateValue.getType());
                            updateValue =
                                builder.create<tensor::FromElementsOp>(loc, resultTy, updateValue);
                        }
                        if (!isa<RankedTensorType>(resultValue.getType())) {
                            Type resultTy = RankedTensorType::get({}, resultValue.getType());
                            resultValue =
                                builder.create<tensor::FromElementsOp>(loc, resultTy, resultValue);
                        }

                        // Set the arguments for the call op
                        std::vector<Value> args{resultValue, updateValue};

                        // Call the function that computes the update
                        Value updated =
                            builder.create<func::CallOp>(loc, updateFnOp, args).getResult(0);
                        // The update function from JAX always produces tensors.
                        // Convert tensor<f64> -> f64 if necessary
                        Value updatedExtracted;
                        if (isa<RankedTensorType>(updated.getType())) {
                            updatedExtracted = builder.create<tensor::ExtractOp>(loc, updated);
                        }
                        else {
                            updatedExtracted = updated;
                        }
                        // Insert the computed update in the results and replace the previous value
                        Value res = builder.create<tensor::InsertOp>(loc, updatedExtracted, results,
                                                                     resultsIndicesValue);
                        builder.create<scf::YieldOp>(loc, res);
                    })
                .getResult(0);
        // Replace the results with the updated one
        rewriter.replaceOp(op, resultValue);
        return success();
    }

    // Structure to store variables for the SCF for op
    struct UpdateData {
        mlir::Value resultsValue;
        std::vector<int64_t> inputsShape;
        mlir::Value updatesValue;
        std::vector<int64_t> updatesShape;
        mlir::Value scatterIndices;
        int64_t indexVectorDim;
        std::vector<int64_t> scatterDimsToOperandDims;
        std::vector<int64_t> updatedWindowsDims;
        std::vector<int64_t> insertedWindowsDims;
        std::vector<int64_t> updatedScatterDims;
        mlir::Value allUpdatesIndicesTensor;
        int64_t size;
    };

    // Store all the necessary variables for the SCF for op in above defined struct
    UpdateData getUpdateData(mhlo::ScatterOp &op, mlir::PatternRewriter &rewriter,
                             mlir::Location loc) const
    {
        UpdateData data;
        // Get the inputs and updates values
        data.resultsValue = op.getInputs().front();
        data.inputsShape = cast<RankedTensorType>(data.resultsValue.getType()).getShape();

        data.updatesValue = op.getUpdates().front();

        // Get the shape of the updates
        data.updatesShape = cast<TensorType>(data.updatesValue.getType()).getShape();
        int64_t updatesSize = data.updatesShape.size();
        std::vector<int64_t> updatesShapeVector(data.updatesShape.begin(), data.updatesShape.end());

        // Get the scatter indices
        data.scatterIndices = op.getScatterIndices();
        data.indexVectorDim = op.getScatterDimensionNumbers().getIndexVectorDim();
        data.scatterDimsToOperandDims =
            op.getScatterDimensionNumbers().getScatterDimsToOperandDims();

        // Get updated windows dims
        data.updatedWindowsDims = op.getScatterDimensionNumbers().getUpdateWindowDims();

        // Get updated inserted windows dims
        data.insertedWindowsDims = op.getScatterDimensionNumbers().getInsertedWindowDims();

        // Separate updated windows dims
        int64_t start = 0;
        std::vector<int64_t> dimensions;

        for (int64_t i = start; i < updatesSize; ++i) {
            dimensions.push_back(i);
        }
        if (!data.updatedWindowsDims.empty()) {
            std::set_difference(dimensions.begin(), dimensions.end(),
                                data.updatedWindowsDims.begin(), data.updatedWindowsDims.end(),
                                std::back_inserter(data.updatedScatterDims));
        }
        else {
            data.updatedScatterDims = dimensions;
        }

        // Generate all possible indices for the update given the shape of updates
        // The indices are in a flat list
        std::vector<int64_t> indices;
        SmallVector<Value> allUpdatesIndices;
        generateIndicesRecursive(updatesShapeVector, indices, 0, allUpdatesIndices, rewriter, loc);

        // The updates indices generated are stored as a tensorOp (Value) for the access in the SCF
        // for loop
        std::vector<int64_t> totalShape;
        if (allUpdatesIndices.size() != 0) {
            totalShape.push_back(allUpdatesIndices.size() / updatesShapeVector.size());
            totalShape.push_back(updatesShapeVector.size());
        }
        data.size = 1;
        // From the flat list of indices and the shape of the update we create a tensor
        // with FromElementsOp
        if (!allUpdatesIndices.empty()) {
            Type resultTy = RankedTensorType::get(totalShape, rewriter.getIndexType());
            data.allUpdatesIndicesTensor =
                rewriter.create<tensor::FromElementsOp>(loc, resultTy, allUpdatesIndices);
            data.size = allUpdatesIndices.size() / updatesShapeVector.size();
        }
        return data;
    }

    // Take the update block from scatter (bb0) and insert an equivalent function if it does not
    // exist
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

    // Given the shape of update, it generates all the possible indices configuration as a single
    // vector
    void generateIndicesRecursive(const std::vector<int64_t> &shape,
                                  std::vector<int64_t> &currentIndex, size_t dimension,
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

    // Follow the algorithm from https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter
    // in order to get the results indices, the goal is to get the full start index and the
    // full window index and add them in order to get the result indices.
    SmallVector<Value> getResultsIndices(SmallVector<Value> updateScatterIndices,
                                         SmallVector<Value> updateWindowsIndices,
                                         ArrayRef<int64_t> inputsShape,
                                         ArrayRef<int64_t> insertedWindowsDims,
                                         Value scatterIndices, int64_t indexVectorDim,
                                         ArrayRef<int64_t> scatterDimsToOperandDims,
                                         OpBuilder &builder, Location loc) const
    {
        // Get the scatter indices from the update scatter indices
        if (!updateScatterIndices.empty()) {
            scatterIndices = extractScatterIndices(updateScatterIndices, scatterIndices,
                                                   indexVectorDim, loc, builder);
        }
        // Now add the full start indices and full window indices

        // Case for no inserted windows dim
        if (insertedWindowsDims.empty()) {
            // Add Scatter in Windows Indices
            SmallVector<Value> results;
            for (size_t i = 0; i < updateWindowsIndices.size(); ++i) {
                auto indexUpdate = updateWindowsIndices[i];
                auto itScatter =
                    std::find(scatterDimsToOperandDims.begin(), scatterDimsToOperandDims.end(), i);
                if (itScatter != scatterDimsToOperandDims.end()) {
                    int innerIndex = std::distance(scatterDimsToOperandDims.begin(), itScatter);
                    Value indexConstantOp = builder.create<index::ConstantOp>(loc, innerIndex);
                    auto indexScatter =
                        builder.create<tensor::ExtractOp>(loc, scatterIndices, indexConstantOp);
                    auto indexUpdateCasted =
                        builder.create<index::CastSOp>(loc, indexScatter.getType(), indexUpdate);
                    Value addValue =
                        builder.create<arith::AddIOp>(loc, indexScatter, indexUpdateCasted);
                    Value addValueCasted =
                        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), addValue);
                    results.push_back(addValueCasted);
                }
                else {
                    results.push_back(indexUpdate);
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
                    int innerIndex = std::distance(scatterDimsToOperandDims.begin(), itScatter);
                    Value indexConstantOp = builder.create<index::ConstantOp>(loc, innerIndex);
                    auto indexScatter =
                        builder.create<tensor::ExtractOp>(loc, scatterIndices, indexConstantOp);
                    fullStartIndex.push_back(indexScatter);
                }
                else {
                    TypedAttr indexAttr = builder.getI32IntegerAttr(0);
                    Value index = builder.create<arith::ConstantOp>(loc, indexAttr);
                    fullStartIndex.push_back(index);
                }
            }
            // Full windows indices
            SmallVector<Value> fullWindowIndex = updateWindowsIndices;
            for (auto insertedDim : insertedWindowsDims) {
                auto c0 = builder.create<index::ConstantOp>(loc, 0);
                fullWindowIndex.insert(fullWindowIndex.begin() + insertedDim, c0);
            }
            // Add
            SmallVector<Value> results;
            for (size_t i = 0; i < fullWindowIndex.size(); ++i) {
                Value indexScatter = fullStartIndex[i];
                Value indexUpdate = fullWindowIndex[i];
                auto indexUpdateCasted =
                    builder.create<index::CastSOp>(loc, indexScatter.getType(), indexUpdate);
                Value addValue =
                    builder.create<arith::AddIOp>(loc, indexScatter, indexUpdateCasted);
                Value addValueCasted =
                    builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), addValue);
                results.push_back(addValueCasted);
            }
            return results;
        }
    }
    // Given a vector of index value (update scatter), it extracts the value from scatterIndices,
    // The index vector dim indicates the dimension to be extracted.
    Value extractScatterIndices(SmallVector<Value> updateScatterIndices, Value scatterIndices,
                                int64_t indexVectorDim, Location loc, OpBuilder builder) const
    {
        auto scatterIndicesTensorType = cast<RankedTensorType>(scatterIndices.getType());
        // Get the rank and shape of scatter indices
        int64_t rank = scatterIndicesTensorType.getRank();
        auto shape = scatterIndicesTensorType.getShape();

        // Offset
        std::vector<int64_t> offsets(rank, 0);
        std::vector<Value> dynOffsets;

        for (int64_t i = 0; i < rank; i++) {
            if (i != indexVectorDim) {
                offsets[i] = ShapedType::kDynamic;
                if (i > indexVectorDim) {
                    dynOffsets.push_back(updateScatterIndices[i - 1]);
                }
                else {
                    dynOffsets.push_back(updateScatterIndices[i]);
                }
            }
        }

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

        return builder.create<tensor::ExtractSliceOp>(loc, resultType, scatterIndices, dynOffsets,
                                                      dynSizes, dynStrides, offsets, sizes,
                                                      strides);
    }
    // From a index value i it extracts the update indices from the tensor of values.
    Value extractUpdateIndices(Value allUpdatesIndicesTensor, Value i, Location loc,
                               OpBuilder builder) const
    {
        RankedTensorType updateType = cast<RankedTensorType>(allUpdatesIndicesTensor.getType());

        auto rank = updateType.getRank();
        auto shape = updateType.getShape();
        // Offset
        std::vector<int64_t> offsets(rank, 0);
        offsets[0] = ShapedType::kDynamic;
        std::vector<Value> dynOffsets = {i};

        // Size
        std::vector<int64_t> sizes = shape;
        sizes[0] = 1;
        std::vector<Value> dynSizes = {};

        // Stides
        std::vector<int64_t> strides(rank, 1);
        std::vector<Value> dynStrides = {};

        // Deduce result types
        auto resultType = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            rank - 1, updateType, offsets, sizes, strides);

        return builder.create<tensor::ExtractSliceOp>(loc, resultType, allUpdatesIndicesTensor,
                                                      dynOffsets, dynSizes, dynStrides, offsets,
                                                      sizes, strides);
    }
};

void populateScatterPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::ScatterOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
