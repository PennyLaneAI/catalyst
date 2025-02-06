// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include "Catalyst/Utils/CallGraph.h"
#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"

using namespace mlir;

namespace catalyst {
namespace gradient {
// Prototypes
static FailureOr<func::FuncOp> cloneCallee(PatternRewriter &rewriter, Operation *callSite,
                                           OperandRange argOperands, func::FuncOp callee,
                                           SmallVectorImpl<Value> &backpropArgs);
static func::FuncOp genQNodeQuantumOnly(PatternRewriter &rewriter, Location loc,
                                        func::FuncOp qnode);
static func::FuncOp genFullGradFunction(PatternRewriter &rewriter, Location loc,
                                        GradientOpInterface op, FunctionType fnType,
                                        func::FuncOp callee, mlir::BoolAttr keepValueResults);

/// Given a statically-shaped tensor type, execute `processWithIndices` for every entry of the
/// tensor. For example, a tensor<3x2xf64> will cause `processWithIndices` to be called with
/// indices:
/// [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)].
void iterateOverEntries(RankedTensorType resultType, OpBuilder &builder, Location loc,
                        function_ref<void(ValueRange indices)> processWithIndices)
{
    assert(resultType.hasStaticShape());

    ArrayRef<int64_t> shape = resultType.getShape();
    // Compute the strides of the tensor
    unsigned product = 1;
    SmallVector<unsigned> strides;
    for (int64_t dim = resultType.getRank() - 1; dim >= 0; dim--) {
        strides.push_back(product);
        product *= shape[dim];
    }
    std::reverse(strides.begin(), strides.end());

    // Unflatten the tensor indices, using the strides to map from the flat to rank-aware
    // indices
    for (unsigned flatIdx = 0; flatIdx < resultType.getNumElements(); flatIdx++) {
        SmallVector<Value> indices;
        for (int64_t dim = 0; dim < resultType.getRank(); dim++) {
            indices.push_back(
                builder.create<index::ConstantOp>(loc, flatIdx / strides[dim] % shape[dim]));
        }

        processWithIndices(indices);
    }
}

void initializeCotangents(TypeRange primalResultTypes, unsigned activeResult, ValueRange indices,
                          OpBuilder &builder, Location loc, SmallVectorImpl<Value> &cotangents)
{
    Type activeResultType = primalResultTypes[activeResult];
    FloatType elementType =
        cast<FloatType>(isa<RankedTensorType>(activeResultType)
                            ? cast<RankedTensorType>(activeResultType).getElementType()
                            : activeResultType);

    Value zero = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(elementType.getFloatSemantics(), 0), elementType);
    Value one = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(elementType.getFloatSemantics(), 1), elementType);

    Value zeroTensor;
    if (auto activeResultTensor = dyn_cast<RankedTensorType>(activeResultType)) {
        zeroTensor = builder.create<tensor::EmptyOp>(loc, activeResultTensor,
                                                     /*dynamicSizes=*/ValueRange{});
    }
    else {
        zeroTensor = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>(), activeResultType);
    }
    zeroTensor = builder.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0);
    Value cotangent = builder.create<tensor::InsertOp>(loc, one, zeroTensor, indices);

    // Initialize cotangents for all of the primal outputs
    for (const auto &[resultIdx, primalResultType] : llvm::enumerate(primalResultTypes)) {
        if (resultIdx == activeResult) {
            cotangents.push_back(cotangent);
        }
        else {
            // We're not differentiating this output, so we push back a completely
            // zero cotangent. This memory still needs to be writable, hence using
            // an explicit empty + fill vs a constant tensor.
            Value zeroTensor;
            if (isa<RankedTensorType>(primalResultType)) {
                zeroTensor = builder.create<tensor::EmptyOp>(loc, primalResultType, ValueRange{});
            }
            else {
                zeroTensor =
                    builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>(), primalResultType);
            }
            cotangents.push_back(
                builder.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0));
        }
    }
}

/// Recursively process all the QNodes of the `callee` being differentiated. The resulting
/// BackpropOps will be called with `backpropArgs`.
static FailureOr<func::FuncOp> cloneCallee(PatternRewriter &rewriter, Operation *callSite,
                                           OperandRange argOperands, func::FuncOp callee,
                                           SmallVectorImpl<Value> &backpropArgs)
{
    assert(callSite && "Operation pointer is null");

    Location loc = callee.getLoc();
    std::string clonedCalleeName = (callee.getName() + ".cloned").str();
    func::FuncOp clonedCallee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callee, rewriter.getStringAttr(clonedCalleeName));
    if (!clonedCallee) {
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);
        // Replace calls with the QNode with the split QNode in the callee.
        clonedCallee = cast<func::FuncOp>(rewriter.clone(*callee));
        clonedCallee.setName(clonedCalleeName);
        SmallPtrSet<Operation *, 4> qnodes;
        SymbolTableCollection symbolTable;
        traverseCallGraph(callee, &symbolTable, [&](func::FuncOp funcOp) {
            if (funcOp == callee && isQNode(callee)) {
                qnodes.insert(callee);
            }
            else if (isQNode(funcOp)) {
                qnodes.insert(funcOp);
            }
        });

        for (Operation *qnodeOp : qnodes) {
            auto qnode = cast<func::FuncOp>(qnodeOp);
            if (getQNodeDiffMethod(qnode) == "finite-diff") {
                return callee.emitError(
                    "A QNode with diff_method='finite-diff' cannot be specified in a "
                    "callee of method='defer'");
            }

            // In order to allocate memory for various tensors relating to the number of gate
            // parameters at runtime we run a function that merely counts up for each gate parameter
            // encountered.
            func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, qnode);
            func::FuncOp qnodeQuantum = genQNodeQuantumOnly(rewriter, loc, qnode);
            func::FuncOp qnodeSplit = genSplitPreprocessed(rewriter, loc, qnode, qnodeQuantum);

            // This attribute tells downstream patterns that this QNode requires the registration of
            // a custom quantum gradient.
            setRequiresCustomGradient(qnode, FlatSymbolRefAttr::get(qnodeQuantum));

            // Enzyme will fail if this function gets inlined.
            qnodeQuantum->setAttr("passthrough",
                                  rewriter.getArrayAttr(rewriter.getStringAttr("noinline")));

            // Replace calls to the original QNode with calls to the split QNode
            if (isQNode(clonedCallee) && qnode == callee) {
                // As the split preprocessed QNode requires the number of gate params as an extra
                // argument, we need to insert a call to the parameter count function at the
                // location of the grad op.
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(callSite);

                Value paramCount =
                    rewriter.create<func::CallOp>(loc, paramCountFn, argOperands).getResult(0);
                backpropArgs.push_back(paramCount);
                // If the callee is a QNode, we want to backprop through the split preprocessed
                // version.
                rewriter.eraseOp(clonedCallee);
                clonedCallee = qnodeSplit;
            }
            // We modify the callgraph in-place as we traverse it, so we can't use a symbol table
            // here or else the lookup will be stale.
            traverseCallGraph(clonedCallee, /*symbolTable=*/nullptr, [&](func::FuncOp funcOp) {
                funcOp.walk([&](func::CallOp callOp) {
                    if (callOp.getCallee() == qnode.getName()) {
                        PatternRewriter::InsertionGuard insertionGuard(rewriter);
                        rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
                        Value paramCount =
                            rewriter
                                .create<func::CallOp>(loc, paramCountFn, callOp.getArgOperands())
                                .getResult(0);
                        callOp.setCallee(qnodeSplit.getName());
                        callOp.getOperandsMutable().append(paramCount);
                    }
                });
            });
        }
    }
    return clonedCallee;
}

LogicalResult HybridGradientLowering::matchAndRewrite(GradOp op, PatternRewriter &rewriter) const
{
    if (op.getMethod() != "auto") {
        return failure();
    }

    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    SmallVector<Value> backpropArgs(op.getArgOperands());
    FailureOr<func::FuncOp> clonedCallee =
        cloneCallee(rewriter, op, op.getArgOperands(), callee, backpropArgs);
    if (failed(clonedCallee)) {
        return failure();
    }

    // We need to special case this because QNode callees require the parameter count being passed
    // into the BackpropOp within the full grad while non-QNode callees do not. Removing the
    // parameter count would then make the full grad function have the same type as the GradOp.
    SmallVector<Type> fullGradArgTypes(op.getOperandTypes());
    if (isQNode(callee)) {
        fullGradArgTypes.push_back(rewriter.getIndexType());
    }

    // GradOp do not need to keep the results
    mlir::BoolAttr keepValueResults = rewriter.getBoolAttr(false);
    func::FuncOp fullGradFn = genFullGradFunction(
        rewriter, op.getLoc(), op, rewriter.getFunctionType(fullGradArgTypes, op.getResultTypes()),
        *clonedCallee, keepValueResults);

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fullGradFn, backpropArgs);
    return success();
}

LogicalResult HybridValueAndGradientLowering::matchAndRewrite(ValueAndGradOp op,
                                                              PatternRewriter &rewriter) const
{
    if (op.getMethod() != "auto") {
        return failure();
    }

    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    SmallVector<Value> backpropArgs(op.getArgOperands());
    FailureOr<func::FuncOp> clonedCallee =
        cloneCallee(rewriter, op, op.getArgOperands(), callee, backpropArgs);
    if (failed(clonedCallee)) {
        return failure();
    }

    // We need to special case this because QNode callees require the parameter count being passed
    // into the BackpropOp within the full grad while non-QNode callees do not. Removing the
    // parameter count would then make the full grad function have the same type as the GradOp.
    SmallVector<Type> fullGradArgTypes(op.getOperandTypes());
    if (isQNode(callee)) {
        fullGradArgTypes.push_back(rewriter.getIndexType());
    }

    // ValueAndGradOp need to keep the results
    mlir::BoolAttr keepValueResults = rewriter.getBoolAttr(true);
    func::FuncOp fullGradFn = genFullGradFunction(
        rewriter, op.getLoc(), op, rewriter.getFunctionType(fullGradArgTypes, op.getResultTypes()),
        *clonedCallee, keepValueResults);

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fullGradFn, backpropArgs);
    return success();
}

/// Generate a version of the QNode that accepts the parameter buffer. This is so Enzyme will
/// see that the gate parameters flow into the custom quantum function.
static func::FuncOp genQNodeQuantumOnly(PatternRewriter &rewriter, Location loc, func::FuncOp qnode)
{
    std::string fnName = (qnode.getName() + ".quantum").str();
    SmallVector<Type> fnArgTypes(qnode.getArgumentTypes());
    auto paramsTensorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(paramsTensorType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, qnode.getResultTypes());

    func::FuncOp modifiedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qnode, rewriter.getStringAttr(fnName));
    if (modifiedCallee) {
        return modifiedCallee;
    }

    modifiedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    modifiedCallee.setPrivate();
    rewriter.cloneRegionBefore(qnode.getBody(), modifiedCallee.getBody(), modifiedCallee.end());
    Block &entryBlock = modifiedCallee.getFunctionBody().front();
    BlockArgument paramsTensor = entryBlock.addArgument(paramsTensorType, loc);

    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&modifiedCallee.getFunctionBody().front());

    MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
    Value paramCounter = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
    Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, cZero, paramCounter);
    Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

    auto loadThenIncrementCounter = [&](OpBuilder &builder, Value counter,
                                        Value paramTensor) -> Value {
        Value index = builder.create<memref::LoadOp>(loc, counter);
        Value nextIndex = builder.create<index::AddOp>(loc, index, cOne);
        builder.create<memref::StoreOp>(loc, nextIndex, counter);
        return builder.create<tensor::ExtractOp>(loc, paramTensor, index);
    };

    modifiedCallee.walk([&](quantum::DifferentiableGate gateOp) {
        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPoint(gateOp);

        ValueRange diffParams = gateOp.getDiffParams();
        SmallVector<Value> newParams{diffParams.size()};
        for (const auto [paramIdx, recomputedParam] : llvm::enumerate(diffParams)) {
            newParams[paramIdx] = loadThenIncrementCounter(rewriter, paramCounter, paramsTensor);
        }
        MutableOperandRange range{gateOp, static_cast<unsigned>(gateOp.getDiffOperandIdx()),
                                  static_cast<unsigned>(diffParams.size())};
        range.assign(newParams);
    });

    // This function is the point where we can remove the classical preprocessing as a later
    // optimization.
    return modifiedCallee;
}

/// Generate a function that computes a Jacobian row-by-row using one or more BackpropOps.
static func::FuncOp genFullGradFunction(PatternRewriter &rewriter, Location loc,
                                        GradientOpInterface op, FunctionType fnType,
                                        func::FuncOp callee, mlir::BoolAttr keepValueResults)
{
    mlir::DenseIntElementsAttr diffArgIndicesAttr = op.getDiffArgIndicesAttr();
    ValueTypeRange<ResultRange> resultTypes = op->getResultTypes();

    assert((isa<GradOp>(op) || isa<ValueAndGradOp>(op)) &&
           "FullGradFunction should only be generated from GradOp or ValueAndGradOp.");
    size_t numGradients = 0;
    SmallVector<Type> valTypes;
    if (isa<GradOp>(op)) {
        numGradients = op->getNumResults();
    }
    else if (isa<ValueAndGradOp>(op)) {
        // one of the results is the value; the remaining ones are grad
        numGradients = op->getNumResults() - 1;

        // Collect value types
        for (auto &&val : cast<ValueAndGradOp>(&op)->getVals()) {
            valTypes.push_back(val.getType());
        }
    }

    // Define the properties of the full gradient function.
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    std::stringstream uniquer;
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = op.getCallee().getLeafReference().str() + ".fullgrad" + uniquer.str();

    func::FuncOp fullGradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, rewriter.getStringAttr(fnName));

    if (!fullGradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        fullGradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType);
        fullGradFn.setPrivate();
        Block *entryBlock = fullGradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        SmallVector<Value> backpropValResults;
        SmallVector<Value> backpropGradResults{numGradients};
        // Iterate over the primal results
        for (const auto &[cotangentIdx, primalResult] : llvm::enumerate(callee.getResultTypes())) {
            // There is one Jacobian per distinct differential argument.
            SmallVector<Value> jacobians;
            for (unsigned argIdx = 0; argIdx < diffArgIndices.size(); argIdx++) {
                Type jacobianType = resultTypes[argIdx + cotangentIdx * diffArgIndices.size()];
                if (auto tensorType = dyn_cast<RankedTensorType>(jacobianType)) {
                    jacobians.push_back(rewriter.create<tensor::EmptyOp>(
                        loc, tensorType.getShape(), tensorType.getElementType()));
                }
                else {
                    jacobians.push_back(rewriter.create<arith::ConstantFloatOp>(
                        loc, APFloat(0.0), cast<FloatType>(jacobianType)));
                }
            }

            if (auto primalTensorResultType = dyn_cast<RankedTensorType>(primalResult)) {
                // Loop over every entry of this result, creating a one-hot cotangent vector and
                // running a backward pass via the BackpropOp.
                iterateOverEntries(
                    primalTensorResultType, rewriter, loc,
                    [&, cotangentIdx = cotangentIdx](ValueRange indices) {
                        // Initialize cotangents for all of the primal outputs
                        SmallVector<Value> cotangents;
                        initializeCotangents(callee.getResultTypes(), cotangentIdx, indices,
                                             rewriter, loc, cotangents);

                        auto backpropOp = rewriter.create<gradient::BackpropOp>(
                            loc, valTypes, computeBackpropTypes(callee, diffArgIndices),
                            SymbolRefAttr::get(callee), entryBlock->getArguments(),
                            /*arg_shadows=*/ValueRange{},
                            /*primal results=*/ValueRange{}, cotangents, diffArgIndicesAttr,
                            keepValueResults);

                        // After backpropagation, collect any possible callee results into the
                        // values of the grad op
                        backpropValResults.insert(backpropValResults.end(),
                                                  backpropOp.getVals().begin(),
                                                  backpropOp.getVals().end());

                        // Then collect the gradients...

                        // Backprop gives a gradient of a single output entry w.r.t.
                        // all active inputs. Catalyst gives transposed Jacobians,
                        // such that the Jacobians have shape
                        // [...shape_outputs, ...shape_inputs,].
                        for (const auto &[backpropIdx, jacobianSlice] :
                             llvm::enumerate(backpropOp.getGradients())) {
                            if (auto sliceType =
                                    dyn_cast<RankedTensorType>(jacobianSlice.getType())) {
                                size_t sliceRank = sliceType.getRank();
                                auto jacobianType =
                                    cast<RankedTensorType>(jacobians[backpropIdx].getType());
                                size_t jacobianRank = jacobianType.getRank();
                                if (sliceRank < jacobianRank) {
                                    // Offsets are [0] * rank of backprop result + [...indices]
                                    SmallVector<OpFoldResult> offsets;
                                    offsets.append(indices.begin(), indices.end());
                                    offsets.append(sliceRank, rewriter.getIndexAttr(0));

                                    // Sizes are [1] * (jacobianRank - sliceRank) +
                                    // [...sliceShape]
                                    SmallVector<OpFoldResult> sizes;
                                    sizes.append(jacobianRank - sliceRank,
                                                 rewriter.getIndexAttr(1));
                                    for (int64_t dim : sliceType.getShape()) {
                                        sizes.push_back(rewriter.getIndexAttr(dim));
                                    }

                                    // Strides are [1] * jacobianRank
                                    SmallVector<OpFoldResult> strides{jacobianRank,
                                                                      rewriter.getIndexAttr(1)};

                                    jacobians[backpropIdx] = rewriter.create<tensor::InsertSliceOp>(
                                        loc, jacobianSlice, jacobians[backpropIdx], offsets, sizes,
                                        strides);
                                }
                                else {
                                    jacobians[backpropIdx] = jacobianSlice;
                                }
                            }
                            else {
                                assert(isa<FloatType>(jacobianSlice.getType()));
                                jacobians[backpropIdx] = rewriter.create<tensor::InsertOp>(
                                    loc, jacobianSlice, jacobians[backpropIdx], indices);
                            }
                            backpropGradResults[backpropIdx +
                                                cotangentIdx * diffArgIndices.size()] =
                                jacobians[backpropIdx];
                        }
                    });
            }
            else {
                // Backprop through a scalar result.
                SmallVector<Value> cotangents;
                initializeCotangents(callee.getResultTypes(), cotangentIdx, ValueRange(), rewriter,
                                     loc, cotangents);

                auto backpropOp = rewriter.create<gradient::BackpropOp>(
                    loc, valTypes, computeBackpropTypes(callee, diffArgIndices),
                    SymbolRefAttr::get(callee), entryBlock->getArguments(),
                    /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
                    diffArgIndicesAttr, keepValueResults);

                // After backpropagation, collect any possible callee results into the values of the
                // grad op
                backpropValResults.insert(backpropValResults.end(), backpropOp.getVals().begin(),
                                          backpropOp.getVals().end());

                // Then collect the gradients...
                for (const auto &[backpropIdx, jacobianSlice] :
                     llvm::enumerate(backpropOp.getGradients())) {
                    size_t resultIdx = backpropIdx * callee.getNumResults() + cotangentIdx;
                    backpropGradResults[resultIdx] = jacobianSlice;
                    if (jacobianSlice.getType() != resultTypes[resultIdx]) {
                        // For ergonomics, if the backprop result is a point tensor and the
                        // user requests a scalar, give it to them.
                        if (isa<RankedTensorType>(jacobianSlice.getType()) &&
                            isa<FloatType>(resultTypes[resultIdx])) {
                            backpropGradResults[resultIdx] = rewriter.create<tensor::ExtractOp>(
                                loc, jacobianSlice, ValueRange{});
                        }
                    }
                }
            }
        }

        // Combine both values and gradients and return them
        SmallVector<Value> backpropResults;
        backpropResults.insert(backpropResults.end(), backpropValResults.begin(),
                               backpropValResults.end());
        backpropResults.insert(backpropResults.end(), backpropGradResults.begin(),
                               backpropGradResults.end());

        rewriter.create<func::ReturnOp>(loc, backpropResults);
    } // if (!fullGradFn)
    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
