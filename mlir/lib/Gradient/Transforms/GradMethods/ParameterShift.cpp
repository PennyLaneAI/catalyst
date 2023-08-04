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

#include "ParameterShift.hpp"
#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include <algorithm>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GetDiffMethod.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"
using llvm::errs;
namespace catalyst {
namespace gradient {

LogicalResult ParameterShiftLowering::match(GradOp op) const
{
    if (getQNodeDiffMethod(op) == "parameter-shift") {
        return success();
    }
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    bool found = false;
    callee.walk([&](func::CallOp op) {
        auto nestedCallee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        if (nestedCallee->hasAttr("qnode") &&
            nestedCallee->getAttrOfType<StringAttr>("diff_method") == "parameter-shift") {
            found = true;
        }
    });
    if (found) {
        return success();
    }
    return failure();
}

/// Generate a version of the QNode that accepts the parameter buffer. This is so Enzyme will see
/// that the gate parameters flow into the custom quantum function.
func::FuncOp genQNodeWithParams(PatternRewriter &rewriter, Location loc, func::FuncOp qnode)
{
    std::string fnName = (qnode.getName() + ".withparams").str();
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

    modifiedCallee.walk([&](Operation *op) {
        if (auto gateOp = dyn_cast<quantum::DifferentiableGate>(op)) {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(gateOp);

            ValueRange diffParams = gateOp.getDiffParams();
            SmallVector<Value> newParams{diffParams.size()};
            for (const auto [paramIdx, recomputedParam] : llvm::enumerate(diffParams)) {
                newParams[paramIdx] =
                    loadThenIncrementCounter(rewriter, paramCounter, paramsTensor);
            }
            MutableOperandRange range{gateOp, static_cast<unsigned>(gateOp.getDiffOperandIdx()),
                                      static_cast<unsigned>(diffParams.size())};
            range.assign(newParams);
        }
    });

    // This function is the point where we can remove the classical preprocessing as a later
    // optimization.
    return modifiedCallee;
}

/// Generate a version of the QNode that writes gate parameters to a buffer before calling a
/// modified QNode that explicitly accepts preprocessed gate parameters.
func::FuncOp genSplitPreprocessed(PatternRewriter &rewriter, Location loc, func::FuncOp qnode,
                                  func::FuncOp qnodeWithParams)
{
    // Copied from the argmap function because it's very similar.
    // Define the properties of the classical preprocessing function.
    std::string fnName = qnode.getSymName().str() + ".splitpreprocessed";
    SmallVector<Type> fnArgTypes(qnode.getArgumentTypes());
    auto paramsBufferType = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(rewriter.getIndexType()); // parameter count
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, qnode.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp argMapFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qnode, rewriter.getStringAttr(fnName));
    if (!argMapFn) {
        // First copy the original function as is, then we can replace all quantum ops by collecting
        // their gate parameters in a memory buffer instead. The size of this vector is passed as an
        // input to the new function.
        argMapFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.cloneRegionBefore(qnode.getBody(), argMapFn.getBody(), argMapFn.end());
        Block &argMapBlock = argMapFn.getFunctionBody().front();
        SmallVector<Value> qnodeWithParamsArgs{argMapBlock.getArguments()};

        Value paramCount = argMapBlock.addArgument(rewriter.getIndexType(), loc);
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&argMapFn.getBody().front());
        Value paramsBuffer = rewriter.create<memref::AllocOp>(loc, paramsBufferType, paramCount);
        Value paramsTensor = rewriter.create<bufferization::ToTensorOp>(loc, paramsBuffer);

        qnodeWithParamsArgs.push_back(paramsTensor);
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        argMapFn.walk([&](Operation *op) {
            // Insert gate parameters into the params buffer.
            if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange diffParams = gate.getDiffParams();
                if (!diffParams.empty()) {
                    Value paramIdx = rewriter.create<memref::LoadOp>(loc, paramsProcessed);
                    for (auto param : diffParams) {
                        rewriter.create<memref::StoreOp>(loc, param, paramsBuffer, paramIdx);
                        paramIdx = rewriter.create<index::AddOp>(loc, paramIdx, cOne);
                    }
                    rewriter.create<memref::StoreOp>(loc, paramIdx, paramsProcessed);
                }

                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            // Return ops should be preceded with calls to the modified QNode
            else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(returnOp);
                auto modifiedCall =
                    rewriter.create<func::CallOp>(loc, qnodeWithParams, qnodeWithParamsArgs);

                returnOp.getOperandsMutable().assign(modifiedCall.getResults());
            }
            // Erase redundant device specifications.
            else if (isa<quantum::DeviceOp>(op)) {
                rewriter.eraseOp(op);
            }
        });

        quantum::removeQuantumMeasurements(argMapFn);
    }

    return argMapFn;
}

void ParameterShiftLowering::rewrite(GradOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    rewriter.setInsertionPointAfter(callee);
    // Replace calls with the QNode with the split QNode in the callee.
    auto clonedCallee = cast<func::FuncOp>(rewriter.clone(*callee));
    std::string clonedCalleeName = (callee.getName() + ".cloned").str();
    clonedCallee.setName(clonedCalleeName);
    SmallPtrSet<Operation *, 4> qnodes;
    auto isQNode = [](func::FuncOp funcOp) { return funcOp->hasAttr("qnode"); };
    if (isQNode(clonedCallee)) {
        qnodes.insert(clonedCallee);
    }
    else {
        clonedCallee.walk([&](func::CallOp callOp) {
            auto nestedCallee =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callOp.getCalleeAttr());
            if (isQNode(nestedCallee)) {
                qnodes.insert(nestedCallee);
            }
        });
    }

    for (Operation *qnodeOp : qnodes) {
        auto qnode = cast<func::FuncOp>(qnodeOp);
        // Determine the number of parameters to shift (= to the total static number of gate
        // parameters occuring in the function) and number of selectors needed (= to the number of
        // loop nests containing quantum instructions with at least one gate parameter).
        auto [numShifts, loopDepth] = analyzeFunction(qnode);

        // In order to allocate memory for various tensors relating to the number of gate parameters
        // at runtime we run a function that merely counts up for each gate parameter encountered.
        func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, qnode);

        func::FuncOp qnodeWithParams = genQNodeWithParams(rewriter, loc, qnode);

        func::FuncOp qnodeSplit = genSplitPreprocessed(rewriter, loc, qnode, qnodeWithParams);

        // Replace calls to the original QNode with calls to the split QNode
        clonedCallee.walk([&](func::CallOp callOp) {
            if (callOp.getCallee() == qnode.getName()) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPointToStart(&clonedCallee.getFunctionBody().front());
                Value paramCount =
                    rewriter.create<func::CallOp>(loc, paramCountFn, callOp.getArgOperands())
                        .getResult(0);
                callOp.setCallee(qnodeSplit.getName());
                callOp.getOperandsMutable().append(paramCount);
            }
        });

        // // Generate the classical argument map from function arguments to gate parameters. This
        // // function will be differentiated to produce the classical jacobian.
        // func::FuncOp argMapFn = genArgMapFunction(rewriter, loc, callee);

        // Generate the shifted version of callee, enabling us to shift an arbitrary gate
        // parameter at runtime.
        func::FuncOp shiftFn = genShiftFunction(rewriter, loc, qnode, numShifts, loopDepth);

        // Generate the quantum gradient function, exploiting the structure of the original function
        // to dynamically compute the partial derivate with respect to each gate parameter.
        func::FuncOp qGradFn =
            genQGradFunction(rewriter, loc, qnode, shiftFn, numShifts, loopDepth);

        // Generate the quantum gradient function at the tensor level, then register it as an
        // attribute.
        qnodeWithParams->setAttr("gradient.qgrad", FlatSymbolRefAttr::get(qGradFn.getNameAttr()));
        // Enzyme will fail if this function gets inlined.
        qnodeWithParams->setAttr("passthrough",
                                 rewriter.getArrayAttr(rewriter.getStringAttr("noinline")));
    }

    rewriter.setInsertionPoint(op);
    std::vector<size_t> diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    SmallVector<Value> backpropResults;
    /*In the event of multiple results, we need as many backprop calls as scalar element results.
      Assume one input and one output for now.
      f = (tensor<3xf64>) -> (tensor<2x2xf64>)
      jacobian = empty() : tensor<3x2x2xf64>
      for i in range(outshape[0]):
        for j in range(outshape[1]):
          cotangent = zeros(outshape)
          cotangent[i, j] = 1.0
          jacobian[:, i, j] = backprop(cotangent)
    */

    assert(clonedCallee.getNumResults() == 1 && "wip multi result");
    Value jacobian = rewriter.create<tensor::EmptyOp>(loc, op.getResultTypes()[0],
                                                      /*dynamicSizes=*/ValueRange{});

    auto resultType = cast<RankedTensorType>(clonedCallee.getResultTypes().front());
    assert(resultType.hasStaticShape());
    ArrayRef<int64_t> shape = resultType.getShape();
    // Compute the strides in reverse
    unsigned product = 1;
    SmallVector<unsigned> strides;
    for (int64_t dim = resultType.getRank() - 1; dim >= 0; dim--) {
        strides.push_back(product);
        product *= shape[dim];
    }
    std::reverse(strides.begin(), strides.end());

    auto materializeIndex = [&rewriter, &loc](int64_t idx) {
        return rewriter.create<index::ConstantOp>(loc, idx);
    };
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(resultType.getElementType(), 0.0));
    Value one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(resultType.getElementType(), 1.0));
    Value zeroTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultType, /*dynamicSizes=*/ValueRange{});
    zeroTensor = rewriter.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0);

    for (unsigned flatIdx = 0; flatIdx < resultType.getNumElements(); flatIdx++) {
        // Unflatten the tensor indices
        SmallVector<Value> indices;
        for (int64_t dim = 0; dim < resultType.getRank(); dim++) {
            indices.push_back(materializeIndex(flatIdx / strides[dim] % shape[dim]));
        }

        Value cotangent = rewriter.create<tensor::InsertOp>(loc, one, zeroTensor, indices);

        auto backpropOp = rewriter.create<gradient::BackpropOp>(
            loc, computeBackpropTypes(clonedCallee, diffArgIndices), clonedCallee.getName(),
            op.getArgOperands(),
            /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangent,
            op.getDiffArgIndicesAttr());

        // May need insert or insert_slice here
        assert(backpropOp.getNumResults() == 1);
        Value result = backpropOp.getResult(0);
        auto sliceType = cast<RankedTensorType>(result.getType());
        auto jacobianType = cast<RankedTensorType>(op.getResultTypes()[0]);
        size_t sliceRank = sliceType.getRank();
        size_t jacobianRank = jacobianType.getRank();
        if (sliceRank < jacobianRank) {
            // Offsets are [...indices] + [0] * rank of backprop input
            SmallVector<OpFoldResult> offsets;
            offsets.append(indices.begin(), indices.end());
            offsets.append(sliceRank, rewriter.getIndexAttr(0));

            // Sizes are [1] * (jacobianRank - sliceRank) + [...sliceShape]
            SmallVector<OpFoldResult> sizes;
            sizes.append(jacobianRank - sliceRank, rewriter.getIndexAttr(1));
            for (int64_t dim : sliceType.getShape()) {
                sizes.push_back(rewriter.getIndexAttr(dim));
            }

            // Strides are [1] * jacobianRank
            SmallVector<OpFoldResult> strides{jacobianRank, rewriter.getIndexAttr(1)};

            // for (auto x : offsets) {
            //     errs() << "offset: " << x << "\n";
            // }
            // for (auto x : sizes) {
            //     errs() << "size: " << x << "\n";
            // }
            // for (auto x : strides) {
            //     errs() << "stride: " << x << "\n";
            // }

            jacobian = rewriter.create<tensor::InsertSliceOp>(loc, backpropOp.getResult(0),
                                                              jacobian, offsets, sizes, strides);
        }
        else {
            jacobian = backpropOp.getResult(0);
        }
    }

    rewriter.replaceOp(op, jacobian);
    return;

    assert(false && "stopping here");

    for (const auto &[activeResultIdx, activeResultType] :
         llvm::enumerate(clonedCallee.getResultTypes())) {
        // Initialize cotangents
        SmallVector<Value> cotangents;
        for (const auto &[resultIdx, resultType] : llvm::enumerate(clonedCallee.getResultTypes())) {
            assert(isDifferentiable(resultType));
            if (resultIdx == activeResultIdx) {
                if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
                    assert(tensorType.hasStaticShape());
                    Value cotangent = rewriter.create<tensor::EmptyOp>(
                        loc, tensorType, /*dynamicSizes=*/ValueRange{});
                    Value one = rewriter.create<arith::ConstantOp>(
                        loc, FloatAttr::get(tensorType.getElementType(), 1.0));
                    Value zero = rewriter.create<arith::ConstantOp>(
                        loc, FloatAttr::get(tensorType.getElementType(), 0.0));

                    if (tensorType.getRank() == 0) {
                        cotangent =
                            rewriter.create<tensor::InsertOp>(loc, one, cotangent, ValueRange{});
                    }
                    else {
                        cotangent =
                            rewriter.create<linalg::FillOp>(loc, zero, cotangent).getResult(0);
                        for (int64_t dim = 0; dim < tensorType.getRank(); dim++) {
                            for (int64_t i = 0; i < tensorType.getDimSize(dim); i++) {
                                SmallVector<Value> indices;
                                cotangent =
                                    rewriter.create<tensor::InsertOp>(loc, one, cotangent, indices);
                            }
                        }
                    }

                    cotangents.push_back(cotangent);
                }
                else {
                    // Result type is a float type
                    Value one =
                        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(resultType, 1.0));
                    using llvm::errs;
                    cotangents.push_back(one);
                }
            }
            else {
                // If this result is not the one currently being differentiated, set its shadow to
                // zero.
                if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
                    Value cotangent =
                        rewriter.create<tensor::EmptyOp>(loc, tensorType, ValueRange{});
                    Value zero = rewriter.create<arith::ConstantOp>(
                        loc, FloatAttr::get(tensorType.getElementType(), 0.0));
                    cotangent = rewriter.create<linalg::FillOp>(loc, zero, cotangent).getResult(0);
                    cotangents.push_back(cotangent);
                }
                else {
                    assert(false && "scalar returns not yet handled");
                }
            }
        }

        auto backpropOp = rewriter.create<gradient::BackpropOp>(
            loc, computeBackpropTypes(clonedCallee, diffArgIndices), clonedCallee.getName(),
            op.getArgOperands(),
            /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
            op.getDiffArgIndicesAttr());
        backpropResults.append(backpropOp.result_begin(), backpropOp.result_end());

        // auto tensorType = cast<RankedTensorType>(resultType);
        // assert(tensorType.hasStaticShape());
        // assert(tensorType.getRank() == 0 && "Expected 0-D tensor");
        // Value cotangent =
        //     rewriter.create<tensor::EmptyOp>(loc, tensorType, /*dynamicSizes=*/ValueRange{});
        // Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(1.0));
        // cotangent = rewriter.create<linalg::FillOp>(loc, one, cotangent).getResult(0);
        // cotangents.push_back(cotangent);
    }

    rewriter.replaceOp(op, backpropResults);
}

std::pair<int64_t, int64_t> ParameterShiftLowering::analyzeFunction(func::FuncOp callee)
{
    int64_t numShifts = 0;
    int64_t loopLevel = 0;
    int64_t maxLoopDepth = 0;

    callee.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<scf::ForOp>(op)) {
            loopLevel++;
        }
        else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
            if (gate.getDiffParams().empty())
                return;

            numShifts += gate.getDiffParams().size();
            maxLoopDepth = std::max(loopLevel, maxLoopDepth);
        }
        else if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp())) {
            loopLevel--;
        }
    });

    return {numShifts, maxLoopDepth};
}

} // namespace gradient
} // namespace catalyst
