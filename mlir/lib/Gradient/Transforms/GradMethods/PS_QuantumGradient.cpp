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

#include <deque>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"

namespace catalyst {
namespace gradient {

/// Store the given iteration variables in the selector vector.
static void updateSelectorVector(PatternRewriter &rewriter, Location loc,
                                 std::vector<std::pair<scf::ForOp, int64_t>> &selectorsToStore,
                                 Value selectorBuffer)
{
    PatternRewriter::InsertionGuard insertGuard(rewriter);

    for (auto &[forOp, idx] : selectorsToStore) {
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value iteration = forOp.getInductionVar();
        Value idxVal = rewriter.create<index::ConstantOp>(loc, idx);
        rewriter.create<memref::StoreOp>(loc, iteration, selectorBuffer, idxVal);
    }

    selectorsToStore.clear();
}

/// Generate calls to the shifted function to compute the current gradient element.
static std::vector<Value> computePartialDerivative(PatternRewriter &rewriter, Location loc,
                                                   int64_t numShifts, int64_t currentShift,
                                                   Value selectorBuffer, func::FuncOp shiftedFn,
                                                   std::vector<Value> callArgs)
{
    constexpr double shift = PI / 2;
    ShapedType shiftVectorType = RankedTensorType::get({numShifts}, rewriter.getF64Type());
    Value selectorVector = rewriter.create<bufferization::ToTensorOp>(loc, selectorBuffer);

    // Define the shift vectors (pos/neg) as sparse tensor constants.
    DenseElementsAttr nonZeroIndices = rewriter.getI64TensorAttr(currentShift);

    DenseElementsAttr nonZeroValuesPos =
        DenseFPElementsAttr::get(RankedTensorType::get(1, rewriter.getF64Type()), shift);
    TypedAttr shiftVectorAttrPos =
        SparseElementsAttr::get(shiftVectorType, nonZeroIndices, nonZeroValuesPos);
    Value shiftVectorPos = rewriter.create<arith::ConstantOp>(loc, shiftVectorAttrPos);

    DenseElementsAttr nonZeroValuesNeg =
        DenseFPElementsAttr::get(RankedTensorType::get(1, rewriter.getF64Type()), -shift);
    TypedAttr shiftVectorAttrNeg =
        SparseElementsAttr::get(shiftVectorType, nonZeroIndices, nonZeroValuesNeg);
    Value shiftVectorNeg = rewriter.create<arith::ConstantOp>(loc, shiftVectorAttrNeg);

    // Compute the partial derivate for this parameter via the simplified
    // parameter-shift rule: df/dx = [f(x + pi/2) - f(x - pi/2)] / 2.
    callArgs.push_back(shiftVectorPos);
    callArgs.push_back(selectorVector);
    ValueRange evalPos = rewriter.create<func::CallOp>(loc, shiftedFn, callArgs).getResults();

    callArgs[callArgs.size() - 2] = shiftVectorNeg;
    ValueRange evalNeg = rewriter.create<func::CallOp>(loc, shiftedFn, callArgs).getResults();

    std::vector<Value> derivatives;
    derivatives.reserve(evalPos.size());

    for (size_t i = 0; i < evalPos.size(); i++) {
        Value diff = rewriter.create<arith::SubFOp>(loc, evalPos[i], evalNeg[i]);
        Value divisor = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(2.0));
        if (auto tensorType = evalPos[i].getType().dyn_cast<TensorType>())
            divisor = rewriter.create<tensor::SplatOp>(loc, divisor, tensorType);
        derivatives.push_back(rewriter.create<arith::DivFOp>(loc, diff, divisor));
    }

    return derivatives;
}

/// Store a partial derivative in the gradient buffer at the next index.
static void storePartialDerivative(PatternRewriter &rewriter, Location loc,
                                   ValueRange gradientBuffers, Value gradientsProcessed,
                                   ValueRange derivatives)
{
    Value gradIdx = rewriter.create<memref::LoadOp>(loc, gradientsProcessed);

    for (size_t i = 0; i < gradientBuffers.size(); i++) {
        Value gradientBuffer = gradientBuffers[i];
        Value derivative = derivatives[i];
        bool isDerivativeTensor = derivative.getType().isa<TensorType>();
        bool isDerivativeScalarTensor =
            isDerivativeTensor && derivative.getType().cast<TensorType>().getRank() == 0;
        if (isDerivativeTensor && !isDerivativeScalarTensor) {
            // In the case of tensor return values, we have to do some extra work to
            // extract a view of the gradient buffer corresponding to a partially
            // indexed array. For example, with a gradient of type tensor<?x4x6xf32> we
            // need a view into the buffer of type tensor<4x6xf32>.
            //
            // The SubView op then needs a set of indices for sizes, offsets, and
            // strides for each dimension of the original shape, where the size of the
            // first dimension should be one, and the offset for the first dimension the
            // index of the desired slice. For the given example:
            //   sizes = [1, 4, 6]
            //   offsets = [idx, 0, 0]
            //   strides = [1, 1, 1]
            // This yields a subview of shape <1x4x6xf32>, where size 1 dimensions can
            // further be discarded by omitting them in the provided result type.
            //
            // Note that dynamic values for these arrays are specifically marked (with kDynamic)
            // and need to be provided separately as SSA values (e.g. from a tensor.dim op).
            MemRefType gradientBufferType = gradientBuffer.getType().cast<MemRefType>();
            int64_t rank = gradientBufferType.getRank();

            std::vector<int64_t> sizes = gradientBufferType.getShape();
            sizes[0] = 1;
            std::vector<Value> dynSizes;
            for (int64_t dim = 1; dim < rank; dim++) {
                if (sizes[dim] == ShapedType::kDynamic) {
                    Value idx = rewriter.create<index::ConstantOp>(loc, dim);
                    Value dimSize = rewriter.create<tensor::DimOp>(loc, gradientBuffer, idx);
                    dynSizes.push_back(dimSize);
                }
            }

            std::vector<int64_t> offsets(rank, 0);
            offsets[0] = ShapedType::kDynamic;
            std::vector<Value> dynOffsets = {gradIdx};

            std::vector<int64_t> strides(rank, 1);
            std::vector<Value> dynStrides = {};

            Type resultType = memref::SubViewOp::inferRankReducedResultType(
                gradientBufferType.getShape().drop_front(), gradientBufferType, offsets, sizes,
                strides);
            Value gradientSubview =
                rewriter.create<memref::SubViewOp>(loc, resultType, gradientBuffer, dynOffsets,
                                                   dynSizes, dynStrides, offsets, sizes, strides);
            rewriter.create<memref::TensorStoreOp>(loc, derivative, gradientSubview);
        }
        else if (isDerivativeScalarTensor) {
            Value extracted = rewriter.create<tensor::ExtractOp>(loc, derivative);
            rewriter.create<memref::StoreOp>(loc, extracted, gradientBuffer, gradIdx);
        }
        else {
            rewriter.create<memref::StoreOp>(loc, derivative, gradientBuffer, gradIdx);
        }
    }

    Value cOne = rewriter.create<index::ConstantOp>(loc, 1);
    Value newGradIdx = rewriter.create<index::AddOp>(loc, gradIdx, cOne);
    rewriter.create<memref::StoreOp>(loc, newGradIdx, gradientsProcessed);
}

func::FuncOp ParameterShiftLowering::genQGradFunction(PatternRewriter &rewriter, Location loc,
                                                      func::FuncOp callee, func::FuncOp shiftedFn,
                                                      const int64_t numShifts,
                                                      const int64_t loopDepth)
{
    // Define the properties of the quantum gradient function. The shape of the returned
    // gradient is unknown as the number of gate parameters in the unrolled circuit is only
    // determined at run time. The dynamic size is an input to the gradient function.
    std::string fnName = callee.getSymName().str() + ".qgrad";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    Type gradientSizeType = rewriter.getIndexType();
    fnArgTypes.push_back(gradientSizeType);
    const std::vector<Type> &gradResTypes = computeQGradTypes(callee);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, gradResTypes);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp gradientFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!gradientFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);

        gradientFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);

        // First copy the entire function as is, then we can modify it to compute the gradient.
        rewriter.cloneRegionBefore(callee.getBody(), gradientFn.getBody(), gradientFn.end());

        const std::vector<Value> callArgs(gradientFn.getArguments().begin(),
                                          gradientFn.getArguments().end());
        Value gradientSize = gradientFn.getBlocks().front().addArgument(gradientSizeType, loc);

        // Allocate the memory for the selector and gradient vectors and define some constants.
        // All shift vectors will be created as constants locally since they consist of only a
        // single non-zero compile-time constant.
        Value cZero, cOne;
        Value selectorBuffer;
        std::vector<Value> gradientBuffers;
        gradientBuffers.reserve(gradResTypes.size());
        Value gradientsProcessed;
        MemRefType selectorBufferType = MemRefType::get({loopDepth}, rewriter.getIndexType());
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(&gradientFn.getBody().front());

            cZero = rewriter.create<index::ConstantOp>(loc, 0);
            cOne = rewriter.create<index::ConstantOp>(loc, 1);

            // Use stack allocation for selector vector as it's not expected to be too big.
            selectorBuffer = rewriter.create<memref::AllocaOp>(loc, selectorBufferType);

            gradientsProcessed = rewriter.create<memref::AllocaOp>(
                loc, MemRefType::get({}, rewriter.getIndexType()));
            rewriter.create<memref::StoreOp>(loc, cZero, gradientsProcessed);

            for (Type gradType : gradResTypes) {
                TensorType gradTensorType = gradType.cast<TensorType>();
                MemRefType gradBufferType =
                    MemRefType::get(gradTensorType.getShape(), gradTensorType.getElementType());

                // TODO: add support for dynamic result dimensions
                gradientBuffers.push_back(
                    rewriter.create<memref::AllocOp>(loc, gradBufferType, gradientSize));
            }
        }

        int64_t currentShift = 0;
        int64_t loopLevel = 0;
        std::vector<std::pair<scf::ForOp, int64_t>> selectorsToStore;

        // Traverse nested IR in pre-order so that selectors for loops are handled
        // before entering the loop body.
        gradientFn.walk<WalkOrder::PreOrder>([&](Operation *op) {
            if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                selectorsToStore.push_back({forOp, loopLevel});
                loopLevel++;
            }
            else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                size_t numParams = gate.getDiffParams().size();
                if (numParams) {
                    updateSelectorVector(rewriter, loc, selectorsToStore, selectorBuffer);

                    for (size_t _ = 0; _ < numParams; _++) {
                        const std::vector<Value> &derivatives =
                            computePartialDerivative(rewriter, loc, numShifts, currentShift++,
                                                     selectorBuffer, shiftedFn, callArgs);
                        storePartialDerivative(rewriter, loc, gradientBuffers, gradientsProcessed,
                                               derivatives);
                    }
                }
            }
            else if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp())) {
                // In case there were no gate parameters in this for loop we need to pop the
                // current iteration variable so it's not written to memory at the next gate.
                scf::ForOp forOp = cast<scf::ForOp>(op->getParentOp());
                if (!selectorsToStore.empty() && selectorsToStore.back().first == forOp) {
                    selectorsToStore.pop_back();
                }
                loopLevel--;
            }
            else if (isa<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(op);

                std::vector<Value> gradientTensors;
                gradientTensors.reserve(gradResTypes.size());
                for (Value gradientBuffer : gradientBuffers) {
                    gradientTensors.push_back(
                        rewriter.create<bufferization::ToTensorOp>(loc, gradientBuffer));
                }
                op->setOperands(gradientTensors);
            }
        });

        // Finally erase all quantum operations.
        gradientFn.walk([&](Operation *op) {
            if (isa<quantum::DeviceInitOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
                // We are undoing the def-use chains of this gate's return values
                // so that we can safely delete it (all quantum ops must be eliminated).
                rewriter.replaceOp(gate, gate.getQubitOperands());
            }
            else if (auto region = dyn_cast<quantum::QuantumRegion>(op)) {
                rewriter.replaceOp(op, region.getRegisterOperand());
            }
            else if (isa<quantum::DeallocOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (isa<quantum::DeviceReleaseOp>(op)) {
                rewriter.eraseOp(op);
            }
        });

        quantum::removeQuantumMeasurements(gradientFn);
        gradientFn->setAttr("QuantumFree", rewriter.getUnitAttr());
    }

    return gradientFn;
}

} // namespace gradient
} // namespace catalyst
