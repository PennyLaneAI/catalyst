// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "scalarize-tensor-extracts"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

// Gate parameters computed from runtime inputs (e.g. Trotterization with
// runtime Hamiltonian coefficients) arrive as long chains of small tensor ops
// consumed only by scalar `tensor.extract` operations; each such tensor
// survives bufferization as an allocation plus copies. This pass sinks
// `tensor.extract` through `linalg.generic` (inlining the scalar payload),
// `tensor.collapse_shape`, and `tensor.extract_slice`, so the extracted
// element is computed in scalar arithmetic and the tensors become dead.
// Payload inlining is limited to small statically-shaped results to bound
// code growth.

namespace {

/// Upper bound on the number of elements of a tensor whose producer payload we
/// are willing to clone per extraction site. 16 covers the 4x4 matrices of
/// two-qubit gates, the largest tensors in gate-parameter dataflow, while
/// keeping the worst-case code growth per extract small.
constexpr int64_t kMaxScalarizedElements = 16;

/// Fold tensor.extract(linalg.generic) by inlining the generic's scalar payload
/// at the extraction point, for elementwise (all-parallel) generics.
struct ExtractOfGeneric : public OpRewritePattern<tensor::ExtractOp> {
    using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                  PatternRewriter &rewriter) const override
    {
        auto genericOp = extractOp.getTensor().getDefiningOp<linalg::GenericOp>();
        if (!genericOp) {
            return failure();
        }

        // Only elementwise generics: every iterator is parallel.
        if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
            return failure();
        }

        // Restrict to small statically shaped results to bound code growth.
        auto resultType = dyn_cast<RankedTensorType>(extractOp.getTensor().getType());
        if (!resultType || !resultType.hasStaticShape() ||
            resultType.getNumElements() > kMaxScalarizedElements) {
            return failure();
        }

        // Identify which result of the generic is being extracted and require an
        // identity indexing map for it, so the iteration indices equal the
        // extraction indices.
        auto resultNumber = cast<OpResult>(extractOp.getTensor()).getResultNumber();
        OpOperand *initOperand = genericOp.getDpsInitOperand(resultNumber);
        AffineMap outputMap = genericOp.getMatchingIndexingMap(initOperand);
        if (!outputMap.isIdentity()) {
            return failure();
        }

        Block *body = genericOp.getBody();

        // The payload must be speculatable scalar code and must not read the
        // accumulator (output block argument).
        for (Operation &op : body->without_terminator()) {
            if (!isPure(&op)) {
                return failure();
            }
        }
        for (OpOperand &outOperand : genericOp.getDpsInitsMutable()) {
            BlockArgument outArg = body->getArgument(outOperand.getOperandNumber());
            if (!outArg.use_empty()) {
                return failure();
            }
        }

        Location loc = extractOp.getLoc();
        SmallVector<Value> iterIndices(extractOp.getIndices());

        // Materialize scalar operands: one tensor.extract per generic input, at
        // indices given by composing that input's indexing map with the
        // extraction indices.
        IRMapping mapping;
        for (OpOperand *inOperand : genericOp.getDpsInputOperands()) {
            BlockArgument blockArg = body->getArgument(inOperand->getOperandNumber());
            Value input = inOperand->get();
            if (!isa<RankedTensorType>(input.getType())) {
                // Scalar operands of the generic map through unchanged.
                mapping.map(blockArg, input);
                continue;
            }
            AffineMap inputMap = genericOp.getMatchingIndexingMap(inOperand);
            SmallVector<Value> inputIndices;
            for (AffineExpr expr : inputMap.getResults()) {
                if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                    inputIndices.push_back(iterIndices[dimExpr.getPosition()]);
                }
                else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
                    inputIndices.push_back(
                        arith::ConstantIndexOp::create(rewriter, loc, constExpr.getValue()));
                }
                else {
                    return failure();
                }
            }
            Value scalar = tensor::ExtractOp::create(rewriter, loc, input, inputIndices);
            mapping.map(blockArg, scalar);
        }

        // Clone the payload, resolving linalg.index to the extraction indices.
        for (Operation &op : body->without_terminator()) {
            if (auto indexOp = dyn_cast<linalg::IndexOp>(op)) {
                mapping.map(indexOp.getResult(), iterIndices[indexOp.getDim()]);
                continue;
            }
            rewriter.clone(op, mapping);
        }

        auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
        Value result = mapping.lookupOrDefault(yieldOp.getOperand(resultNumber));
        rewriter.replaceOp(extractOp, result);
        return success();
    }
};

/// Fold tensor.extract(tensor.collapse_shape) for collapses that only drop or
/// merge unit dimensions (at most one non-unit dimension per reassociation
/// group), by extracting directly from the source.
struct ExtractOfCollapseShape : public OpRewritePattern<tensor::ExtractOp> {
    using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                  PatternRewriter &rewriter) const override
    {
        auto collapseOp = extractOp.getTensor().getDefiningOp<tensor::CollapseShapeOp>();
        if (!collapseOp) {
            return failure();
        }
        auto srcType = collapseOp.getSrcType();
        if (!srcType.hasStaticShape()) {
            return failure();
        }

        Location loc = extractOp.getLoc();
        SmallVector<Value> srcIndices(srcType.getRank());

        Value zero;
        auto getZero = [&]() {
            if (!zero) {
                zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
            }
            return zero;
        };

        SmallVector<ReassociationIndices> groups = collapseOp.getReassociationIndices();
        // A rank-0 result means every source dimension is a unit dimension.
        if (groups.empty()) {
            for (int64_t dim = 0; dim < srcType.getRank(); ++dim) {
                srcIndices[dim] = getZero();
            }
        }
        for (const auto &[groupIdx, group] : llvm::enumerate(groups)) {
            int64_t nonUnitDim = -1;
            for (int64_t srcDim : group) {
                if (srcType.getDimSize(srcDim) != 1) {
                    if (nonUnitDim != -1) {
                        return failure(); // true merge of two non-unit dims
                    }
                    nonUnitDim = srcDim;
                }
            }
            for (int64_t srcDim : group) {
                srcIndices[srcDim] =
                    (srcDim == nonUnitDim) ? extractOp.getIndices()[groupIdx] : getZero();
            }
        }

        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(extractOp, collapseOp.getSrc(), srcIndices);
        return success();
    }
};

/// Fold tensor.extract(tensor.extract_slice) by extracting from the source at
/// offset + index * stride.
struct ExtractOfExtractSlice : public OpRewritePattern<tensor::ExtractOp> {
    using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                  PatternRewriter &rewriter) const override
    {
        auto sliceOp = extractOp.getTensor().getDefiningOp<tensor::ExtractSliceOp>();
        if (!sliceOp) {
            return failure();
        }

        Location loc = extractOp.getLoc();
        int64_t srcRank = sliceOp.getSourceType().getRank();

        // The slice may be rank-reducing: map each source dim to its position in
        // the result (or none if the dim was dropped).
        llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();

        auto materialize = [&](OpFoldResult ofr) -> Value {
            if (auto val = dyn_cast<Value>(ofr)) {
                return val;
            }
            return arith::ConstantIndexOp::create(rewriter, loc,
                                                  cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
        };

        SmallVector<Value> srcIndices;
        unsigned resultDim = 0;
        for (int64_t dim = 0; dim < srcRank; ++dim) {
            Value offset = materialize(sliceOp.getMixedOffsets()[dim]);
            if (droppedDims.test(dim)) {
                srcIndices.push_back(offset);
                continue;
            }
            Value index = extractOp.getIndices()[resultDim++];
            Value stride = materialize(sliceOp.getMixedStrides()[dim]);
            Value scaled = arith::MulIOp::create(rewriter, loc, index, stride);
            srcIndices.push_back(arith::AddIOp::create(rewriter, loc, offset, scaled));
        }

        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(extractOp, sliceOp.getSource(), srcIndices);
        return success();
    }
};

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_SCALARIZETENSOREXTRACTSPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct ScalarizeTensorExtractsPass
    : public impl::ScalarizeTensorExtractsPassBase<ScalarizeTensorExtractsPass> {
    using impl::ScalarizeTensorExtractsPassBase<
        ScalarizeTensorExtractsPass>::ScalarizeTensorExtractsPassBase;

    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<ExtractOfGeneric, ExtractOfCollapseShape, ExtractOfExtractSlice>(context);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace catalyst
