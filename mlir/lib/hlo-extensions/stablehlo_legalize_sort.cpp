// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is taken from the
//   tensorflow/mlir-hlo
// repository, under the Apache 2.0 License, at
//   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/legalize_sort/legalize_sort.cc
// with the following copyright notice:

/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// The modifications are porting the pass from the upstream stablehlo namespace to
// catalyst namespace.

// This file implements logic for lowering stablehlo.sort to the SCF dialect.
#include <iterator>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

#include "hlo-extensions/Passes.h"

using namespace mlir;
using namespace stablehlo;
using namespace catalyst;

namespace catalyst {

#define GEN_PASS_DEF_STABLEHLOLEGALIZESORTPASS
#define GEN_PASS_DECL_STABLEHLOLEGALIZESORTPASS
#include "hlo-extensions/Passes.h.inc"

} // namespace catalyst

namespace {

using ::mlir::arith::AddIOp;
using ::mlir::arith::MinSIOp;
using ::mlir::arith::SelectOp;

constexpr int64_t kInsertionSortSize = 16;

// Inlines the `comparator` region (without terminator) at the current insertion
// point, replacing the arguments with the given values from `lhs` and `rhs`.
Value emitComparison(ImplicitLocOpBuilder &b, SmallVector<Value> &lhs, SmallVector<Value> &rhs,
                     Region &comparator)
{
    assert(comparator.hasOneBlock() && "Comparator must have only one block.");
    Block &block = comparator.front();
    assert(block.getTerminator()->getOperands().size() == 1 &&
           "Comparator must return a single value");

    IRMapping mapping;
    for (auto [idx, arg] : llvm::enumerate(comparator.getArguments())) {
        Value value = idx % 2 == 0 ? lhs[idx / 2] : rhs[idx / 2];
        Type type = RankedTensorType::get({}, value.getType());
        mapping.map(arg, b.create<tensor::FromElementsOp>(type, value));
    }

    for (Operation &op : block.without_terminator())
        b.clone(op, mapping);
    Value result = mapping.lookup(block.getTerminator()->getOperands().front());

    return b.create<tensor::ExtractOp>(result, ValueRange());
}

// Emits a binary search of `pivots` in `arrayMemrefs` (all rank 1) in the range
// [`left`;`right`). `arrayMemrefs` must be sorted according to `comparator`.
Value emitBinarySearch(ImplicitLocOpBuilder &b, Value leftInit, Value rightInit,
                       SmallVector<Value> &pivots, ValueRange arrayMemrefs, Region &comparator)
{
    SmallVector<Type, 2> types{leftInit.getType(), rightInit.getType()};
    ArithBuilder arith(b, b.getLoc());

    // while (
    auto whileOp = b.create<scf::WhileOp>(types, SmallVector<Value, 2>{leftInit, rightInit});
    OpBuilder::InsertionGuard guard(b);

    //        left < right) {
    Block *before =
        b.createBlock(&whileOp.getBefore(), {}, types, {whileOp.getLoc(), whileOp.getLoc()});
    {
        Value left = before->getArgument(0), right = before->getArgument(1);
        b.setInsertionPointToEnd(before);
        b.create<scf::ConditionOp>(arith.slt(left, right), before->getArguments());
    }

    Block *after =
        b.createBlock(&whileOp.getAfter(), {}, types, {whileOp.getLoc(), whileOp.getLoc()});
    {
        Value left = after->getArgument(0), right = after->getArgument(1);
        b.setInsertionPointToEnd(after);
        //   int mid = (left + right) >> 1;
        Value one = b.create<arith::ConstantIndexOp>(1);
        Value mid = b.create<arith::ShRUIOp>(arith.add(left, right), one);
        Value midPlusOne = b.create<AddIOp>(mid, one);

        auto arraysAtMid =
            llvm::to_vector(llvm::map_range(arrayMemrefs, [&](Value arrayMemref) -> Value {
                return b.create<memref::LoadOp>(arrayMemref, mid);
            }));
        Value cond = emitComparison(b, pivots, arraysAtMid, comparator);
        //   if (comparator(pivot, array[mid]))
        //     right = mid;
        //   else
        //     left = mid + 1;
        Value newLeft = arith.select(cond, left, midPlusOne);
        Value newRight = arith.select(cond, mid, right);

        // }
        b.create<scf::YieldOp>(ValueRange{newLeft, newRight});
    }

    return whileOp.getResult(0);
}

SmallVector<Value> loadTensorElements(ImplicitLocOpBuilder &b, ValueRange tensors, Value index)
{
    return llvm::to_vector(llvm::map_range(tensors, [&](Value tensor) -> Value {
        return b.create<tensor::ExtractOp>(tensor, index);
    }));
}

SmallVector<Value> loadMemrefElements(ImplicitLocOpBuilder &b, ValueRange memrefs, Value index)
{
    return llvm::to_vector(llvm::map_range(memrefs, [&](Value memref) -> Value {
        Type type = mlir::cast<MemRefType>(memref.getType()).getElementType();
        return b.create<memref::LoadOp>(type, memref, index);
    }));
}

void storeMemrefElements(ImplicitLocOpBuilder &b, ValueRange memrefs, Value index,
                         ValueRange values)
{
    for (auto [value, memref] : llvm::zip(values, memrefs)) {
        b.create<memref::StoreOp>(value, memref, index);
    }
}

// Insertion sorts `inputTensors` in the range [`lo`; `hi`), storing the results
// in `outputMemrefs`. `inputTensors` and `outputMemrefs` must all be rank 1 and
// of identical size.
void emitInsertionSort(ImplicitLocOpBuilder &b, Value lo, Value hi, ValueRange inputTensors,
                       ValueRange outputMemrefs, mlir::Region &comparator)
{
    ArithBuilder arith(b, b.getLoc());
    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);

    // array[lo] = tensors[lo];
    storeMemrefElements(b, outputMemrefs, lo, loadTensorElements(b, inputTensors, lo));

    // for (int start = lo + 1; start < hi; ++start)
    {
        auto forOp = b.create<scf::ForOp>(arith.add(lo, one), hi, one);
        OpBuilder::InsertionGuard outerGuard(b);
        b.setInsertionPointToStart(forOp.getBody());
        Value start = forOp.getInductionVar();

        //   T pivot = tensors[start];
        auto pivots = loadTensorElements(b, inputTensors, start);

        //   int index = binarySearch(lo, start, pivot, array, comparator);
        auto index = emitBinarySearch(b, lo, start, pivots, outputMemrefs, comparator);

        //   int n = start - index;  // The number of elements to move
        Value n = arith.sub(start, index);

        // memmove(&array[index + 1], &array[index], n * sizeof(T))
        // memref::CopyOp would be nice to use here, but:
        // 1. It lowers to a quite inefficient library call in the general case
        //    (strides != 1).
        // 2. It implements memcpy semantics, but we need memmove here.
        // So we go with a loop instead.
        auto copyForOp = b.create<scf::ForOp>(zero, n, one);
        {
            OpBuilder::InsertionGuard innerGuard(b);
            b.setInsertionPointToStart(copyForOp.getBody());
            Value copyLoopIndex = copyForOp.getBody()->getArgument(0);

            Value dstIndex = arith.sub(start, copyLoopIndex);
            Value srcIndex = arith.sub(dstIndex, one);
            storeMemrefElements(b, outputMemrefs, dstIndex,
                                loadMemrefElements(b, outputMemrefs, srcIndex));
        }
        //   array[index] = pivot;
        storeMemrefElements(b, outputMemrefs, index, pivots);
    }
}

void emitMerge(ImplicitLocOpBuilder &b, Value lo, Value mid, Value hi, ValueRange readBufs,
               ValueRange writeBufs, mlir::Region &comparator)
{
    ArithBuilder arith(b, b.getLoc());
    // The while loop runs until we reach the end of either interval. It has three
    // loop-carried variables:
    // 1. current output index
    // 2. current read index for interval 1
    // 3. current read index for interval 2
    SmallVector<Type> whileArgTypes{lo.getType(), lo.getType(), mid.getType()};
    SmallVector<Value> whileInitArgs{lo, lo, mid};
    SmallVector<Location> whileArgLocs(whileArgTypes.size(), b.getLoc());

    // while(
    auto whileOp = b.create<scf::WhileOp>(whileArgTypes, whileInitArgs);
    {
        OpBuilder::InsertionGuard guard(b);
        {
            Block *before = b.createBlock(&whileOp.getBefore(), {}, whileArgTypes, whileArgLocs);
            Value i0 = before->getArgument(1), i1 = before->getArgument(2);
            b.setInsertionPointToEnd(before);

            //     i0 < mid && i1 < hi) {
            Value inbounds0 = arith.slt(i0, mid);
            Value inbounds1 = arith.slt(i1, hi);

            b.create<scf::ConditionOp>(arith._and(inbounds0, inbounds1), before->getArguments());
        }

        {
            Block *after = b.createBlock(&whileOp.getAfter(), {}, whileArgTypes, whileArgLocs);
            Value iOut = after->getArgument(0), i0 = after->getArgument(1),
                  i1 = after->getArgument(2);
            b.setInsertionPointToEnd(after);

            //   auto vals0 = readBufs[i0], vals1 = readBufs[i1];
            SmallVector<Value> vals0 = loadMemrefElements(b, readBufs, i0);
            SmallVector<Value> vals1 = loadMemrefElements(b, readBufs, i1);

            //   writeBufs[iOut] = comparator(vals1, vals0)
            //                       ? readBufs[i1++] : readBufs[i0++];
            Value cmp = emitComparison(b, vals1, vals0, comparator);
            SmallVector<Value> pickedVals;
            for (auto [val0, val1] : llvm::zip(vals0, vals1)) {
                pickedVals.push_back(b.create<SelectOp>(cmp, val1, val0));
            }
            storeMemrefElements(b, writeBufs, iOut, pickedVals);

            Value one = b.create<arith::ConstantIndexOp>(1);
            Value nexti0 = b.create<SelectOp>(cmp, i0, arith.add(i0, one));
            Value nexti1 = b.create<SelectOp>(cmp, arith.add(i1, one), i1);
            //   ++iOut;
            Value nextIOut = b.create<AddIOp>(iOut, one);
            b.create<scf::YieldOp>(ValueRange{nextIOut, nexti0, nexti1});
        }
    }

    // At this point, exactly one of the input ranges will have leftover elements.
    Value iOut = whileOp->getResult(0);
    Value i0 = whileOp->getResult(1);
    Value i1 = whileOp->getResult(2);

    // We could use memref::CopyOp here, but typically, there aren't many leftover
    // elements for randomly shuffled inputs.
    Value leftoverIn0 = arith.slt(i0, mid);
    Value start = arith.select(leftoverIn0, i0, i1);
    Value end = arith.select(leftoverIn0, mid, hi);
    Value n = arith.sub(end, start);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);
    auto forOp = b.create<scf::ForOp>(zero, n, one);
    b.setInsertionPointToStart(forOp.getBody());
    Value copyIndex = forOp.getBody()->getArgument(0);

    Value srcIndex = arith.add(start, copyIndex);
    Value dstIndex = arith.add(iOut, copyIndex);
    storeMemrefElements(b, writeBufs, dstIndex, loadMemrefElements(b, readBufs, srcIndex));
}

// Emits a bottom up merge sort of `inputTensors` in the range [`lo`; `hi`), and
// writes the results to either `outputs0` or `outputs1`.
// Returns 0 if the results are in `outputs0`, 1 if they are in `outputs1`.
// TODO(jreiffers): Consider implementing top-down merge sort.
Value emitBottomUpMergeSort(ImplicitLocOpBuilder &b, Value lo, Value hi, int64_t staticSortDimSize,
                            ValueRange inputTensors, ValueRange outputs0, ValueRange outputs1,
                            mlir::Region &comparator)
{
    ArithBuilder arith(b, b.getLoc());
    Value size = arith.sub(hi, lo);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value insertionSortSize = b.create<arith::ConstantIndexOp>(kInsertionSortSize);

    // Run insertion sort on blocks of size kInsertionSortSize.
    // for (int start = 0; start < size; start += kInsertionSortSize) {
    {
        auto forOp = b.create<scf::ForOp>(zero, size, insertionSortSize);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(forOp.getBody());
        Value start = forOp.getBody()->getArgument(0);
        Value end = arith.add(b.create<MinSIOp>(arith.add(start, insertionSortSize), size), lo);
        emitInsertionSort(b, start, end, inputTensors, outputs0, comparator);
    }

    Value initParity = b.create<arith::ConstantIntOp>(0, 1);
    if (staticSortDimSize >= 0 && staticSortDimSize < kInsertionSortSize) {
        return initParity;
    }

    // The while arguments are:
    // 1. the current size
    // 2. the original index of the buffers we're currently reading from
    // 3. the buffers we're currently reading from
    // 4. the buffers we're currently writing to.
    //
    // 1 gets doubled each iteration, 2 gets negated, 3 and 4 are swapped.
    // int currentSize = 16;
    SmallVector<Value> whileInitArgs{insertionSortSize, initParity};
    // First we read from `outputs0` (initialized by the insertion sort above).
    llvm::copy(outputs0, std::back_inserter(whileInitArgs));
    llvm::copy(outputs1, std::back_inserter(whileInitArgs));

    SmallVector<Type> whileArgTypes;
    for (auto val : whileInitArgs)
        whileArgTypes.push_back(val.getType());

    SmallVector<Location> whileArgLocs(whileArgTypes.size(), b.getLoc());

    // while (
    auto whileOp = b.create<scf::WhileOp>(whileArgTypes, whileInitArgs);
    OpBuilder::InsertionGuard guard(b);

    //        currentSize < totalSize)
    {
        Block *before = b.createBlock(&whileOp.getBefore(), {}, whileArgTypes, whileArgLocs);
        Value currentSize = before->getArgument(0);
        b.setInsertionPointToEnd(before);
        b.create<scf::ConditionOp>(arith.slt(currentSize, size), before->getArguments());
    }

    size_t numArgs = inputTensors.size();
    //                                 {
    {
        Block *after = b.createBlock(&whileOp.getAfter(), {}, whileArgTypes, whileArgLocs);

        Value currentSize = after->getArgument(0);
        Value parity = after->getArgument(1);
        auto readBufs = after->getArguments().drop_front(2).take_front(numArgs);
        auto writeBufs = after->getArguments().take_back(numArgs);

        Value twoCurrentSize = arith.add(currentSize, currentSize);

        // for (int start = 0; start < size; start += 2*currentSize) {
        {
            auto forOp = b.create<scf::ForOp>(zero, size, twoCurrentSize);
            b.setInsertionPointToStart(forOp.getBody());
            Value start = forOp.getBody()->getArgument(0);

            Value mid = b.create<MinSIOp>(size, arith.add(start, currentSize));
            Value end = b.create<MinSIOp>(size, arith.add(start, twoCurrentSize));
            emitMerge(b, start, mid, end, readBufs, writeBufs, comparator);
            b.setInsertionPointAfter(forOp);
        }
        // }

        // parity = !parity;
        Value one = b.create<arith::ConstantIntOp>(1, 1);
        Value notParity = arith.sub(one, parity);
        // currentSize *= 2;
        SmallVector<Value> nextWhileArgs{twoCurrentSize, notParity};
        llvm::copy(writeBufs, std::back_inserter(nextWhileArgs));
        llvm::copy(readBufs, std::back_inserter(nextWhileArgs));
        b.create<scf::YieldOp>(nextWhileArgs);
    }
    // }

    // The result is the parity bit.
    return whileOp.getResults().drop_front(1).front();
}

// Helper struct for extracting 1d slices from tensors and memrefs.
struct Slicer {
    Slicer(OpBuilder &b, uint64_t sortDim, Value sortDimSize, ValueRange ivs)
        : sizes(ivs.size() + 1, b.getI64IntegerAttr(1)),
          strides(ivs.size() + 1, b.getI64IntegerAttr(1))
    {
        sizes[sortDim] = sortDimSize;
        for (size_t i = 0; i < ivs.size() + 1; ++i) {
            if (i == sortDim) {
                offsets.push_back(b.getI64IntegerAttr(0));
            }
            else {
                offsets.push_back(ivs[i - static_cast<int>(i > sortDim)]);
            }
        }
    }

    RankedTensorType toSlicedType(RankedTensorType sourceType)
    {
        return tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            /*resultRank=*/1, sourceType, offsets, sizes, strides);
    }

    MemRefType toSlicedType(MemRefType sourceType)
    {
        return mlir::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            {ShapedType::kDynamic} /*1D output*/, sourceType, offsets, sizes, strides));
    }

    template <typename Op, typename Ty> Value slice(ImplicitLocOpBuilder &b, Value input)
    {
        Ty ty = mlir::cast<Ty>(input.getType());
        return b.create<Op>(toSlicedType(ty), input, offsets, sizes, strides).getResult();
    }

    Value apply(ImplicitLocOpBuilder &b, Value input)
    {
        Type inTy = input.getType();
        if (mlir::isa<RankedTensorType>(inTy)) {
            return slice<tensor::ExtractSliceOp, RankedTensorType>(b, input);
        }
        assert(mlir::isa<MemRefType>(inTy));
        return slice<memref::SubViewOp, MemRefType>(b, input);
    }

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
};

SmallVector<Value> sliceMemrefsOrTensors(ImplicitLocOpBuilder &b, SmallVector<Value> &ivs,
                                         Value sortDimSize, ValueRange memrefsOrTensors, SortOp op)
{
    if (ivs.empty())
        return memrefsOrTensors;

    SmallVector<Value> outputs;
    Slicer slicer(b, op.getDimension(), sortDimSize, ivs);
    // Create subviews/slices.
    for (Value out : memrefsOrTensors) {
        outputs.push_back(slicer.apply(b, out));
    }

    return outputs;
}

struct SortOpPattern : public OpRewritePattern<SortOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(SortOp op, PatternRewriter &rewriter) const override
    {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        // Note: the output memrefs aren't necessarily the ones that we return,
        SmallVector<Value> outputMemrefs;
        SmallVector<Value> scratchMemrefs;

        Value firstOperand = op.getOperands().front();
        auto firstOperandType = mlir::cast<ShapedType>(firstOperand.getType());
        int64_t inputRank = firstOperandType.getRank();

        Value sortDimSize = b.createOrFold<tensor::DimOp>(
            firstOperand, b.create<arith::ConstantIndexOp>(op.getDimension()));
        int64_t staticSortDimSize = firstOperandType.getDimSize(op.getDimension());

        SmallVector<Value> dynamicDims;
        for (int i = 0; i < inputRank; ++i) {
            if (!firstOperandType.isDynamicDim(i))
                continue;
            Value index = b.create<arith::ConstantIndexOp>(i);
            Value dimOp = b.create<tensor::DimOp>(firstOperand, index);
            dynamicDims.push_back(dimOp);
        }

        // Allocate output and scratch memrefs. If the size of the sort dimension is
        // statically known to be <= kInsertionSortSize, `scratchMemrefs` are unused
        // and will be cleaned up later.
        for (auto input : op.getOperands()) {
            auto inputType = mlir::cast<ShapedType>(input.getType());
            auto memRefType = MemRefType::get(inputType.getShape(), inputType.getElementType());

            outputMemrefs.push_back(b.create<memref::AllocOp>(memRefType, dynamicDims));
            scratchMemrefs.push_back(b.create<memref::AllocOp>(memRefType, dynamicDims));
        }

        b.setInsertionPoint(op);
        Value zero = b.create<arith::ConstantIndexOp>(0);
        Value one = b.create<arith::ConstantIndexOp>(1);

        Value forInitArg = b.create<arith::ConstantIntOp>(0, 1);
        SmallVector<scf::ForOp> forOps;
        SmallVector<Value> ivs;
        forOps.reserve(inputRank - 1);
        ivs.reserve(inputRank - 1);
        for (int64_t i = 0; i < inputRank; ++i) {
            if (i != static_cast<int64_t>(op.getDimension())) {
                Value dim = b.create<arith::ConstantIndexOp>(i);
                Value ub = b.create<tensor::DimOp>(firstOperand, dim);
                scf::ForOp &forOp = forOps.emplace_back(
                    b.create<scf::ForOp>(zero, ub, one, ValueRange{forInitArg}));
                ivs.push_back(forOp.getInductionVar());
                b.setInsertionPointToStart(&forOp.getRegion().front());
            }
        }
        SmallVector<Value> inputs =
            sliceMemrefsOrTensors(b, ivs, sortDimSize, op.getOperands(), op);
        SmallVector<Value> outputs = sliceMemrefsOrTensors(b, ivs, sortDimSize, outputMemrefs, op);
        SmallVector<Value> scratches =
            sliceMemrefsOrTensors(b, ivs, sortDimSize, scratchMemrefs, op);

        Value parity = emitBottomUpMergeSort(b, zero, sortDimSize, staticSortDimSize, inputs,
                                             outputs, scratches, op.getRegion());

        // Pass the parity bit through the for loops.
        for (auto i = static_cast<int64_t>(forOps.size() - 1); i >= 0; --i) {
            b.setInsertionPointToEnd(&forOps[i].getRegion().front());
            b.create<scf::YieldOp>(ValueRange{parity});
            parity = forOps[i]->getResult(0);
        }
        b.setInsertionPoint(op);

        SmallVector<Value> outputTensors;
        for (auto [out0, out1] : llvm::zip(outputMemrefs, scratchMemrefs)) {
            Value s = b.create<SelectOp>(parity, out1, out0).getResult();
            outputTensors.push_back(b.create<bufferization::ToTensorOp>(
                memref::getTensorTypeFromMemRefType(s.getType()), s, /*restrict=*/true));
        }

        rewriter.replaceOp(op, outputTensors);
        return success();
    }
};

struct StablehloLegalizeSortPass
    : public catalyst::impl::StablehloLegalizeSortPassBase<StablehloLegalizeSortPass> {
    // Perform the lowering to MLIR control flow.
    void runOnOperation() override
    {
        func::FuncOp f = getOperation();
        MLIRContext *ctx = f.getContext();

        RewritePatternSet patterns(ctx);
        patterns.add<SortOpPattern>(ctx);

        mlir::ConversionTarget target(*ctx);
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addIllegalOp<stablehlo::SortOp>();

        if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> catalyst::createStablehloLegalizeSortPass()
{
    return std::make_unique<StablehloLegalizeSortPass>();
}
