// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "one-shot-mcm"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace stablehlo;
using namespace catalyst;

namespace {
void clearFuncExceptShots(func::FuncOp qfunc, Value shots)
{
    // Delete the body of a funcop, except the operations that produce the shots value
    // We need triple loop to explicitly iterate in reverse order, due to erasure.

    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    LogicalResult bsr = getBackwardSlice(shots, &backwardSlice, options);
    assert(bsr.succeeded() && "expected a backward slice");

    // For whatever reason, upstream mlir decided to not include the op itself into its backward
    // slice
    backwardSlice.insert(shots.getDefiningOp());

    SmallVector<Operation *> eraseWorklist;
    for (auto &region : qfunc->getRegions()) {
        for (auto &block : region.getBlocks()) {
            for (auto op = block.rbegin(); op != block.rend(); ++op) {
                if (!backwardSlice.contains(&*op)) {
                    eraseWorklist.push_back(&*op);
                }
            }
        }
    }

    for (Operation *op : eraseWorklist) {
        op->erase();
    }
}

func::FuncOp createOneShotKernel(IRRewriter &builder, func::FuncOp qfunc, Value shots,
                                 Operation *mod)
{
    OpBuilder::InsertionGuard guard(builder);
    Location loc = mod->getLoc();
    MLIRContext *ctx = mod->getContext();

    Type i64Type = builder.getI64Type();

    // 1. Clone the original quantum function and give it a new name
    // Because we need to make sure the current qfunc name is reserved as entry point
    // Set the number of shots in the new kernel to one.
    builder.setInsertionPointToStart(&qfunc->getParentOfType<ModuleOp>()->getRegion(0).front());
    auto qkernel = cast<func::FuncOp>(qfunc->clone());
    qkernel.setSymNameAttr(StringAttr::get(ctx, qfunc.getSymName() + ".quantum_kernel"));
    builder.insert(qkernel);

    builder.setInsertionPointToStart(&qkernel.getBody().front());
    auto kernelDeviceInitOp = *qkernel.getOps<quantum::DeviceInitOp>().begin();
    auto one = builder.create<arith::ConstantOp>(loc, i64Type, builder.getIntegerAttr(i64Type, 1));
    Value originalKernelShots = kernelDeviceInitOp.getShots();
    kernelDeviceInitOp->setOperand(0, one);
    if (originalKernelShots.getNumUses() == 0) {
        originalKernelShots.getDefiningOp()->erase();
    }

    // 2. Clear the original qfunc. Its new contents will be the one-shot logic.
    // Keep the SSA value for the shots. It needs to be used as the upper bound of the for
    // loop.
    clearFuncExceptShots(qfunc, shots);
    return qkernel;
}

scf::ForOp createForLoop(IRRewriter &builder, Value shots, ValueRange loopIterArgs)
{
    // Create a for loop op with an empty body that loops from 0 to num_shots with step size one

    OpBuilder::InsertionGuard guard(builder);
    Location loc = shots.getLoc();
    Type indexType = builder.getIndexType();

    auto lb =
        builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 0));
    auto step =
        builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 1));
    auto ub = builder.create<index::CastSOp>(loc, builder.getIndexType(), shots);
    auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step, loopIterArgs);

    return forOp;
}

void eraseAllUsers(Value v, Operation *check_erased)
{
    // Erase all users of a Value, and all users of these users, etc., in its forward slice.
    // Also check that the operation `check_erased` is one of the erased ops.

    // We need to use a SetVector, since during the erasure we have to delete from bottom up
    // So we need an ordered container
    SetVector<Operation *> forwardSlice;
    getForwardSlice(v, &forwardSlice);

    // Safety check
    assert(forwardSlice.contains(check_erased) &&
           "Expected the multi-shot MP of the mcm to be erased in the one-shot kernel.");

    for (auto op = forwardSlice.rbegin(); op != forwardSlice.rend(); ++op) {
        (*op)->erase();
    }
}

} // anonymous namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ONESHOTMCMPASS
#include "Quantum/Transforms/Passes.h.inc"

struct OneShotMCMPass : public impl::OneShotMCMPassBase<OneShotMCMPass> {
    using impl::OneShotMCMPassBase<OneShotMCMPass>::OneShotMCMPassBase;

    void editKernelMCMExpval(IRRewriter &builder, func::FuncOp qfunc, quantum::MCMObsOp mcmobs)
    {
        // If the kernel returns expval on a mcm,
        // the single-shot expval of a mcm is just the mcm boolean result itself
        // So we just cast it to the correct type and return it.

        OpBuilder::InsertionGuard guard(builder);
        Location loc = qfunc->getLoc();

        // Erase all users of the mcm obs: the new return does not need them.
        eraseAllUsers(mcmobs.getObs(), qfunc.getBody().back().getTerminator());

        // Cast the I1 mcm to the correct type and return it
        // MCM itself is I1
        // Expval kernel returns tensor<f64>
        // We need to cast as I1 -> I64 -> F64 -> tensor<f64>
        builder.setInsertionPointToEnd(&qfunc.getBody().back());
        Value mcm = mcmobs.getMcm();
        auto extuiOp = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), mcm);
        auto int2floatCastOp =
            builder.create<arith::SIToFPOp>(loc, builder.getF64Type(), extuiOp.getOut());
        auto fromElementsOp = builder.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get({}, builder.getF64Type()), int2floatCastOp.getResult());

        builder.create<func::ReturnOp>(loc, fromElementsOp.getResult());

        mcmobs->erase();
    }

    void editKernelProbsExpval(IRRewriter &builder, func::FuncOp qfunc, quantum::MCMObsOp mcmobs)
    {
        // If the kernel returns probs on a mcm,
        // the single-shot probs of the mcm is either [1,0] or [0,1]

        OpBuilder::InsertionGuard guard(builder);
        Location loc = qfunc->getLoc();

        Type f64Type = builder.getF64Type();

        // Erase all users of the mcm obs: they new return does not need them.
        eraseAllUsers(mcmobs.getObs(), qfunc.getBody().back().getTerminator());

        // Create the probs tensor and write the 1 into it
        // The location of the 1 is exactly the mcm result
        // i.e. mcm 0, probs is [1,0]
        // mcm 1, probs is [0,1]
        builder.setInsertionPointToEnd(&qfunc.getBody().back());

        // Create zero tensor
        auto zero =
            builder.create<arith::ConstantOp>(loc, f64Type, builder.getFloatAttr(f64Type, 0));
        auto zeroTensor = builder.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get({2}, f64Type), ValueRange{zero, zero});

        // Convert mcm (I1) to an index and insert 1 into the zero tensor at the index
        Value mcm = mcmobs.getMcm();
        auto extuiOp = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), mcm);
        auto indexOp =
            builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), extuiOp.getOut());
        auto one =
            builder.create<arith::ConstantOp>(loc, f64Type, builder.getFloatAttr(f64Type, 1));
        auto insertedTensor = builder.create<tensor::InsertOp>(loc, one.getResult(), zeroTensor,
                                                               ValueRange{indexOp.getResult()});

        builder.create<func::ReturnOp>(loc, insertedTensor.getResult());

        mcmobs->erase();
    }

    void handleExpvalOrProbsOneShot(IRRewriter &builder, func::FuncOp qfunc, MeasurementProcess mp,
                                    Value shots, Operation *mod, bool isExpval)
    {
        // Add the result from each shot, and divide by the number of shots.
        // We simply perform the addition in the for loop body: no need to initialize a big
        // empty tensor to hold all the results.
        // This way we save some space, generate fewer ops, and also this method works for
        // both static and dynamic shots.

        OpBuilder::InsertionGuard guard(builder);
        Location loc = mod->getLoc();

        // If the MP is on a MCM, we need to massage the one-shot kernel a bit
        Operation *MPSourceOp = mp.getObs().getDefiningOp();
        if (isa<quantum::MCMObsOp>(MPSourceOp)) {
            if (isExpval) {
                editKernelMCMExpval(builder, qfunc, cast<quantum::MCMObsOp>(MPSourceOp));
            }
            else {
                editKernelProbsExpval(builder, qfunc, cast<quantum::MCMObsOp>(MPSourceOp));
            }
        }

        Type f64Type = builder.getF64Type();

        // Each expval result is a tensor<f64>
        // Each probs result is a tensor<blahxf64>
        auto expvalOrProbsType = dyn_cast<ShapedType>(qfunc.getResultTypes()[0]);

        func::FuncOp qkernel = createOneShotKernel(builder, qfunc, shots, mod);

        // Create the for loop.
        // Each loop iteration adds to the total expval or probs sum.
        builder.setInsertionPointToEnd(&qfunc.getBody().front());
        auto expvalOrProbsSum = builder.create<stablehlo::ConstantOp>(
            loc, expvalOrProbsType,
            DenseElementsAttr::get(expvalOrProbsType, builder.getFloatAttr(f64Type, 0)));
        scf::ForOp newForOp = createForLoop(builder, shots, ValueRange{expvalOrProbsSum});

        // Each loop iteration calls the one-shot kernel and adds to the sum
        // Since the kernel was a clone of the original qfunc, their arguments are the same!
        builder.setInsertionPointToEnd(newForOp.getBody());
        auto kernalCallOp = builder.create<func::CallOp>(
            loc, qkernel.getFunctionType().getResults(), qkernel.getSymName(),
            qfunc.getBody().front().getArguments());

        auto addOp = builder.create<stablehlo::AddOp>(loc, kernalCallOp.getResult(0),
                                                      newForOp.getRegionIterArg(0));

        builder.create<scf::YieldOp>(loc, addOp.getResult());

        // Divide and return
        builder.setInsertionPointToEnd(&qfunc.getBody().front());

        // shots Value is I64, need to turn into tensor<f64> then broadcast for division
        auto int2floatCastOp = builder.create<arith::SIToFPOp>(loc, f64Type, shots);
        auto shotsFromElementsOp = builder.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get({}, f64Type), int2floatCastOp.getResult());

        // TODO: no need to broadcast if it's expval
        auto broadcastedShots = builder.create<stablehlo::BroadcastInDimOp>(
            loc, expvalOrProbsType, shotsFromElementsOp.getResult(),
            builder.getDenseI64ArrayAttr({}));

        auto divOp = builder.create<stablehlo::DivOp>(loc, newForOp->getResult(0),
                                                      broadcastedShots->getResult(0));
        builder.create<func::ReturnOp>(loc, divOp.getResult());
    }

    SmallVector<int64_t> editKernelSampleShapes(func::FuncOp qkernel, ShapedType fullSampleType)
    {
        // Change the one-shot kernel's sample op, and its users, to return one-shot shaped results

        SmallVector<int64_t> oneShotSampleShape = {1, fullSampleType.getShape()[1]};
        auto oneShotSampleType =
            RankedTensorType::get(oneShotSampleShape, fullSampleType.getElementType());
        qkernel.setFunctionType(FunctionType::get(
            qkernel->getContext(), qkernel.getFunctionType().getInputs(), {oneShotSampleType}));

        // Start from the sample op, and visit down the operand chain until the return op
        auto sampleOp = *qkernel.getOps<quantum::SampleOp>().begin();
        SetVector<Operation *> sampleOpForwardSlice;
        getForwardSlice(sampleOp, &sampleOpForwardSlice);
        sampleOpForwardSlice.insert(sampleOp);

        // Safety check: since we edited the function type, we must also edit the return
        auto retOp = cast<func::ReturnOp>(qkernel.getBody().back().getTerminator());
        assert(sampleOpForwardSlice.contains(retOp) &&
               "Expected the quantum kernel to return the samples");

        for (Operation *op : sampleOpForwardSlice) {
            for (Value v : op->getOpResults()) {
                ShapedType oldType = dyn_cast<ShapedType>(v.getType());
                if (oldType.getShape() == fullSampleType.getShape()) {
                    SmallVector<int64_t> newShape = {1, oldType.getShape()[1]};
                    v.setType(RankedTensorType::get(newShape, oldType.getElementType()));
                }
            }
        }
        return oneShotSampleShape;
    }

    void handleSampleOneShot(IRRewriter &builder, func::FuncOp qfunc, Value shots, Operation *mod)
    {
        // Total sample is the sample of each shot concatenated together.
        // We initialize an empty tensor, then insert sample results of each shot
        // Sample's return shape is tensor<shots X num_qubits>

        OpBuilder::InsertionGuard guard(builder);
        Location loc = mod->getLoc();

        auto fullSampleType = dyn_cast<ShapedType>(qfunc.getResultTypes()[0]);
        assert(fullSampleType.getShape().size() == 2 &&
               "Expected sample result type to be a tensor of size shot X num_qubits");

        // Create one shot quantum kernel, and change the shot dimension in sample's return shape to
        // one
        func::FuncOp qkernel = createOneShotKernel(builder, qfunc, shots, mod);
        SmallVector<int64_t> oneShotSampleShape = editKernelSampleShapes(qkernel, fullSampleType);

        // Create the for loop.
        // Each loop iteration inserts its sample to the total sample result tensor.
        builder.setInsertionPointToEnd(&qfunc.getBody().front());
        auto fullSampleResults = builder.create<tensor::EmptyOp>(loc, fullSampleType.getShape(),
                                                                 fullSampleType.getElementType());
        scf::ForOp newForOp = createForLoop(builder, shots, ValueRange{fullSampleResults});

        // Each loop iteration calls the one-shot kernel and adds to the sum
        // Since the kernel was a clone of the original qfunc, their arguments are the same!
        builder.setInsertionPointToEnd(newForOp.getBody());
        auto kernalCallOp = builder.create<func::CallOp>(
            loc, qkernel.getFunctionType().getResults(), qkernel.getSymName(),
            qfunc.getBody().front().getArguments());

        auto zero = builder.getIndexAttr(0);
        auto one = builder.getIndexAttr(1);
        SmallVector<OpFoldResult> offsets = {newForOp.getInductionVar(), zero};
        SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(oneShotSampleShape[0]),
                                           builder.getIndexAttr(oneShotSampleShape[1])};
        SmallVector<OpFoldResult> strides = {one, one};

        auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
            loc, kernalCallOp.getResult(0), newForOp.getRegionIterArg(0), offsets, sizes, strides);

        builder.create<scf::YieldOp>(loc, insertSliceOp.getResult());

        // Return the full samples
        builder.setInsertionPointToEnd(&qfunc.getBody().front());
        builder.create<func::ReturnOp>(loc, newForOp.getResult(0));
    }

    void handleCountsOneShot(IRRewriter &builder, func::FuncOp qfunc, Value shots, Operation *mod)
    {
        // Add the counts from each shot
        // Very similar to expval and probs, but with two differences
        // First of all, counts kernel returns 2 result values instead of 1, so the loop yields
        // two results from the kernel call
        // Second of all, counts does not need a division at the end

        OpBuilder::InsertionGuard guard(builder);
        Location loc = mod->getLoc();

        auto eigensType = dyn_cast<ShapedType>(qfunc.getResultTypes()[0]);
        auto countsType = dyn_cast<ShapedType>(qfunc.getResultTypes()[1]);

        func::FuncOp qkernel = createOneShotKernel(builder, qfunc, shots, mod);

        // Create the for loop.
        // Each loop iteration adds to the total counts.
        // counts return type is <blah x i64>, for both eigens and counts
        builder.setInsertionPointToEnd(&qfunc.getBody().front());
        auto countsSum = builder.create<stablehlo::ConstantOp>(
            loc, countsType,
            DenseElementsAttr::get(countsType, builder.getIntegerAttr(builder.getI64Type(), 0)));

        // We also need to yield the eigens from the kernel call inside the loop body
        // However, this requires the loop to have an iteration argument for the eigens tensor
        // We just initialize an empty one.
        auto eigensPlaceholder = builder.create<tensor::EmptyOp>(loc, eigensType.getShape(),
                                                                 eigensType.getElementType());
        scf::ForOp newForOp =
            createForLoop(builder, shots, ValueRange{eigensPlaceholder, countsSum});

        // Each loop iteration calls the one-shot kernel and adds to the sum
        // Since the kernel was a clone of the original qfunc, their arguments are the same!
        builder.setInsertionPointToEnd(newForOp.getBody());
        auto kernalCallOp = builder.create<func::CallOp>(
            loc, qkernel.getFunctionType().getResults(), qkernel.getSymName(),
            qfunc.getBody().front().getArguments());

        auto addOp = builder.create<stablehlo::AddOp>(loc, kernalCallOp.getResult(1),
                                                      newForOp.getRegionIterArg(1));

        // Eigens are the same every loop iteration, so just yield it, no need to do anything
        builder.create<scf::YieldOp>(loc, ValueRange{kernalCallOp.getResult(0), addOp.getResult()});

        // Return the results
        builder.setInsertionPointToEnd(&qfunc.getBody().front());
        builder.create<func::ReturnOp>(loc, newForOp.getResults());
    }

    void runOnOperation() override
    {
        Operation *mod = getOperation();
        IRRewriter builder(mod->getContext());

        bool illegalMP = false, isExpval = false, isProbs = false, isSample = false,
             isCounts = false;
        func::FuncOp qfunc;
        MeasurementProcess mp;
        mod->walk([&](MeasurementProcess _mp) {
            if (isa<quantum::ExpvalOp>(_mp)) {
                isExpval = true;
                qfunc = _mp->getParentOfType<func::FuncOp>();
                mp = _mp;
            }
            else if (isa<quantum::ProbsOp>(_mp)) {
                isProbs = true;
                qfunc = _mp->getParentOfType<func::FuncOp>();
                mp = _mp;
            }
            else if (isa<quantum::SampleOp>(_mp)) {
                isSample = true;
                qfunc = _mp->getParentOfType<func::FuncOp>();
                mp = _mp;
            }
            else if (isa<quantum::CountsOp>(_mp)) {
                isCounts = true;
                qfunc = _mp->getParentOfType<func::FuncOp>();
                mp = _mp;
            }
            else if (isa<quantum::VarianceOp>(_mp)) {
                _mp->emitOpError(
                    "VarianceOp is not currently supported for conversion to one-shot MCM.");
                illegalMP = true;
            }
            else if (isa<quantum::StateOp>(_mp)) {
                _mp->emitOpError("StateOp is not compatible with shot-based execution, and has no "
                                 "valid conversion to one-shot MCM.");
                illegalMP = true;
            }
            return WalkResult::interrupt();
        });

        if (illegalMP) {
            return signalPassFailure();
        }

        size_t numReturns = isCounts ? 2 : 1;
        assert(qfunc.getResultTypes().size() == numReturns &&
               "Multiple terminal MPs not yet supported in one shot transform");

        // Get the shots SSA value. It is an operand to the device init op.
        auto deviceInitOp = *qfunc.getOps<quantum::DeviceInitOp>().begin();
        Value shots = deviceInitOp.getShots();

        if (isExpval || isProbs) {
            handleExpvalOrProbsOneShot(builder, qfunc, mp, shots, mod, isExpval);
        }
        else if (isSample) {
            handleSampleOneShot(builder, qfunc, shots, mod);
        }
        else if (isCounts) {
            handleCountsOneShot(builder, qfunc, shots, mod);
        }
    }
};

} // namespace quantum
} // namespace catalyst
