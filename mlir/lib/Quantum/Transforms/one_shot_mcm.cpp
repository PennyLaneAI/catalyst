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

#include <optional>

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
#include "llvm/ADT/SmallSet.h"

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

void eraseAllUsersExcept(Value v, Operation *exception)
{
    // Erase all users of a Value, and all users of these users, etc., in its forward slice,
    // except the marked exception.

    // We need to use a SetVector, since during the erasure we have to delete from bottom up
    // So we need an ordered container
    SetVector<Operation *> forwardSlice;
    getForwardSlice(v, &forwardSlice);
    forwardSlice.remove(exception);

    for (auto op = forwardSlice.rbegin(); op != forwardSlice.rend(); ++op) {
        (*op)->erase();
    }
}

} // anonymous namespace

namespace catalyst {
namespace quantum {

std::optional<MeasurementProcess> getMPFromValue(Value v)
{
    // Get the MP operation that produces a Value v somewhere in the MP's forward slice.
    // In other words, get the MP operations that a Value v comes from.
    // Because an MP and the corresponding returned Value is usually not super far away, doing
    // an entire backslice would not be ideal, since that could possibly backslice to the very
    // beginning of a function.
    // So we just depth-first search from the Value directly.

    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        // Reached a block argument
        return std::nullopt;
    }

    if (isa<MeasurementProcess>(defOp)) {
        return cast<MeasurementProcess>(defOp);
    }

    for (Value operand : defOp->getOperands()) {
        std::optional<MeasurementProcess> candidate = getMPFromValue(operand);
        if (candidate.has_value()) {
            return candidate;
        }
    }

    return std::nullopt;
}

#define GEN_PASS_DEF_ONESHOTMCMPASS
#include "Quantum/Transforms/Passes.h.inc"

struct OneShotMCMPass : public impl::OneShotMCMPassBase<OneShotMCMPass> {
    using impl::OneShotMCMPassBase<OneShotMCMPass>::OneShotMCMPassBase;

    void editKernelMCMExpval(IRRewriter &builder, func::FuncOp oneShotKernel,
                             quantum::MCMObsOp mcmobs, size_t retIdx)
    {
        // If the kernel returns expval on a mcm,
        // the single-shot expval of a mcm is just the mcm boolean result itself
        // So we just cast it to the correct type and return it.

        OpBuilder::InsertionGuard guard(builder);
        Location loc = oneShotKernel->getLoc();

        // Cast the I1 mcm to the correct type and return it
        // MCM itself is I1
        // Expval kernel returns tensor<f64>
        // We need to cast as I1 -> I64 -> F64 -> tensor<f64>
        Operation *retOp = oneShotKernel.getBody().back().getTerminator();
        builder.setInsertionPoint(retOp);

        Value mcm = mcmobs.getMcm();
        auto extuiOp = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), mcm);
        auto int2floatCastOp =
            builder.create<arith::SIToFPOp>(loc, builder.getF64Type(), extuiOp.getOut());
        auto fromElementsOp = builder.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get({}, builder.getF64Type()), int2floatCastOp.getResult());

        // Return the new mcm expval
        retOp->setOperand(retIdx, fromElementsOp.getResult());

        // Erase all users of the mcm obs: the new return does not need them.
        eraseAllUsersExcept(mcmobs.getObs(), retOp);
        mcmobs->erase();
    }

    void editKernelMCMProbs(IRRewriter &builder, func::FuncOp oneShotKernel,
                            quantum::MCMObsOp mcmobs, size_t retIdx)
    {
        // If the kernel returns probs on a mcm,
        // the single-shot probs of the mcm is either [1,0] or [0,1]

        OpBuilder::InsertionGuard guard(builder);
        Location loc = oneShotKernel->getLoc();
        Type f64Type = builder.getF64Type();

        // Create the probs tensor and write the 1 into it
        // The location of the 1 is exactly the mcm result
        // i.e. mcm 0, probs is [1,0]
        // mcm 1, probs is [0,1]
        Operation *retOp = oneShotKernel.getBody().back().getTerminator();
        builder.setInsertionPoint(retOp);

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

        retOp->setOperand(retIdx, insertedTensor.getResult());

        // Erase all users of the mcm obs: they new return does not need them.
        eraseAllUsersExcept(mcmobs.getObs(), retOp);
        mcmobs->erase();
    }

    void editKernelMCMSample(IRRewriter &builder, func::FuncOp oneShotKernel,
                             quantum::MCMObsOp mcmobs, size_t retIdx)
    {
        // If the kernel returns sample on a mcm,
        // the single-shot sample of a mcm is just the mcm boolean result itself
        // So we just cast it to the correct type and return it.

        OpBuilder::InsertionGuard guard(builder);
        Location loc = oneShotKernel->getLoc();

        // Cast the I1 mcm to the correct type and return it
        // MCM itself is I1
        // Sample kernel returns tensor<1x1xi64>
        Operation *retOp = oneShotKernel.getBody().back().getTerminator();
        builder.setInsertionPoint(retOp);

        Value mcm = mcmobs.getMcm();
        auto extuiOp = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), mcm);
        auto fromElementsOp = builder.create<tensor::FromElementsOp>(
            loc, oneShotKernel.getFunctionType().getResults()[retIdx], extuiOp.getOut());

        retOp->setOperand(retIdx, fromElementsOp.getResult());

        // Erase all users of the mcm obs: the new return does not need them.
        eraseAllUsersExcept(mcmobs.getObs(), retOp);
        mcmobs->erase();
    }

    SmallVector<int64_t> editKernelSampleShapes(func::FuncOp oneShotKernel,
                                                ShapedType fullSampleType,
                                                quantum::SampleOp sampleOp, size_t retIdx)
    {
        // Change the one-shot kernel's sample op, and its users, to return one-shot shaped results

        SmallVector<int64_t> oneShotSampleShape = {1, fullSampleType.getShape()[1]};
        auto oneShotSampleType =
            RankedTensorType::get(oneShotSampleShape, fullSampleType.getElementType());

        auto oldFunctionType = oneShotKernel.getFunctionType();
        SmallVector<Type> oneShotKernelOutTypes(
            oldFunctionType.getResults().begin(),
            oldFunctionType.getResults().end()); // getFunctionType() returns ArrayRef
        oneShotKernelOutTypes[retIdx] = oneShotSampleType;
        oneShotKernel.setFunctionType(FunctionType::get(
            oneShotKernel->getContext(), oldFunctionType.getInputs(), oneShotKernelOutTypes));

        // Start from the sample op, and visit down the operand chain until the return op
        SetVector<Operation *> sampleOpForwardSlice;
        getForwardSlice(sampleOp, &sampleOpForwardSlice);
        sampleOpForwardSlice.insert(sampleOp);

        // Safety check: since we edited the function type, we must also edit the return
        auto retOp = cast<func::ReturnOp>(oneShotKernel.getBody().back().getTerminator());
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

    void runOnOperation() override
    {
        Operation *mod = getOperation();
        Location loc = mod->getLoc();
        IRRewriter builder(mod->getContext());

        Type f64Type = builder.getF64Type();

        // Collect all qnode functions.
        // We find qnode functions by identifying the parent function ops of MPs
        SetVector<func::FuncOp> qnodeFuncs;
        mod->walk([&](MeasurementProcess _mp) {
            qnodeFuncs.insert(_mp->getParentOfType<func::FuncOp>());
        });

        // For each qnode function, find the returned MPs
        // Then handle the one shot logic for each MP type
        for (auto qnodeFunc : qnodeFuncs) {
            // Get the shots SSA value. It is an operand to the device init op.
            auto deviceInitOp = *qnodeFunc.getOps<quantum::DeviceInitOp>().begin();
            Value shots = deviceInitOp.getShots();

            // Clone the qnode function and give it a new name.
            // Because we need to make sure the current qfunc name is reserved as entry point
            // Set the number of shots in the new kernel to one.
            // Also clear the original qnode function. Its new contents will be the one-shot logic.
            // Keep the SSA value for the shots. It needs to be used as the upper bound of the for
            // loop.
            func::FuncOp oneShotKernel = createOneShotKernel(builder, qnodeFunc, shots, mod);

            // Create the for loop
            // Depending on the MP, the for loop might need some iteration arguments
            Operation *retOp = oneShotKernel.getBody().back().getTerminator();
            assert(isa<func::ReturnOp>(retOp) && "Expected a qnode function to return values");

            SmallVector<MeasurementProcess> qnodeMPs;
            for (Value returnValue : retOp->getOperands()) {
                qnodeMPs.push_back(*getMPFromValue(returnValue));
            }
            llvm::SmallSet<quantum::CountsOp, 8> handledCountsOps;

            builder.setInsertionPointToEnd(&qnodeFunc.getBody().front());
            SmallVector<Value> loopIterArgs;
            SmallVector<std::string> loopIterArgsMPKinds;

            SmallVector<int64_t> oneShotSampleShape;

            for (auto [i, _mp] : llvm::enumerate(qnodeMPs)) {
                // llvm::enumerate() marks the iterators with const
                auto mp = const_cast<catalyst::quantum::MeasurementProcess &>(_mp);

                if (isa<quantum::StateOp>(mp)) {
                    mp->emitOpError(
                        "StateOp is not compatible with shot-based execution, and has no "
                        "valid conversion to one-shot MCM.");
                    return signalPassFailure();
                }

                else if (isa<quantum::VarianceOp>(mp)) {
                    mp->emitOpError(
                        "VarianceOp is not currently supported for conversion to one-shot MCM.");
                    return signalPassFailure();
                }

                else if (isa<quantum::ExpvalOp>(mp)) {

                    // The one shot kernel itself might need some massaging if the MP is on a MCM.
                    Operation *MPSourceOp = mp.getObs().getDefiningOp();
                    if (isa<quantum::MCMObsOp>(MPSourceOp)) {
                        editKernelMCMExpval(builder, oneShotKernel,
                                            cast<quantum::MCMObsOp>(MPSourceOp), i);
                    }

                    // Each expval result is a tensor<f64>
                    auto expvalType = dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i]);
                    auto expvalSum = builder.create<stablehlo::ConstantOp>(
                        loc, expvalType,
                        DenseElementsAttr::get(expvalType, builder.getFloatAttr(f64Type, 0)));
                    loopIterArgs.push_back(expvalSum);
                    loopIterArgsMPKinds.push_back("expval");
                }

                else if (isa<quantum::ProbsOp>(mp)) {

                    // The one shot kernel itself might need some massaging if the MP is on a MCM.
                    Operation *MPSourceOp = mp.getObs().getDefiningOp();
                    if (isa<quantum::MCMObsOp>(MPSourceOp)) {
                        editKernelMCMProbs(builder, oneShotKernel,
                                           cast<quantum::MCMObsOp>(MPSourceOp), i);
                    }

                    // Each probs result is a tensor<blahxf64>
                    auto probsType = dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i]);
                    auto probsSum = builder.create<stablehlo::ConstantOp>(
                        loc, probsType,
                        DenseElementsAttr::get(probsType, builder.getFloatAttr(f64Type, 0)));
                    loopIterArgs.push_back(probsSum);
                    loopIterArgsMPKinds.push_back("probs");
                }

                else if (isa<quantum::SampleOp>(mp)) {

                    auto fullSampleType = dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i]);
                    assert(fullSampleType.getShape().size() == 2 &&
                           "Expected sample result type to be a tensor of size shot X num_qubits");

                    oneShotSampleShape = editKernelSampleShapes(oneShotKernel, fullSampleType,
                                                                cast<quantum::SampleOp>(mp), i);

                    // If the sample MP is on a MCM, we need to massage the one-shot kernel a bit
                    Operation *MPSourceOp = mp.getObs().getDefiningOp();
                    if (isa<quantum::MCMObsOp>(MPSourceOp)) {
                        editKernelMCMSample(builder, oneShotKernel,
                                            cast<quantum::MCMObsOp>(MPSourceOp), i);
                    }
                    auto fullSampleResults = builder.create<tensor::EmptyOp>(
                        loc, fullSampleType.getShape(), fullSampleType.getElementType());

                    loopIterArgs.push_back(fullSampleResults);
                    loopIterArgsMPKinds.push_back("sample");
                }

                else if (isa<quantum::CountsOp>(mp)) {
                    if (!handledCountsOps.contains(cast<quantum::CountsOp>(mp))) {
                        auto eigensType = dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i]);
                        auto countsType =
                            dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i + 1]);

                        auto countsSum = builder.create<stablehlo::ConstantOp>(
                            loc, countsType,
                            DenseElementsAttr::get(
                                countsType, builder.getIntegerAttr(builder.getI64Type(), 0)));

                        // We also need to yield the eigens from the kernel call inside the loop
                        // body However, this requires the loop to have an iteration argument for
                        // the eigens tensor We just initialize an empty one.
                        auto eigensPlaceholder = builder.create<tensor::EmptyOp>(
                            loc, eigensType.getShape(), eigensType.getElementType());

                        loopIterArgs.push_back(eigensPlaceholder);
                        loopIterArgs.push_back(countsSum);
                        loopIterArgsMPKinds.push_back("eigens");
                        loopIterArgsMPKinds.push_back("counts");
                        handledCountsOps.insert(cast<quantum::CountsOp>(mp));
                    }
                }
            }
            scf::ForOp forOp = createForLoop(builder, shots, loopIterArgs);

            // Each loop iteration calls the one-shot kernel
            // Since the kernel was a clone of the original qfunc, their arguments are the same!
            builder.setInsertionPointToEnd(forOp.getBody());
            auto kernalCallOp = builder.create<func::CallOp>(
                loc, oneShotKernel.getFunctionType().getResults(), oneShotKernel.getSymName(),
                qnodeFunc.getBody().front().getArguments());

            // Perform each MP's necessary handling inside the loop body
            SmallVector<Value> loopYields;
            for (auto [i, mpKind] : llvm::enumerate(loopIterArgsMPKinds)) {

                if (mpKind == "expval") {
                    // Add the expval from each iteration
                    auto addOp = builder.create<stablehlo::AddOp>(loc, kernalCallOp.getResult(i),
                                                                  forOp.getRegionIterArg(i));
                    loopYields.push_back(addOp.getResult());
                }

                else if (mpKind == "probs") {
                    // Add the probs from each iteration
                    auto addOp = builder.create<stablehlo::AddOp>(loc, kernalCallOp.getResult(i),
                                                                  forOp.getRegionIterArg(i));
                    loopYields.push_back(addOp.getResult());
                }

                else if (mpKind == "sample") {
                    // Insert the one shot kernel's sample into the full sample result tensor
                    auto zero = builder.getIndexAttr(0);
                    auto one = builder.getIndexAttr(1);
                    SmallVector<OpFoldResult> offsets = {forOp.getInductionVar(), zero};
                    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(oneShotSampleShape[0]),
                                                       builder.getIndexAttr(oneShotSampleShape[1])};
                    SmallVector<OpFoldResult> strides = {one, one};

                    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
                        loc, kernalCallOp.getResult(i), forOp.getRegionIterArg(i), offsets, sizes,
                        strides);

                    loopYields.push_back(insertSliceOp.getResult());
                }

                else if (mpKind == "eigens") {
                    loopYields.push_back(kernalCallOp.getResult(i));
                }

                else if (mpKind == "counts") {
                    auto addOp = builder.create<stablehlo::AddOp>(loc, kernalCallOp.getResult(i),
                                                                  forOp.getRegionIterArg(i));
                    loopYields.push_back(addOp.getResult());
                }
            }

            builder.create<scf::YieldOp>(loc, loopYields);

            // Perform each MP's necessary handling outside the loop body
            builder.setInsertionPointToEnd(&qnodeFunc.getBody().front());
            SmallVector<Value> retVals;
            for (auto [i, mpKind] : llvm::enumerate(loopIterArgsMPKinds)) {

                if (mpKind == "expval") {
                    // Divide the sum by shots
                    // shots Value is I64, need to turn into tensor<f64> for division
                    auto int2floatCastOp = builder.create<arith::SIToFPOp>(loc, f64Type, shots);
                    auto shotsFromElementsOp = builder.create<tensor::FromElementsOp>(
                        loc, RankedTensorType::get({}, f64Type), int2floatCastOp.getResult());

                    auto divOp = builder.create<stablehlo::DivOp>(loc, forOp->getResult(i),
                                                                  shotsFromElementsOp.getResult());
                    retVals.push_back(divOp.getResult());
                }

                else if (mpKind == "probs") {
                    // Divide the sum by shots
                    // shots Value is I64, need to turn into tensor<f64> and then broadcast for
                    // division
                    auto int2floatCastOp = builder.create<arith::SIToFPOp>(loc, f64Type, shots);
                    auto shotsFromElementsOp = builder.create<tensor::FromElementsOp>(
                        loc, RankedTensorType::get({}, f64Type), int2floatCastOp.getResult());

                    auto probsType = dyn_cast<ShapedType>(oneShotKernel.getResultTypes()[i]);
                    auto broadcastedShots = builder.create<stablehlo::BroadcastInDimOp>(
                        loc, probsType, shotsFromElementsOp.getResult(),
                        builder.getDenseI64ArrayAttr({}));

                    auto divOp = builder.create<stablehlo::DivOp>(loc, forOp->getResult(i),
                                                                  broadcastedShots.getResult());
                    retVals.push_back(divOp.getResult());
                }

                else if (mpKind == "sample") {
                    retVals.push_back(forOp->getResult(i));
                }
                else if (mpKind == "eigens") {
                    retVals.push_back(forOp->getResult(i));
                }
                else if (mpKind == "counts") {
                    retVals.push_back(forOp->getResult(i));
                }
            }

            builder.create<func::ReturnOp>(loc, retVals);
        }
    } // runOnOperation()
};

} // namespace quantum
} // namespace catalyst
