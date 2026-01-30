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

} // anonymous namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ONESHOTMCMPASS
#include "Quantum/Transforms/Passes.h.inc"

struct OneShotMCMPass : public impl::OneShotMCMPassBase<OneShotMCMPass> {
    using impl::OneShotMCMPassBase<OneShotMCMPass>::OneShotMCMPassBase;

    void handleExpvalOrProbsOneShot(IRRewriter &builder, func::FuncOp qfunc, Value shots,
                                    Operation *mod)
    {
        // Add the result from each shot, and divide by the number of shots.
        // We simply perform the addition in the for loop body: no need to initialize a big
        // empty tensor to hold all the results.
        // This way we save some space, generate fewer ops, and also this method works for
        // both static and dynamic shots.

        OpBuilder::InsertionGuard guard(builder);
        Location loc = mod->getLoc();

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

    void runOnOperation() override
    {
        Operation *mod = getOperation();
        IRRewriter builder(mod->getContext());

        bool illegalMP = false, isExpval = false,
             isProbs = false; //, isSample = false, isCounts = false;
        func::FuncOp qfunc;
        mod->walk([&](MeasurementProcess mp) {
            if (isa<quantum::ExpvalOp>(mp)) {
                isExpval = true;
                qfunc = mp->getParentOfType<func::FuncOp>();
            }
            else if (isa<quantum::ProbsOp>(mp)) {
                isProbs = true;
                qfunc = mp->getParentOfType<func::FuncOp>();
            }
            else if (isa<quantum::VarianceOp>(mp)) {
                mp->emitOpError(
                    "VarianceOp is not currently supported for conversion to one-shot MCM.");
                illegalMP = true;
            }
            else if (isa<quantum::StateOp>(mp)) {
                mp->emitOpError("StateOp is not compatible with shot-based execution, and has no "
                                "valid conversion to one-shot MCM.");
                illegalMP = true;
            }
            return WalkResult::interrupt();
        });

        if (illegalMP) {
            return signalPassFailure();
        }

        assert(qfunc.getResultTypes().size() == 1 &&
               "Multiple terminal MPs not yet supported in one shot transform");

        // Get the shots SSA value. It is an oeprand to the device init op.
        auto deviceInitOp = *qfunc.getOps<quantum::DeviceInitOp>().begin();
        Value shots = deviceInitOp.getShots();

        if (isExpval || isProbs) {
            handleExpvalOrProbsOneShot(builder, qfunc, shots, mod);
        }
    }
};

} // namespace quantum
} // namespace catalyst
