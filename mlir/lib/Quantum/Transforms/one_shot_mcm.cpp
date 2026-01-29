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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace {
void clearFuncExceptShots(func::FuncOp qfunc, Operation *shotsDefOp)
{
    // Delete the body of a funcop, except the operation that produces the shots value
    // We need triple loop to explicitly iterate in reverse order, due to erasure.
    SmallVector<Operation *> eraseWorklist;
    for (auto &region : qfunc->getRegions()) {
        for (auto &block : region.getBlocks()) {
            for (auto op = block.rbegin(); op != block.rend(); ++op) {
                if (&*op != shotsDefOp) {
                    eraseWorklist.push_back(&*op);
                }
            }
        }
    }

    for (Operation *op : eraseWorklist) {
        op->erase();
    }
}
} // anonymous namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ONESHOTMCMPASS
#include "Quantum/Transforms/Passes.h.inc"

struct OneShotMCMPass : public impl::OneShotMCMPassBase<OneShotMCMPass> {
    using impl::OneShotMCMPassBase<OneShotMCMPass>::OneShotMCMPassBase;

    void handleExpvalOneShot() {}

    void runOnOperation() override
    {
        Operation *mod = getOperation();
        IRRewriter builder(mod->getContext());
        Location loc = mod->getLoc();
        MLIRContext *ctx = &getContext();

        bool illegalMP = false,
             isExpval = false; //, isProbs = false, isSample = false, isCounts = false;
        func::FuncOp qfunc;
        mod->walk([&](MeasurementProcess mp) {
            if (isa<quantum::ExpvalOp>(mp)) {
                isExpval = true;
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

        // Get the shots SSA value. It is an oeprand to the device init op.
        auto deviceInitOp = *qfunc.getOps<quantum::DeviceInitOp>().begin();
        Value shots = deviceInitOp.getShots();
        Operation *shotsDefOp = shots.getDefiningOp();

        if (isExpval) {
            // Add the expval result from each shot, and divide by the number of shots.
            // We simply perform the addition in the for loop body: no need to initialize a big
            // empty tensor to hold all the results.
            // This way we save some space, generate fewer ops, and also this method works for
            // both static and dynamic shots.

            Type i64Type = builder.getI64Type();
            Type f64Type = builder.getF64Type();
            Type indexType = builder.getIndexType();

            // 1. Clone the quantum kernel and give it a new name
            // Because we need to make sure the current qfunc name is reserved as entry point
            // Set the number of shots in the new kernel to one.
            builder.setInsertionPointToStart(&mod->getRegion(0).front());
            auto qkernel = cast<func::FuncOp>(qfunc->clone());
            qkernel.setSymNameAttr(StringAttr::get(ctx, qfunc.getSymName() + ".quantum_kernel"));
            builder.insert(qkernel);

            builder.setInsertionPointToStart(&qkernel.getBody().front());
            auto kernelDeviceInitOp = *qkernel.getOps<quantum::DeviceInitOp>().begin();
            auto one =
                builder.create<arith::ConstantOp>(loc, i64Type, builder.getIntegerAttr(i64Type, 1));
            Value originalKernelShots = kernelDeviceInitOp.getShots();
            kernelDeviceInitOp->setOperand(0, one);
            if (originalKernelShots.getNumUses() == 0) {
                originalKernelShots.getDefiningOp()->erase();
            }

            // 2. Clear the qfunc. Its new contents will be the one-shot logic.
            // Keep the SSA value for the shots. It needs to be used as the upper bound of the for
            // loop.
            clearFuncExceptShots(qfunc, shotsDefOp);
            builder.setInsertionPointToEnd(&qfunc.getBody().front());

            // 3. Create the for loop.
            // Each loop iteration adds to the total expval sum.
            auto expvalSum =
                builder.create<arith::ConstantOp>(loc, f64Type, builder.getFloatAttr(f64Type, 0));

            auto lb = builder.create<arith::ConstantOp>(loc, indexType,
                                                        builder.getIntegerAttr(indexType, 0));
            auto step = builder.create<arith::ConstantOp>(loc, indexType,
                                                          builder.getIntegerAttr(indexType, 1));
            auto ub = builder.create<index::CastSOp>(loc, builder.getIndexType(), shots);
            auto newForOp = builder.create<scf::ForOp>(loc, lb, ub, step, ValueRange{expvalSum});

            builder.setInsertionPointToEnd(newForOp.getBody());

            auto kernalCallOp = builder.create<func::CallOp>(
                loc, qkernel.getFunctionType().getResults(), qkernel.getSymName(),
                qfunc.getBody().front().getArguments());

            // Each expval result is a tensor<f64>
            auto extractOp =
                builder.create<tensor::ExtractOp>(loc, kernalCallOp.getResults()[0], ValueRange{});

            auto addOp =
                builder.create<arith::AddFOp>(loc, extractOp, newForOp.getRegionIterArg(0));

            builder.create<scf::YieldOp>(loc, addOp.getResult());

            // 4. Divide and return
            builder.setInsertionPointToEnd(&qfunc.getBody().front());
            auto int2floatCastOp = builder.create<arith::SIToFPOp>(loc, f64Type, shots);
            auto divOp = builder.create<arith::DivFOp>(loc, newForOp->getResult(0),
                                                       int2floatCastOp.getResult());
            auto fromElementsOp = builder.create<tensor::FromElementsOp>(
                loc, RankedTensorType::get({}, f64Type), divOp.getResult());
            builder.create<func::ReturnOp>(loc, fromElementsOp.getResult());

            handleExpvalOneShot();
        } // expval
    }
};

} // namespace quantum
} // namespace catalyst
