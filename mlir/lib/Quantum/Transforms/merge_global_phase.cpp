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

#define DEBUG_TYPE "merge-global-phase"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_MERGEGLOBALPHASEPASS
#define GEN_PASS_DEF_MERGEGLOBALPHASEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct MergeGlobalPhasePass : impl::MergeGlobalPhasePassBase<MergeGlobalPhasePass> {
    using impl::MergeGlobalPhasePassBase<MergeGlobalPhasePass>::MergeGlobalPhasePassBase;

    void runOnOperation() final
    {
        ModuleOp mod = getOperation();
        MLIRContext *ctx = mod->getContext();
        OpBuilder builder(mod->getContext());

        mod.walk([&](Operation *op) {
            for (Region &reg : op->getRegions()) {
                for (Block &block : reg.getBlocks()) {
                    auto phases = block.getOps<GlobalPhaseOp>();
                    auto simplePhases = llvm::make_filter_range(phases, [](GlobalPhaseOp phaseOp) {
                        return phaseOp.getInCtrlQubits().empty();
                    });
                    if (simplePhases.empty()) {
                        continue;
                    }

                    GlobalPhaseOp firstPhase = *simplePhases.begin();
                    builder.setInsertionPoint(firstPhase);

                    Value runningSum = arith::ConstantFloatOp::create(
                        builder, firstPhase->getLoc(), Float64Type::get(ctx), llvm::APFloat(0.0));
                    for (GlobalPhaseOp phaseOp : simplePhases) {
                        llvm::SmallVector<Value, 2> args{runningSum, phaseOp.getParams()};
                        runningSum = arith::AddFOp::create(builder, phaseOp.getLoc(), args);
                    }
                    GlobalPhaseOp::create(builder, firstPhase.getLoc(), {}, runningSum, false, {},
                                          {});

                    for (GlobalPhaseOp phaseOp : llvm::make_early_inc_range(simplePhases)) {
                        phaseOp->erase();
                    }
                }
            }
        });
    }
};

} // namespace quantum
} // namespace catalyst
