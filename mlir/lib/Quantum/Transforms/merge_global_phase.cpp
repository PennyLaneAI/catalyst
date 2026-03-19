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

#include <iterator>

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
        OpBuilder builder(mod->getContext());

        mod.walk([&](Operation *op) {
            for (Region &reg : op->getRegions()) {
                for (Block &block : reg.getBlocks()) {
                    auto phases = block.getOps<GlobalPhaseOp>();
                    auto simplePhases = llvm::make_filter_range(phases, [](GlobalPhaseOp phaseOp) {
                        return phaseOp.getInCtrlQubits().empty();
                    });
                    if (simplePhases.empty() ||
                        std::next(simplePhases.begin()) == simplePhases.end()) {
                        continue;
                    }

                    GlobalPhaseOp firstPhase = *simplePhases.begin();
                    auto remainingPhases = llvm::drop_begin(simplePhases);

                    builder.setInsertionPoint(firstPhase);
                    Value runningSum = firstPhase.getAngle();
                    for (GlobalPhaseOp phaseOp : llvm::make_early_inc_range(remainingPhases)) {
                        llvm::SmallVector<Value, 2> args{runningSum, phaseOp.getAngle()};
                        runningSum = arith::AddFOp::create(builder, phaseOp.getLoc(), args);
                        phaseOp->erase();
                    }
                    firstPhase.getAngleMutable().assign(runningSum);
                }
            }
        });
    }
};

} // namespace quantum
} // namespace catalyst
