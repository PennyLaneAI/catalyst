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

#define DEBUG_TYPE "combine-global-phases"

#include <iterator>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_COMBINEGLOBALPHASESPASS
#define GEN_PASS_DEF_COMBINEGLOBALPHASESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CombineGlobalPhasesPass : impl::CombineGlobalPhasesPassBase<CombineGlobalPhasesPass> {
    using impl::CombineGlobalPhasesPassBase<CombineGlobalPhasesPass>::CombineGlobalPhasesPassBase;

    void runOnOperation() final {
        ModuleOp mod = getOperation();
        OpBuilder builder(mod->getContext());

        mod.walk([&](Block *block) {
            auto phases = block->getOps<GlobalPhaseOp>();
            auto simplePhases = llvm::make_filter_range(
                phases, [](GlobalPhaseOp phaseOp) { return phaseOp.getInCtrlQubits().empty(); });
            if (simplePhases.empty() || std::next(simplePhases.begin()) == simplePhases.end()) {
                return WalkResult::advance();
            }

            GlobalPhaseOp lastPhase = *std::prev(simplePhases.end());
            auto remainingPhases = llvm::drop_end(simplePhases);

            builder.setInsertionPoint(lastPhase);
            Value runningSum = lastPhase.getAngle();
            if (lastPhase.getAdjoint()) {
                runningSum = arith::NegFOp::create(builder, lastPhase->getLoc(), runningSum);
                lastPhase.setAdjoint(false);
            }

            for (GlobalPhaseOp phaseOp : llvm::make_early_inc_range(remainingPhases)) {
                llvm::SmallVector<Value, 2> args{runningSum, phaseOp.getAngle()};
                if (phaseOp.getAdjoint()) {
                    runningSum = arith::SubFOp::create(builder, phaseOp.getLoc(), args);
                } else {
                    runningSum = arith::AddFOp::create(builder, phaseOp.getLoc(), args);
                }
                phaseOp->erase();
            }
            lastPhase.getAngleMutable().assign(runningSum);

            return WalkResult::advance();
        });
    }
};

} // namespace quantum
} // namespace catalyst
