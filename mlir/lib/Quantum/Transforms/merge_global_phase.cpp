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

#include <algorithm>
#include <iterator>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceEffect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

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

/// The pass merges the uncontrolled global phases of each block. Loop bodies and callees are
/// functions of their own, and the phases of branches are assumed to be merged with the others.
static void combineGlobalPhaseResources(ResourceResult &function, const Pass &) {
    double numPhases = 0;
    for (StringRef name : {"GlobalPhase", "Adjoint(GlobalPhase)"}) {
        auto phases = function.operations.find(name);
        if (phases == function.operations.end()) {
            continue;
        }
        for (const auto &[size, count] : phases->second) {
            numPhases += count;
        }
        function.operations.erase(phases);
    }
    if (numPhases > 0) {
        function.operations["GlobalPhase"][{0, 1}] = std::min(numPhases, 1.0);
    }
}

void registerCombineGlobalPhasesResourceEffect() {
    [[maybe_unused]] static const bool once = [] {
        ResourceEffectRegistry::get().add("combine-global-phases",
                                          makeTransferEffect(combineGlobalPhaseResources));
        return true;
    }();
}

} // namespace quantum
} // namespace catalyst
