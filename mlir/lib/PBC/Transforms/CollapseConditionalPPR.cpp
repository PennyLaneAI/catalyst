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

#define DEBUG_TYPE "pbc-ref-collapse-if-ppr"

#include <cstdlib>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCOps.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_COLLAPSECONDITIONALPPRPASS
#define GEN_PASS_DEF_COLLAPSECONDITIONALPPRPASS
#include "PBC/Transforms/Passes.h.inc"

namespace {

// Ops that can leave a branch: pi/2 PPRs become conditional, global phases are dropped, and ops
// without memory effects are hoisted.
bool opIsLegalToCollapse(Operation &op) {
    if (auto ppr = dyn_cast<RefPPRotationOp>(op)) {
        return std::abs(ppr.getRotationKind()) == 2;
    }
    if (auto phase = dyn_cast<qref::GlobalPhaseOp>(op)) {
        return phase.getCtrlQubits().empty();
    }
    if (auto get = dyn_cast<qref::GetOp>(op)) {
        return true;
    }
    return op.getNumRegions() == 0 && isPure(&op);
}

bool blockIsLegalToCollapse(Block *block) {
    return !block || llvm::all_of(block->without_terminator(), opIsLegalToCollapse);
}

// Moves the ops of `block` before `ifOp`, with the condition of `ifOp` (negated if `negated`) as
// the condition of its PPRs.
void hoistBranch(OpBuilder &builder, Block &block, scf::IfOp ifOp, bool negated) {
    Value guard;
    for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
        if (isa<qref::GlobalPhaseOp>(op)) {
            op.erase();
            continue;
        }

        op.moveBefore(ifOp);
        auto ppr = dyn_cast<RefPPRotationOp>(op);
        if (!ppr) {
            continue;
        }
        builder.setInsertionPoint(ppr);
        if (!guard) {
            guard = ifOp.getCondition();
            if (negated) {
                Location loc = ifOp.getLoc();
                Value one = arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
                guard = arith::XOrIOp::create(builder, loc, guard, one);
            }
        }
        Value condition = guard;
        if (Value existing = ppr.getCondition()) {
            condition = arith::AndIOp::create(builder, ppr.getLoc(), existing, guard);
        }
        ppr.getConditionMutable().assign(condition);
    }
}

bool collapseIfOp(scf::IfOp ifOp) {
    if (ifOp.getNumResults() != 0 || !blockIsLegalToCollapse(ifOp.thenBlock()) ||
        !blockIsLegalToCollapse(ifOp.elseBlock())) {
        return false;
    }

    OpBuilder builder(ifOp);
    hoistBranch(builder, *ifOp.thenBlock(), ifOp, /*negated=*/false);
    if (Block *elseBlock = ifOp.elseBlock()) {
        hoistBranch(builder, *elseBlock, ifOp, /*negated=*/true);
    }
    ifOp.erase();
    return true;
}

} // namespace

struct CollapseConditionalPPRPass
    : impl::CollapseConditionalPPRPassBase<CollapseConditionalPPRPass> {
    using impl::CollapseConditionalPPRPassBase<
        CollapseConditionalPPRPass>::CollapseConditionalPPRPassBase;

    void runOnOperation() final {
        // Post-order, so that nested branches collapse from the inside out.
        getOperation()->walk<WalkOrder::PostOrder>([&](scf::IfOp ifOp) { collapseIfOp(ifOp); });
    }
};

} // namespace pbc
} // namespace catalyst
