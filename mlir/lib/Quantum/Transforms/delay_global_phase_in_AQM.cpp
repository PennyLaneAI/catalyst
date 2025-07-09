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

#define DEBUG_TYPE "delay-global-phase-in-AQM"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_DELAYGLOBALPHASEINAQMPASS
#define GEN_PASS_DECL_DELAYGLOBALPHASEINAQMPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DelayGlobalPhaseInAQMPass : impl::DelayGlobalPhaseInAQMPassBase<DelayGlobalPhaseInAQMPass> {
    using DelayGlobalPhaseInAQMPassBase::DelayGlobalPhaseInAQMPassBase;

    Operation *findLastGateOp(Operation *module)
    {
        // Find the last gate op that's not a globalphase op
        Operation *finalGate = nullptr;
        module->walk([&](Operation *op) {
            if (isa<quantum::CustomOp>(op) || isa<quantum::SetStateOp>(op) ||
                isa<quantum::SetBasisStateOp>(op) || isa<quantum::MultiRZOp>(op) ||
                isa<quantum::QubitUnitaryOp>(op)) {
                finalGate = op;
                return WalkResult::advance();
            }
            else {
                return WalkResult::skip();
            }
        });
        return finalGate;
    }

    void runOnOperation() final
    {
        Operation *module = getOperation();
        mlir::IRRewriter builder(module->getContext());
        mlir::Location loc = module->getLoc();

        // Get the device
        // Is there an easy way other than just walking?
        quantum::DeviceInitOp deviceInitOp;
        size_t _devCount = 0;
        module->walk([&](quantum::DeviceInitOp dev) {
            deviceInitOp = dev;
            _devCount++;
        });
        if (_devCount == 0) {
            // Not a qnode, nothing to do
            return;
        }
        assert(_devCount == 1 && "A qfunc must have exactly one device init op.");

        if (!deviceInitOp.getAutoQubitManagement()) {
            // Not in AQM mode, nothing to do
            return;
        }

        // Find the last gate op that's not a globalphase op
        Operation *finalGate = findLastGateOp(module);
        builder.setInsertionPointAfter(finalGate);

        // Clump all global phases and create one after the last gate op
        SmallVector<Value> phases;
        SmallVector<quantum::GlobalPhaseOp> gphaseRemovalWorklist;
        module->walk([&](quantum::GlobalPhaseOp gphaseOp) {
            assert(gphaseOp.getInCtrlQubits().size() == 0 &&
                   "Gloabl phase ops with control qubits is not yet supported with automatic qubit "
                   "management mode.");

            if (gphaseOp.getAdjoint()) {
                phases.push_back(
                    builder.create<arith::NegFOp>(loc, gphaseOp.getParams()).getResult());
            }
            else {
                phases.push_back(gphaseOp.getParams());
            }
            gphaseRemovalWorklist.push_back(gphaseOp);
        });

        if (phases.size() == 0) {
            // No gphase ops, nothing to do
            return;
        }

        Value accumedPhase = phases[0];
        for (size_t i = 1; i < phases.size(); i++) {
            accumedPhase = builder.create<arith::AddFOp>(loc, accumedPhase, phases[i]).getResult();
        }
        builder.create<quantum::GlobalPhaseOp>(loc, TypeRange{}, accumedPhase, false, ValueRange{},
                                               ValueRange{});

        for (auto op : gphaseRemovalWorklist) {
            op->erase();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createDelayGlobalPhaseInAQMPass()
{
    return std::make_unique<quantum::DelayGlobalPhaseInAQMPass>();
}

} // namespace catalyst
