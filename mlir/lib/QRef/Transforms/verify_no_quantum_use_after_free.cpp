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

#define DEBUG_TYPE "verify-no-quantum-use-after-free"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst;

namespace {
bool hasUseAfterFree(Value qubit, Operation *gate, DominanceInfo &domInfo) {
    if (auto getOp = qubit.getDefiningOp<qref::GetOp>()) {
        Value qreg = getOp.getQreg();
        for (Operation *user : qreg.getUsers()) {
            if (auto deallocOp = dyn_cast<qref::DeallocOp>(user)) {
                if (domInfo.properlyDominates(deallocOp, gate)) {
                    return true;
                }
            }
        }
    } else {
        for (Operation *user : qubit.getUsers()) {
            if (auto deallocQubitOp = dyn_cast<qref::DeallocQubitOp>(user)) {
                if (domInfo.properlyDominates(deallocQubitOp, gate)) {
                    return true;
                }
            }
        }
    }
    return false;
}
} // namespace

namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_VERIFYNOQUANTUMUSEAFTERFREEPASS
#define GEN_PASS_DEF_VERIFYNOQUANTUMUSEAFTERFREEPASS
#include "QRef/Transforms/Passes.h.inc"

struct VerifyNoQuantumUseAfterFreePass
    : impl::VerifyNoQuantumUseAfterFreePassBase<VerifyNoQuantumUseAfterFreePass> {
    using VerifyNoQuantumUseAfterFreePassBase::VerifyNoQuantumUseAfterFreePassBase;

    void runOnOperation() final {
        Operation *mod = getOperation();
        DominanceInfo domInfo(mod);

        WalkResult wr = mod->walk([&](qref::QuantumOperation qOp) {
            for (Value &qubit : qOp.getQubitOperands()) {
                if (hasUseAfterFree(qubit, qOp, domInfo)) {
                    qOp.emitOpError("Detected use of a qubit after deallocation");
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
        if (wr.wasInterrupted()) {
            return signalPassFailure();
        }
    }
};

} // namespace qref
} // namespace catalyst
