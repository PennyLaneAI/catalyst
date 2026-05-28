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

#define DEBUG_TYPE "ppm-specs"

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "nlohmann/json.hpp"

#include "Catalyst/Utils/SCFUtils.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Utils/PBCLayer.h"
#include "PBC/Utils/PBCOpUtils.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;
using json = nlohmann::json;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#include "PBC/Transforms/Passes.h.inc"

struct CountPPMSpecsPass : public impl::CountPPMSpecsPassBase<CountPPMSpecsPass> {
    using CountPPMSpecsPassBase::CountPPMSpecsPassBase;

    using PPMSpecsType = llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>>;

    LogicalResult countLogicalQubit(Operation *op, PPMSpecsType &PPMSpecs)
    {
        uint64_t numQubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(0);

        if (numQubits == 0) {
            return op->emitOpError("PPM specs with dynamic number of qubits is not supported");
        }

        auto parentFuncOp = op->getParentOfType<func::FuncOp>();
        PPMSpecs[parentFuncOp.getName()]["logical_qubits"] = numQubits;
        return success();
    }

    LogicalResult countPPM(pbc::PPMeasurementOp op, PPMSpecsType &PPMSpecs)
    {
        if (isOpInWhileOp(op)) {
            return op->emitOpError("PBC statistics is not available when there are while loops.");
        }

        auto parentFuncOp = op->getParentOfType<func::FuncOp>();

        // Handle when PPM op is in a static for loop
        // Note that countStaticForloopIterations returns -1 when it bails out for
        // dynamic loop bounds
        // When bailing out on dynamic, just error.
        int64_t forLoopMultiplier = countStaticForloopIterations(op);
        if (forLoopMultiplier == -1) {
            return op->emitOpError(
                "PBC statistics is not available when there are dynamically sized for loops.");
        }
        PPMSpecs[parentFuncOp.getName()]["num_of_ppm"] += forLoopMultiplier;
        return success();
    }

    LogicalResult countPPR(pbc::PPRotationOp op, PPMSpecsType &PPMSpecs,
                           llvm::BumpPtrAllocator &stringAllocator)
    {
        if (isOpInWhileOp(op)) {
            return op->emitOpError("PBC statistics is not available when there are while loops.");
        }

        int8_t rotationKind = op.getRotationKind();
        auto PauliProductAttr = op.getPauliProductAttr();
        auto parentFuncOp = op->getParentOfType<func::FuncOp>();
        StringRef funcName = parentFuncOp.getName();
        llvm::StringSaver saver(stringAllocator);
        StringRef numRotationKindKey =
            saver.save("pi" + std::to_string(abs(rotationKind)) + "_ppr");
        StringRef maxWeightRotationKindKey =
            saver.save("max_weight_pi" + std::to_string(abs(rotationKind)));

        // Handle when PPR op is in a static for loop
        // Note that countStaticForloopIterations returns -1 when it bails out for
        // dynamic loop bounds
        // When bailing out on dynamic, just error.
        int64_t forLoopMultiplier = countStaticForloopIterations(op);
        if (forLoopMultiplier == -1) {
            return op->emitOpError(
                "PBC statistics is not available when there are dynamically sized for loops.");
        }
        PPMSpecs[funcName][numRotationKindKey] += forLoopMultiplier;

        PPMSpecs[funcName][maxWeightRotationKindKey] =
            std::max(PPMSpecs[funcName][maxWeightRotationKindKey],
                     static_cast<int>(PauliProductAttr.size()));
        return success();
    }

    // Worst-case PBC layer depth for a single function, treating scf.if as
    // a barrier whose contribution is `max(depth(then), depth(else))`.
    // depth_type:
    //   depth-0: any commuting PPMs may share a layer, even on overlapping qubits.
    //   depth-1: only PPMs on non-overlapping qubits may share a layer.
    LogicalResult computeFuncDepth(func::FuncOp funcOp, bool onlyDisjointQubit,
                                   PPMSpecsType &PPMSpecs)
    {
        if (!funcOp.getBody()
                 .walk([&](PBCOpInterface) { return WalkResult::interrupt(); })
                 .wasInterrupted()) {
            return success();
        }

        PBCLayerContext layerContext;
        FailureOr<int64_t> depth =
            layerContext.computeWorstCaseDepth(&funcOp.getBody().front(), onlyDisjointQubit);
        if (failed(depth)) {
            return failure();
        }

        StringRef funcName = funcOp.getName();
        PPMSpecs[funcName]["depth_type"] = onlyDisjointQubit ? 1 : 0;
        PPMSpecs[funcName]["depth"] = static_cast<int>(*depth);
        return success();
    }

    LogicalResult computeAllFuncDepth(bool onlyDisjointQubit, PPMSpecsType &PPMSpecs)
    {
        WalkResult wr = getOperation()->walk([&](func::FuncOp funcOp) {
            return failed(computeFuncDepth(funcOp, onlyDisjointQubit, PPMSpecs))
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
        });
        return wr.wasInterrupted() ? failure() : success();
    }

    LogicalResult printSpecs(bool onlyDisjointQubit)
    {
        llvm::BumpPtrAllocator stringAllocator;
        PPMSpecsType PPMSpecs;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        WalkResult wr = getOperation()->walk([&](Operation *op) {
            // Count Logical Qubit
            if (isa<quantum::AllocOp>(op)) {
                if (failed(countLogicalQubit(op, PPMSpecs))) {
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            }

            // Count PPMs
            else if (isa<pbc::PPMeasurementOp>(op)) {
                if (failed(countPPM(cast<pbc::PPMeasurementOp>(op), PPMSpecs))) {
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            }

            // Count PPRs
            else if (isa<pbc::PPRotationOp>(op)) {
                if (failed(countPPR(cast<pbc::PPRotationOp>(op), PPMSpecs, stringAllocator))) {
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            }

            // Skip other ops
            else {
                return WalkResult::advance();
            }
        });

        if (wr.wasInterrupted()) {
            return failure();
        }

        // Compute depth, but still emit JSON even if depth analysis fails for some
        // functions (counts may have succeeded). Pass failure is reported at the end.
        LogicalResult depthResult = computeAllFuncDepth(onlyDisjointQubit, PPMSpecs);

        json PPMSpecsJson = PPMSpecs;
        llvm::outs() << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        return depthResult;
    }

    void runOnOperation() final
    {
        if (failed(printSpecs(onlyDisjointQubit))) {
            signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
