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

    LogicalResult
    countLogicalQubit(Operation *op,
                      llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> &PPMSpecs)
    {
        uint64_t numQubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(0);

        if (numQubits == 0) {
            return op->emitOpError("PPM specs with dynamic number of qubits is not supported");
        }

        auto parentFuncOp = op->getParentOfType<func::FuncOp>();
        PPMSpecs[parentFuncOp.getName()]["logical_qubits"] = numQubits;
        return success();
    }

    LogicalResult countPPM(pbc::PPMeasurementOp op,
                           llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> &PPMSpecs)
    {
        if (isOpInIfOp(op) || isOpInWhileOp(op)) {
            return op->emitOpError(
                "PPM statistics is not available when there are conditionals or while loops.");
        }

        auto parentFuncOp = op->getParentOfType<func::FuncOp>();

        // Handle when PPM op is in a static for loop
        // Note that countStaticForloopIterations returns -1 when it bails out for
        // dynamic loop bounds
        // When bailing out on dynamic, just error.
        int64_t forLoopMultiplier = countStaticForloopIterations(op);
        if (forLoopMultiplier == -1) {
            return op->emitOpError(
                "PPM statistics is not available when there are dynamically sized for loops.");
        }
        PPMSpecs[parentFuncOp.getName()]["num_of_ppm"] += forLoopMultiplier;
        return success();
    }

    LogicalResult countPPR(pbc::PPRotationOp op,
                           llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> &PPMSpecs,
                           llvm::BumpPtrAllocator &stringAllocator)
    {
        if (isOpInIfOp(op) || isOpInWhileOp(op)) {
            return op->emitOpError(
                "PPM statistics is not available when there are conditionals or while loops.");
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
                "PPM statistics is not available when there are dynamically sized for loops.");
        }
        PPMSpecs[funcName][numRotationKindKey] += forLoopMultiplier;

        PPMSpecs[funcName][maxWeightRotationKindKey] =
            std::max(PPMSpecs[funcName][maxWeightRotationKindKey],
                     static_cast<int>(PauliProductAttr.size()));
        return success();
    }

    LogicalResult printSpecs(bool onlyDisjointQubit)
    {
        llvm::BumpPtrAllocator stringAllocator;
        llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> PPMSpecs;

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

        // Count depth using the same layer grouping as partition-layers.
        PBCLayerContext layerContext;
        auto groupLayers = layerContext.groupLayers(getOperation(), onlyDisjointQubit);

        if (!groupLayers.empty() && !groupLayers.front().empty()) {
            auto funcOp =
                groupLayers.front().front().getOperation()->getParentOfType<func::FuncOp>();
            if (!funcOp) {
                return groupLayers.front().front().emitOpError(
                    "expected PBC op to be nested in func.func");
            }
            StringRef funcName = funcOp.getName();

            // depth_type:
            // depth-0: Depth calculated by allowing any commuting logical PPMs to be computed
            // together, even with overlapping support.
            // depth-1: Depth calculated by allowing only PPMs with non-overlapping support to be
            // computed together.
            PPMSpecs[funcName]["depth_type"] = onlyDisjointQubit ? 1 : 0;
            PPMSpecs[funcName]["depth"] = groupLayers.size();
        }

        json PPMSpecsJson = PPMSpecs;
        llvm::outs() << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        return success();
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
