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

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Utils/SCFUtils.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/IR/QECOps.h"
#include "QEC/Utils/QECLayer.h"
#include "QEC/Utils/QECOpUtils.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;
using json = nlohmann::json;

namespace catalyst {
namespace qec {

#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#include "QEC/Transforms/Passes.h.inc"

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

    LogicalResult countPPM(qec::PPMeasurementOp op,
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

    LogicalResult countPPR(qec::PPRotationOp op,
                           llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> &PPMSpecs,
                           llvm::BumpPtrAllocator &stringAllocator)
    {
        if (isOpInIfOp(op) || isOpInWhileOp(op)) {
            return op->emitOpError(
                "PPM statistics is not available when there are conditionals or while loops.");
        }

        int16_t rotationKind = op.getRotationKindAttr().getValue().getZExtValue();
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

    bool commuteToLayer(QECOpInterface rhsOp, QECLayer &lhsLayer)
    {
        for (auto lhsOp : lhsLayer.getOps()) {
            if (!commutes(rhsOp, lhsOp)) {
                return false;
            }
        }
        return true;
    }

    bool isPPR(QECOpInterface op) { return isa<qec::PPRotationOp>(op); }
    bool isPPM(QECOpInterface op) { return isa<qec::PPMeasurementOp>(op); }

    // Check if two ops have the same rotation kind.
    bool equalTypes(QECOpInterface lhsOp, QECOpInterface rhsOp)
    {
        return (isPPR(lhsOp) == isPPR(rhsOp) || isPPM(lhsOp) == isPPM(rhsOp));
    }

    // Add op to current layer if it commutes with the last op in the layer and has the same type.
    bool canAddToCurrentLayer(QECOpInterface op, QECLayer &currentLayer)
    {
        if (currentLayer.empty())
            return true;

        auto lastOp = currentLayer.getOps().back();
        return equalTypes(lastOp, op) && commuteToLayer(op, currentLayer);
    }

    void countDepths(std::vector<QECLayer> &layers,
                     llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> &PPMSpecs,
                     llvm::BumpPtrAllocator &stringAllocator)
    {
        for (auto &layer : layers) {
            assert(!layer.empty() && "Layer is empty");

            auto op = layer.getOps().back();

            int16_t absRk = 0;
            if (auto pprOp = dyn_cast<PPRotationOp>(op.getOperation())) {
                absRk = std::abs(static_cast<int16_t>(pprOp.getRotationKind()));
            }
            auto parentFuncOp = op->getParentOfType<func::FuncOp>();
            StringRef funcName = parentFuncOp.getName();
            llvm::StringSaver saver(stringAllocator);
            StringRef key = isPPR(op) ? saver.save("depth_pi" + std::to_string(absRk) + "_ppr")
                                      : saver.save("depth_ppm");

            PPMSpecs[funcName][key]++;
        }
    }

    LogicalResult printSpecs()
    {
        llvm::BumpPtrAllocator stringAllocator;
        llvm::DenseMap<StringRef, llvm::DenseMap<StringRef, int>> PPMSpecs;

        QECLayerContext layerContext;
        QECLayer currentLayer(&layerContext);
        std::vector<QECLayer> layers;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        WalkResult wr = getOperation()->walk([&](Operation *op) {
            // Count Depth
            if (auto qecOp = dyn_cast<QECOpInterface>(op)) {
                if (!canAddToCurrentLayer(qecOp, currentLayer)) {
                    layers.emplace_back(std::move(currentLayer));
                    currentLayer = QECLayer(&layerContext);
                }
                currentLayer.insertToLayer(qecOp);
            }

            // Count Logical Qubit
            if (isa<quantum::AllocOp>(op)) {
                if (failed(countLogicalQubit(op, PPMSpecs))) {
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            }

            // Count PPMs
            else if (isa<qec::PPMeasurementOp>(op)) {
                if (failed(countPPM(cast<qec::PPMeasurementOp>(op), PPMSpecs))) {
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            }

            // Count PPRs
            else if (isa<qec::PPRotationOp>(op)) {
                if (failed(countPPR(cast<qec::PPRotationOp>(op), PPMSpecs, stringAllocator))) {
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

        // Add the last layer if it is not empty.
        if (!currentLayer.empty()) {
            layers.emplace_back(std::move(currentLayer));
        }

        countDepths(layers, PPMSpecs, stringAllocator);

        json PPMSpecsJson = PPMSpecs;
        llvm::outs() << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        return success();
    }

    void runOnOperation() final
    {
        if (failed(printSpecs())) {
            signalPassFailure();
        }
    }
};

} // namespace qec
} // namespace catalyst
