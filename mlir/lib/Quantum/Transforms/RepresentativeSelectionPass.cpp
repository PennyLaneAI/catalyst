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

// RepresentativeSelectionPass — Phase 3, Task 1
//
// Requires: freeze_partition ops already annotated by doqaoa-bias-shift
//   (cluster_k, cluster_assignments, representatives, bias_shifts, b_values).
//
// For each sub-problem k in every freeze_partition:
//   - If k is the cluster representative → mode 0  (full optimisation)
//   - Else if bias_shifts[k] < biasThreshold → mode 1  (direct copy, zero cost)
//   - Else → mode 2  (warm start from representative parameters)
//
// Transfer mode encoding:
//   0 = representative  (needs full variational optimisation)
//   1 = direct copy     (parameters inherited from rep unchanged)
//   2 = warm start      (short fine-tuning from rep parameters)
//
// Attributes written onto freeze_partition:
//   is_representative  : array<i32, 2^m>  -- 1 for cluster reps, 0 otherwise
//   transfer_modes     : array<i32, 2^m>  -- per-sub-problem mode code
//   direct_copy_count  : i32
//   warm_start_count   : i32

#include <cstdint>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_REPRESENTATIVESELECTIONPASS
#define GEN_PASS_DEF_REPRESENTATIVESELECTIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct RepresentativeSelectionPass
    : impl::RepresentativeSelectionPassBase<RepresentativeSelectionPass> {
    using RepresentativeSelectionPassBase::RepresentativeSelectionPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();
        Builder builder(ctx);

        double btVal = biasThreshold;

        func.walk([&](FreezePartitionOp op) {
            // ── Guard: need bias-shift annotations ────────────────────────
            auto clKAttr = op->getAttr("cluster_k");
            auto clAsnAttr = op->getAttr("cluster_assignments");
            auto repsAttr = op->getAttr("representatives");
            auto bsAttr = op->getAttr("bias_shifts");

            if (!clKAttr || !clAsnAttr || !repsAttr || !bsAttr) {
                op->emitWarning() << "doqaoa-representative-selection: missing bias-shift "
                                     "attributes; run doqaoa-bias-shift first, skipping";
                return;
            }

            // ── Extract cluster data ──────────────────────────────────────
            auto hotspotAttr = op.getHotspotIndices();
            unsigned m = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP = 1u << m;

            (void)clAsnAttr; // cluster_assignments not needed: we use repsArr directly
            auto repsArr = cast<DenseI32ArrayAttr>(repsAttr);

            // bias_shifts is a dense tensor<numSP x f64>
            auto bsDense = cast<DenseElementsAttr>(bsAttr);
            std::vector<double> biasShifts;
            biasShifts.reserve(numSP);
            for (double v : bsDense.getValues<double>())
                biasShifts.push_back(v);

            // Build a fast lookup: for each cluster c, which k is its rep?
            // repsArr[c] = index of representative sub-problem in cluster c
            std::vector<bool> isRep(numSP, false);
            for (unsigned c = 0; c < static_cast<unsigned>(repsArr.size()); ++c) {
                int32_t repK = repsArr[c];
                if (repK >= 0 && static_cast<unsigned>(repK) < numSP)
                    isRep[static_cast<unsigned>(repK)] = true;
            }

            // ── Compute per-sub-problem attributes ────────────────────────
            std::vector<int32_t> isRepArr(numSP, 0);
            std::vector<int32_t> modes(numSP, 0);
            int32_t directCopyCount = 0;
            int32_t warmStartCount = 0;

            for (unsigned k = 0; k < numSP; ++k) {
                if (isRep[k]) {
                    isRepArr[k] = 1;
                    modes[k] = 0; // representative: full optimisation
                }
                else if (biasShifts[k] < btVal) {
                    isRepArr[k] = 0;
                    modes[k] = 1; // direct copy
                    ++directCopyCount;
                }
                else {
                    isRepArr[k] = 0;
                    modes[k] = 2; // warm start
                    ++warmStartCount;
                }
            }

            // ── Annotate op ───────────────────────────────────────────────
            op->setAttr("is_representative", DenseI32ArrayAttr::get(ctx, isRepArr));
            op->setAttr("transfer_modes", DenseI32ArrayAttr::get(ctx, modes));
            op->setAttr("direct_copy_count", builder.getI32IntegerAttr(directCopyCount));
            op->setAttr("warm_start_count", builder.getI32IntegerAttr(warmStartCount));

            // Emit informational note on the transfer plan
            unsigned repCount = static_cast<unsigned>(repsArr.size());
            (void)repCount;
            llvm::SmallString<160> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-representative-selection: " << numSP
               << " sub-problems — reps=" << repsArr.size() << " copy=" << directCopyCount
               << " warm_start=" << warmStartCount;
            op->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
