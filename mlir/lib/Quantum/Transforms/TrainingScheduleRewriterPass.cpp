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

// TrainingScheduleRewriterPass — Phase 3, Task 2
//
// Lowers 2^m sub-problems into a three-phase training schedule:
//
//   Phase 1 — Full optimisation  (mode 0):  K representatives, fullEpochs each
//   Phase 2 — Warm-start         (mode 2):  non-reps with large ΔB, warmstartEpochs each
//   Phase 3 — Direct transfer    (mode 1):  non-reps with small ΔB, 0 epochs
//
// Requires: transfer_modes, representatives, cluster_assignments, cluster_k
//   (all from doqaoa-representative-selection, which requires the full prior chain).
//
// Attributes written onto freeze_partition:
//   training_schedule   : array<i32, 2^m>  -- k indices in execution order
//   schedule_phase_ends : array<i32: 3>    -- [K, K+warm_count, 2^m]
//   schedule_epochs     : array<i32, 2^m>  -- epoch budget per k (original order)
//   schedule_sources    : array<i32, 2^m>  -- param source sub-problem per k

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

#define GEN_PASS_DECL_TRAININGSCHEDULEREWRITERPASS
#define GEN_PASS_DEF_TRAININGSCHEDULEREWRITERPASS
#include "Quantum/Transforms/Passes.h.inc"

struct TrainingScheduleRewriterPass
    : impl::TrainingScheduleRewriterPassBase<TrainingScheduleRewriterPass> {
    using TrainingScheduleRewriterPassBase::TrainingScheduleRewriterPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx  = &getContext();

        unsigned fullEp      = fullEpochs;
        unsigned warmEp      = warmstartEpochs;

        func.walk([&](FreezePartitionOp op) {

            // ── Guard: need representative-selection annotations ───────────
            auto modesAttr  = op->getAttr("transfer_modes");
            auto repsAttr   = op->getAttr("representatives");
            auto clAsnAttr  = op->getAttr("cluster_assignments");
            auto clKAttr    = op->getAttr("cluster_k");

            if (!modesAttr || !repsAttr || !clAsnAttr || !clKAttr) {
                op->emitWarning()
                    << "doqaoa-training-schedule: missing representative-selection "
                       "attributes; run doqaoa-representative-selection first, skipping";
                return;
            }

            // ── Extract transfer modes ────────────────────────────────────
            auto hotspotAttr = op.getHotspotIndices();
            unsigned m       = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP   = 1u << m;

            auto modesArr = cast<DenseI32ArrayAttr>(modesAttr);
            auto repsArr  = cast<DenseI32ArrayAttr>(repsAttr);
            auto clAsnArr = cast<DenseI32ArrayAttr>(clAsnAttr);

            // ── Build schedule in three phases ────────────────────────────
            //
            // training_schedule — ordered list of sub-problem indices:
            //   [phase-1 reps | phase-2 warm | phase-3 copy]
            //
            // schedule_epochs[k]  — epoch budget for sub-problem k (original order)
            // schedule_sources[k] — which rep to take params from (original order)

            std::vector<int32_t> schedule;
            schedule.reserve(numSP);

            std::vector<int32_t> epochs(numSP, 0);
            std::vector<int32_t> sources(numSP, -1);

            int32_t phase1End, phase2End;

            // Phase 1: representatives (mode 0)
            for (unsigned k = 0; k < numSP; ++k) {
                if (modesArr[k] == 0) {
                    schedule.push_back(static_cast<int32_t>(k));
                    epochs[k]  = static_cast<int32_t>(fullEp);
                    sources[k] = static_cast<int32_t>(k); // optimises itself
                }
            }
            phase1End = static_cast<int32_t>(schedule.size());

            // Phase 2: warm-start (mode 2)
            for (unsigned k = 0; k < numSP; ++k) {
                if (modesArr[k] == 2) {
                    schedule.push_back(static_cast<int32_t>(k));
                    epochs[k] = static_cast<int32_t>(warmEp);
                    // source = the representative of this sub-problem's cluster
                    unsigned c = static_cast<unsigned>(clAsnArr[k]);
                    sources[k] = (c < static_cast<unsigned>(repsArr.size()))
                                     ? repsArr[c]
                                     : static_cast<int32_t>(k);
                }
            }
            phase2End = static_cast<int32_t>(schedule.size());

            // Phase 3: direct copy (mode 1)
            for (unsigned k = 0; k < numSP; ++k) {
                if (modesArr[k] == 1) {
                    schedule.push_back(static_cast<int32_t>(k));
                    epochs[k] = 0; // zero training cost
                    unsigned c = static_cast<unsigned>(clAsnArr[k]);
                    sources[k] = (c < static_cast<unsigned>(repsArr.size()))
                                     ? repsArr[c]
                                     : static_cast<int32_t>(k);
                }
            }

            // ── Build phase_ends array: [K, K+warm_count, 2^m] ───────────
            std::vector<int32_t> phaseEnds = {
                phase1End,
                phase2End,
                static_cast<int32_t>(numSP)
            };

            // ── Annotate op ───────────────────────────────────────────────
            op->setAttr("training_schedule",
                        DenseI32ArrayAttr::get(ctx, schedule));
            op->setAttr("schedule_phase_ends",
                        DenseI32ArrayAttr::get(ctx, phaseEnds));
            op->setAttr("schedule_epochs",
                        DenseI32ArrayAttr::get(ctx, epochs));
            op->setAttr("schedule_sources",
                        DenseI32ArrayAttr::get(ctx, sources));

            // Emit informational remark summarising the schedule
            int32_t warmCount  = phase2End - phase1End;
            int32_t copyCount  = static_cast<int32_t>(numSP) - phase2End;
            llvm::SmallString<200> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-training-schedule: "
               << numSP << " sub-problems — "
               << "phase1(full_opt)=" << phase1End
               << " phase2(warm_start)=" << warmCount
               << " phase3(direct_copy)=" << copyCount
               << " | epochs: full=" << fullEp
               << " warm=" << warmEp;
            op->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
