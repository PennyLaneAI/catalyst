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

// SharedParameterBufferPass — Phase 3, Task 4
//
// Computes the compile-time layout of the shared parameter buffer θ*_rep that
// holds the optimised (γ, β) angles for every cluster representative.
//
// After this pass, every freeze_partition op carries:
//
//   param_buffer_size  : i32
//       Number of f64 values per parameter vector. Always 2 for p=1 QAOA
//       (one γ and one β). The LLVM lowering will allocate this many doubles
//       per cluster-representative slot.
//
//   buffer_slot_map    : array<i32, 2^m>
//       buffer_slot_map[k] = cluster index c of sub-problem k.
//       Representatives write their result into slot c; non-representatives
//       read from the same slot (direct copy) or use it as warm-start init.
//
//   init_params        : tensor<K × 2 × f64>
//       Row c = [γ_init, β_init] for cluster c's representative.
//       Uses basin_{gamma,beta} when bias-shift analysis found a reliable basin;
//       falls back to {init_gamma, init_beta} from shortcut initialisation.
//
//   use_atomic_guards  : i32  (always 1)
//       Signals the LLVM lowering pass to emit a global spin-lock alongside
//       the parameter buffer global so concurrent sub-problem threads do not
//       race on reads/writes.
//
// Requires: training_schedule, schedule_phase_ends, schedule_sources,
//           cluster_assignments, cluster_k  (from doqaoa-training-schedule
//           and its predecessors in the full pipeline).
//
// Optional (falls back gracefully):
//   basin_gamma, basin_beta  — from doqaoa-bias-shift
//   init_gamma,  init_beta   — from doqaoa-bias-shift

#include <cstdint>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_SHAREDPARAMETERBUFFERPASS
#define GEN_PASS_DEF_SHAREDPARAMETERBUFFERPASS
#include "Quantum/Transforms/Passes.h.inc"

struct SharedParameterBufferPass
    : impl::SharedParameterBufferPassBase<SharedParameterBufferPass> {
    using SharedParameterBufferPassBase::SharedParameterBufferPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx  = &getContext();
        Builder builder(ctx);

        func.walk([&](FreezePartitionOp op) {

            // ── Guard: need training-schedule annotations ─────────────────
            auto schedAttr  = op->getAttr("training_schedule");
            auto phaseAttr  = op->getAttr("schedule_phase_ends");
            auto srcAttr    = op->getAttr("schedule_sources");
            auto clAsnAttr  = op->getAttr("cluster_assignments");
            auto clKAttr    = op->getAttr("cluster_k");

            if (!schedAttr || !phaseAttr || !srcAttr || !clAsnAttr || !clKAttr) {
                op->emitWarning()
                    << "doqaoa-shared-buffer: missing training-schedule "
                       "attributes; run doqaoa-training-schedule first, skipping";
                return;
            }

            // ── Extract dimensions ────────────────────────────────────────
            auto hotspotAttr = op.getHotspotIndices();
            unsigned m       = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP   = 1u << m;

            int32_t K = cast<IntegerAttr>(clKAttr).getInt();
            if (K <= 0) K = 1;

            auto clAsnArr = cast<DenseI32ArrayAttr>(clAsnAttr);

            // ── Build buffer_slot_map[k] = cluster index of sub-problem k ─
            std::vector<int32_t> slotMap(numSP);
            for (unsigned k = 0; k < numSP; ++k) {
                int32_t c = (k < static_cast<unsigned>(clAsnArr.size()))
                                ? clAsnArr[k]
                                : 0;
                slotMap[k] = c;
            }

            // ── Build init_params: tensor<K × 2 × f64> ───────────────────
            //
            // Row c holds [γ_init, β_init] for the representative of cluster c.
            // Priority: basin_gamma/beta > init_gamma/beta > defaults.

            double defaultGamma = -M_PI / 6.0;
            double defaultBeta  = -M_PI / 8.0;

            double ig = defaultGamma, ib = defaultBeta;
            double bg = defaultGamma, bb = defaultBeta;

            if (auto a = op->getAttr("init_gamma"))
                ig = cast<FloatAttr>(a).getValueAsDouble();
            if (auto a = op->getAttr("init_beta"))
                ib = cast<FloatAttr>(a).getValueAsDouble();
            if (auto a = op->getAttr("basin_gamma"))
                bg = cast<FloatAttr>(a).getValueAsDouble();
            if (auto a = op->getAttr("basin_beta"))
                bb = cast<FloatAttr>(a).getValueAsDouble();

            // All clusters share the same shortcut/basin init in p=1 DO-QAOA.
            // (Per-cluster init is left for future p>1 extension.)
            std::vector<double> initFlat;
            initFlat.reserve(static_cast<unsigned>(K) * 2);
            for (int32_t c = 0; c < K; ++c) {
                initFlat.push_back(bg); // γ: prefer basin centre
                initFlat.push_back(bb); // β: prefer basin centre
            }

            // Pack into tensor<K × 2 × f64>
            auto initType  = RankedTensorType::get(
                {static_cast<int64_t>(K), 2}, builder.getF64Type());
            auto initAttr  = DenseElementsAttr::get(
                initType, llvm::ArrayRef<double>(initFlat));

            // ── Annotate op ───────────────────────────────────────────────
            op->setAttr("param_buffer_size",
                        builder.getI32IntegerAttr(2)); // γ and β
            op->setAttr("buffer_slot_map",
                        DenseI32ArrayAttr::get(ctx, slotMap));
            op->setAttr("init_params", initAttr);
            op->setAttr("use_atomic_guards",
                        builder.getI32IntegerAttr(1));

            // Remark summarising buffer layout
            llvm::SmallString<180> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-shared-buffer: K=" << K
               << " slots, param_buffer_size=2"
               << " | init=[" << llvm::format("%.4f", bg)
               << ", " << llvm::format("%.4f", bb) << "]"
               << " | atomic_guards=1";
            op->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
