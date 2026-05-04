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

// AggregateMinLoweringPass — Phase 3, Task 7
//
// Resolves quantum.aggregate_min at compile time using the pre-computed energy
// values from doqaoa-warmstart-scheduler (warmstart_final_energy attribute).
//
// Algorithm:
//   For sub-problems with a finite (non-NaN) energy in warmstart_final_energy:
//     best_k = argmin(energy_k)
//   If all energies are NaN (e.g. K=1, single mode-0 rep only):
//     best_k = 0   (runtime will refine; this is a conservative default)
//
// The best_k is encoded as a bitstring of length m:
//   bitstring[i] = (best_k >> i) & 1
//   bit = 0  →  hotspot qubit i is in spin +1 state
//   bit = 1  →  hotspot qubit i is in spin −1 state
//
// Attributes written onto freeze_partition:
//   agg_best_k              : i32
//   agg_min_energy          : f64   (NaN when all energies are NaN)
//   agg_best_bitstring      : array<i32, m>
//   agg_candidates_evaluated: i32   (sub-problems with finite energy)
//
// Requires: warmstart_final_energy (from doqaoa-warmstart-scheduler).
// Falls back gracefully when the attribute is absent.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "llvm/Support/Format.h"
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

#define GEN_PASS_DECL_AGGREGATEMINLOWERINGPASS
#define GEN_PASS_DEF_AGGREGATEMINLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct AggregateMinLoweringPass
    : impl::AggregateMinLoweringPassBase<AggregateMinLoweringPass> {
    using AggregateMinLoweringPassBase::AggregateMinLoweringPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx  = &getContext();
        Builder builder(ctx);

        func.walk([&](FreezePartitionOp fpOp) {

            // ── Extract hotspot count m ───────────────────────────────────
            auto hotspotAttr = fpOp.getHotspotIndices();
            unsigned m       = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP   = 1u << m;

            // ── Try to get pre-computed energies ──────────────────────────
            // warmstart_final_energy: tensor<2^m × f64>
            // mode-0 and mode-1 rows are NaN (not evaluated at compile time).
            // mode-2 rows hold the post-Adam ⟨H_k⟩.
            std::vector<double> energies(numSP,
                std::numeric_limits<double>::quiet_NaN());

            if (auto attr = fpOp->getAttr("warmstart_final_energy")) {
                auto dense = cast<DenseElementsAttr>(attr);
                unsigned idx = 0;
                for (double v : dense.getValues<double>()) {
                    if (idx >= numSP) break;
                    energies[idx++] = v;
                }
            }

            // ── Find argmin over finite energies ──────────────────────────
            int32_t bestK   = 0;
            double  minE    = std::numeric_limits<double>::quiet_NaN();
            int32_t evalCnt = 0;

            for (unsigned k = 0; k < numSP; ++k) {
                double e = energies[k];
                if (std::isnan(e))
                    continue;
                ++evalCnt;
                if (std::isnan(minE) || e < minE) {
                    minE  = e;
                    bestK = static_cast<int32_t>(k);
                }
            }
            // If all NaN: bestK stays 0, minE stays NaN — runtime refines.

            // ── Encode best_k as bitstring ────────────────────────────────
            // bitstring[i] = (bestK >> i) & 1  for i in 0..m-1
            std::vector<int32_t> bitstring(m);
            for (unsigned i = 0; i < m; ++i)
                bitstring[i] = (bestK >> static_cast<int>(i)) & 1;

            // ── Annotate freeze_partition ─────────────────────────────────
            fpOp->setAttr("agg_best_k",
                          builder.getI32IntegerAttr(bestK));
            fpOp->setAttr("agg_best_bitstring",
                          DenseI32ArrayAttr::get(ctx, bitstring));
            fpOp->setAttr("agg_candidates_evaluated",
                          builder.getI32IntegerAttr(evalCnt));

            // agg_min_energy: NaN → 0x7FF8000000000000 in f64
            auto f64Ty = Float64Type::get(ctx);
            fpOp->setAttr("agg_min_energy",
                          FloatAttr::get(f64Ty, minE));

            // ── Remark ────────────────────────────────────────────────────
            llvm::SmallString<160> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-aggregate-min: best_k=" << bestK;
            if (!std::isnan(minE))
                ss << " min_energy=" << llvm::format("%.6f", minE);
            else
                ss << " min_energy=NaN (mode-0 runtime optimised)";
            ss << " | candidates_evaluated=" << evalCnt << "/" << numSP;
            ss << " | bitstring=[";
            for (unsigned i = 0; i < m; ++i) {
                ss << bitstring[i];
                if (i + 1 < m) ss << ",";
            }
            ss << "]";
            fpOp->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
