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

// BiasShiftAnalysis — Phase 2, Tasks 5/7/8
//
// Requires: freeze_partition ops already annotated by doqaoa-landscape-overlap
//   (cluster_k, cluster_assignments, h_quad, h_lin, hotspot_indices).
//
// Produces per-op attributes:
//   Task 5 — b_values, bias_shifts, representatives
//   Task 7 — init_gamma, init_beta
//   Task 8 — basin_gamma, basin_beta  (+ optional warning)

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
#include "Quantum/Transforms/EnergyEval.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_BIASSHIFTANALYSISPASS
#define GEN_PASS_DEF_BIASSHIFTANALYSISPASS
#include "Quantum/Transforms/Passes.h.inc"

// ─────────────────────────────────────────────────────────────────────────────
// Attribute extraction helpers (shared with LandscapeOverlapAnalysis)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> bsaExtractJ(Operation *op, unsigned &numNodes)
{
    auto hq = op->getAttr("h_quad");
    if (!hq)
        return {};

    if (auto d = dyn_cast<DenseGraphAttr>(hq)) {
        numNodes = d.getNumNodes();
        std::vector<double> J(numNodes * numNodes, 0.0);
        unsigned i = 0;
        for (double v : d.getWeights().getValues<double>())
            J[i++] = v;
        return J;
    }
    if (auto s = dyn_cast<SparseGraphAttr>(hq)) {
        numNodes = s.getNumNodes();
        std::vector<double> J(numNodes * numNodes, 0.0);
        auto rows = s.getRowIndices();
        auto cols = s.getColIndices();
        unsigned e = 0;
        for (double v : s.getWeights().getValues<double>()) {
            unsigned r = rows[e], c = cols[e];
            J[r * numNodes + c] = v;
            J[c * numNodes + r] = v;
            ++e;
        }
        return J;
    }
    return {};
}

static std::vector<double> bsaExtractH(Operation *op, unsigned numNodes)
{
    std::vector<double> h(numNodes, 0.0);
    if (auto a = op->getAttr("h_lin"))
        if (auto d = dyn_cast<DenseElementsAttr>(a)) {
            unsigned i = 0;
            for (double v : d.getValues<double>())
                h[i++] = v;
        }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid coordinate recovery  (Task 8)
// Must match the grid formula used in EnergyEval::buildLandscapeVector.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double kPi = 3.14159265358979323846;

/// Convert landscape-vector argmin index to (gamma, beta) grid coordinates.
static std::pair<double, double> indexToAngles(std::size_t idx, unsigned gridSize)
{
    unsigned gi = static_cast<unsigned>(idx) / gridSize;
    unsigned gj = static_cast<unsigned>(idx) % gridSize;
    double gs1 = (gridSize > 1) ? static_cast<double>(gridSize - 1) : 1.0;
    double gamma = -kPi + 2.0 * kPi * gi / gs1;
    double beta = -kPi / 2.0 + kPi * gj / gs1;
    return {gamma, beta};
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct BiasShiftAnalysisPass : impl::BiasShiftAnalysisPassBase<BiasShiftAnalysisPass> {
    using BiasShiftAnalysisPassBase::BiasShiftAnalysisPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();
        Builder builder(ctx);

        // Physics-informed shortcut starting angles (Task 7)
        constexpr double kInitGamma = -kPi / 6.0; // ≈ −0.5236 rad
        constexpr double kInitBeta = -kPi / 8.0;  // ≈ −0.3927 rad

        double btVal = biasThreshold; // extract before format
        double tolVal = basinTol;
        unsigned gs = gridSize;

        func.walk([&](FreezePartitionOp op) {
            // ── Guard: need cluster_assignments from prior pass ────────────
            auto clKAttr = op->getAttr("cluster_k");
            auto clAsnAttr = op->getAttr("cluster_assignments");
            if (!clKAttr || !clAsnAttr) {
                op->emitWarning() << "doqaoa-bias-shift: missing cluster attributes; "
                                     "run doqaoa-landscape-overlap first, skipping";
                return;
            }

            // ── Extract graph data ────────────────────────────────────────
            unsigned numNodes = 0;
            std::vector<double> J = bsaExtractJ(op, numNodes);
            if (J.empty())
                return;
            std::vector<double> h = bsaExtractH(op, numNodes);

            auto hotspotAttr = op.getHotspotIndices();
            std::vector<int32_t> hotspotIndices(hotspotAttr.begin(), hotspotAttr.end());
            unsigned m = static_cast<unsigned>(hotspotIndices.size());
            unsigned numSP = 1u << m;

            // ── Read cluster assignments ──────────────────────────────────
            auto clAsnArr = cast<DenseI32ArrayAttr>(clAsnAttr);
            int32_t clusterK = cast<IntegerAttr>(clKAttr).getInt();

            // ── Build GraphDesc ───────────────────────────────────────────
            energy::GraphDesc graph;
            graph.numNodes = numNodes;
            graph.J = J;
            graph.h = h;
            graph.hotspotIndices = hotspotIndices;

            // ── Task 5a: compute B_k for every sub-problem ────────────────
            std::vector<double> bValues(numSP);
            for (unsigned k = 0; k < numSP; ++k)
                bValues[k] = energy::computeBias(graph, k);

            // ── Task 5b: find representative per cluster (min B_k) ────────
            // representative[c] = sub-problem in cluster c with min B_k
            std::vector<int32_t> representatives(static_cast<unsigned>(clusterK), -1);
            std::vector<double> repB(static_cast<unsigned>(clusterK),
                                     std::numeric_limits<double>::max());

            for (unsigned k = 0; k < numSP; ++k) {
                unsigned c = static_cast<unsigned>(clAsnArr[k]);
                if (bValues[k] < repB[c]) {
                    repB[c] = bValues[k];
                    representatives[c] = static_cast<int32_t>(k);
                }
            }

            // ── Task 5c: compute ΔB_k for every sub-problem ───────────────
            std::vector<double> biasShifts(numSP);
            for (unsigned k = 0; k < numSP; ++k) {
                unsigned c = static_cast<unsigned>(clAsnArr[k]);
                biasShifts[k] = std::abs(bValues[k] - repB[c]);
            }

            // ── Task 7: shortcut init angles ──────────────────────────────
            // (universal for all sub-problems in the concentrated p=1 regime)

            // ── Task 8: basin analysis on the primary representative ───────
            // representative of cluster 0 (the primary / only cluster)
            unsigned repK = static_cast<unsigned>(representatives[0]);

            // buildLandscapeVector: cache-hit if doqaoa-landscape-overlap ran
            // first on the same FuncOp (cache is NOT flushed here).
            std::vector<double> landscape = energy::buildLandscapeVector(graph, repK, gs);

            // argmin of the landscape (lowest energy = deepest basin)
            std::size_t argmin = 0;
            for (std::size_t i = 1; i < landscape.size(); ++i)
                if (landscape[i] < landscape[argmin])
                    argmin = i;

            auto [basinGamma, basinBeta] = indexToAngles(argmin, gs);

            // Warn if basin centre deviates more than basinTol from shortcut
            double dgamma = std::abs(basinGamma - kInitGamma);
            double dbeta = std::abs(basinBeta - kInitBeta);
            if (dgamma > tolVal || dbeta > tolVal) {
                llvm::SmallString<160> msg;
                llvm::raw_svector_ostream ss(msg);
                ss << llvm::format("doqaoa-bias-shift: basin centre (%.3f, %.3f) deviates "
                                   "from shortcut (%.3f, %.3f) by (%.3f, %.3f) > tol=%.3f; "
                                   "shortcut init may be suboptimal for this graph",
                                   basinGamma, basinBeta, kInitGamma, kInitBeta, dgamma, dbeta,
                                   tolVal);
                op->emitWarning() << msg;
            }

            // ── Annotate op ───────────────────────────────────────────────

            // b_values and bias_shifts stored as tensor<numSP x f64>
            auto f64TensorType =
                RankedTensorType::get({static_cast<int64_t>(numSP)}, builder.getF64Type());

            op->setAttr("b_values",
                        DenseElementsAttr::get(f64TensorType, ArrayRef<double>(bValues)));
            op->setAttr("bias_shifts",
                        DenseElementsAttr::get(f64TensorType, ArrayRef<double>(biasShifts)));
            op->setAttr("representatives", DenseI32ArrayAttr::get(ctx, representatives));
            op->setAttr("init_gamma", builder.getF64FloatAttr(kInitGamma));
            op->setAttr("init_beta", builder.getF64FloatAttr(kInitBeta));
            op->setAttr("basin_gamma", builder.getF64FloatAttr(basinGamma));
            op->setAttr("basin_beta", builder.getF64FloatAttr(basinBeta));

            // Emit a note for direct-copy candidates (ΔB < threshold)
            unsigned directCopy = 0;
            for (double db : biasShifts)
                if (db < btVal)
                    ++directCopy;
            (void)directCopy; // available for downstream passes
        });
    }
};

} // namespace quantum
} // namespace catalyst
