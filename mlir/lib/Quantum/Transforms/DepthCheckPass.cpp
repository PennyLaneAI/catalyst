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

// DepthCheckPass — Phase 3, Task 9
//
// Computes per-sub-problem CNOT counts from graph topology and optionally
// asserts that the maximum does not exceed a given bound (regression mode).
//
// This is the regression gate that enforces Table IV of arXiv:2602.21689v1:
//
//   Table IV (selected rows):
//     10-node MaxCut  m=2   representative sub-circuit: 2 × |E_free| CNOTs
//     Power-Law       m=3   representative sub-circuit: ≤ 3 CNOTs
//
// CNOT count formula (p=1 QAOA):
//   cnt = 2 × |{(u,v) upper-triangle : u,v free, |J[u,v]| > 0}|
//
// When expectedMaxCnots > 0 (regression mode):
//   - If max_cnots > expectedMaxCnots: signalPassFailure() (hard error)
//   - depth_regression_ok = 0 in this case
//
// Attributes written onto freeze_partition:
//   depth_cnot_counts    : array<i32, 2^m>
//   depth_max_cnots      : i32
//   depth_regression_ok  : i32   (1 if max <= expectedMaxCnots or no bound set)

#include <cmath>
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

#define GEN_PASS_DECL_DEPTHCHECKPASS
#define GEN_PASS_DEF_DEPTHCHECKPASS
#include "Quantum/Transforms/Passes.h.inc"

// Count free-free upper-triangle edges with |J[u,v]| > 0 (same as noise pass).
static int32_t cnotCount(const std::vector<double> &J, unsigned N, const std::vector<bool> &frozen)
{
    int32_t cnt = 0;
    constexpr double kTol = 1e-12;
    for (unsigned u = 0; u < N; ++u) {
        if (frozen[u])
            continue;
        for (unsigned v = u + 1; v < N; ++v) {
            if (frozen[v])
                continue;
            if (std::fabs(J[u * N + v]) > kTol)
                ++cnt;
        }
    }
    return 2 * cnt; // 2 CNOTs per ZZ edge
}

static std::vector<double> dcpExtractJ(Operation *op, unsigned &numNodes)
{
    auto hq = op->getAttr("h_quad");
    if (!hq) {
        numNodes = 0;
        return {};
    }
    if (auto dense = dyn_cast<DenseGraphAttr>(hq)) {
        numNodes = static_cast<unsigned>(dense.getNumNodes());
        std::vector<double> J;
        for (double v : dense.getWeights().getValues<double>())
            J.push_back(v);
        return J;
    }
    if (auto sparse = dyn_cast<SparseGraphAttr>(hq)) {
        numNodes = static_cast<unsigned>(sparse.getNumNodes());
        unsigned N = numNodes;
        std::vector<double> J(static_cast<size_t>(N) * N, 0.0);
        auto rows = sparse.getRowIndices();
        auto cols = sparse.getColIndices();
        auto wt = sparse.getWeights().getValues<double>();
        auto wit = wt.begin();
        for (unsigned e = 0; e < static_cast<unsigned>(rows.size()); ++e, ++wit) {
            unsigned r = static_cast<unsigned>(rows[e]);
            unsigned c = static_cast<unsigned>(cols[e]);
            J[r * N + c] = *wit;
            J[c * N + r] = *wit;
        }
        return J;
    }
    numNodes = 0;
    return {};
}

struct DepthCheckPass : impl::DepthCheckPassBase<DepthCheckPass> {
    using DepthCheckPassBase::DepthCheckPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();
        Builder builder(ctx);

        unsigned bound = expectedMaxCnots;

        func.walk([&](FreezePartitionOp fpOp) {
            // ── Extract graph ─────────────────────────────────────────────
            unsigned numNodes = 0;
            auto J = dcpExtractJ(fpOp, numNodes);
            if (numNodes == 0) {
                fpOp->emitWarning() << "doqaoa-depth-check: missing h_quad, skipping";
                return;
            }

            auto hotspotAttr = fpOp.getHotspotIndices();
            unsigned m = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP = 1u << m;

            // Build frozen mask
            std::vector<bool> frozen(numNodes, false);
            for (auto idx : hotspotAttr)
                frozen[static_cast<unsigned>(idx)] = true;

            // ── CNOT count is identical for all sub-problems in p=1 QAOA  ─
            // (freezing only changes effective bias, not free-free topology)
            int32_t repCnots = cnotCount(J, numNodes, frozen);
            std::vector<int32_t> cnotCounts(numSP, repCnots);
            int32_t maxCnots = repCnots;

            // ── Regression gate ───────────────────────────────────────────
            bool regrOk = true;
            if (bound > 0 && static_cast<unsigned>(maxCnots) > bound) {
                regrOk = false;
                fpOp->emitError() << "doqaoa-depth-check: max_cnots=" << maxCnots
                                  << " exceeds expected-max-cnots=" << bound
                                  << " (Table IV regression failure)";
                signalPassFailure();
                return;
            }

            // ── Annotate ──────────────────────────────────────────────────
            fpOp->setAttr("depth_cnot_counts", DenseI32ArrayAttr::get(ctx, cnotCounts));
            fpOp->setAttr("depth_max_cnots", builder.getI32IntegerAttr(maxCnots));
            fpOp->setAttr("depth_regression_ok", builder.getI32IntegerAttr(regrOk ? 1 : 0));

            // Remark
            llvm::SmallString<160> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-depth-check: max_cnots=" << maxCnots << " free_edges=" << (maxCnots / 2)
               << " N_free=" << (numNodes - m) << " regression_ok=" << (regrOk ? "1" : "0");
            if (bound > 0)
                ss << " (bound=" << bound << ")";
            fpOp->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
