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

// NoiseModelPreservationPass — Phase 3, Task 8
//
// Records IBM FakeBrisbane noise parameters on freeze_partition ops and
// validates that each sub-circuit's CNOT count stays within the T1 decoherence
// budget.
//
// CNOT count per sub-problem k  (p=1 QAOA):
//   cnt_k = 2 × |{(u,v) : u,v free, J[u*N+v] ≠ 0}|
//   (each ZZ Ising term compiles to a CNOT–Rz–CNOT pair → 2 CNOTs per edge)
//
// FakeBrisbane defaults (Table IV, arXiv:2602.21689v1):
//   T1           = 127 000 ns
//   T2           = 218 000 ns
//   CX fidelity  = 0.9918
//   CX gate time =    533 ns
//
// T1-budget check:
//   circuit_time_ns = noise_cnot_counts[k] × noise_cx_time_ns
//   depth_ok = circuit_time_ns < T1
//   If any sub-circuit exceeds T1, a warning is emitted (not a hard failure).
//
// expected-max-cnots option: if > 0, the pass signals failure when any
// sub-circuit's CNOT count exceeds this value (regression mode).
//
// Attributes written onto freeze_partition:
//   noise_t1_ns          : f64
//   noise_t2_ns          : f64
//   noise_cx_fidelity    : f64
//   noise_cx_time_ns     : f64
//   noise_cnot_counts    : array<i32, 2^m>
//   noise_max_cnots      : i32
//   noise_depth_ok       : i32   (1 if all sub-circuits within T1 budget)

#include <cmath>
#include <cstdint>
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

#define GEN_PASS_DECL_NOISEMODELPRESERVATIONPASS
#define GEN_PASS_DEF_NOISEMODELPRESERVATIONPASS
#include "Quantum/Transforms/Passes.h.inc"

// ─────────────────────────────────────────────────────────────────────────────
// Graph helpers (mirrors BiasShiftAnalysis pattern)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> nmpExtractJ(Operation *op, unsigned &numNodes)
{
    auto hq = op->getAttr("h_quad");
    if (!hq) { numNodes = 0; return {}; }

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
        auto wt   = sparse.getWeights().getValues<double>();
        auto wit  = wt.begin();
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

// Count free-free edges (J[u,v] != 0, both u and v are free) for sub-problem k.
static int cnotCountForSubproblem(const std::vector<double> &J,
                                   unsigned numNodes,
                                   const std::vector<int32_t> &hotspots,
                                   unsigned /*k*/)
{
    // Build free-qubit set (same for all k in p=1 QAOA — frozen qubits don't
    // change the free-free edge count, only the effective bias changes).
    std::vector<bool> isFrozen(numNodes, false);
    for (int32_t h : hotspots)
        if (static_cast<unsigned>(h) < numNodes)
            isFrozen[static_cast<unsigned>(h)] = true;

    // Count unique upper-triangle free-free edges with |J[u,v]| > threshold.
    int cnt = 0;
    constexpr double kTol = 1e-12;
    for (unsigned u = 0; u < numNodes; ++u) {
        if (isFrozen[u]) continue;
        for (unsigned v = u + 1; v < numNodes; ++v) {
            if (isFrozen[v]) continue;
            if (std::fabs(J[u * numNodes + v]) > kTol)
                ++cnt;
        }
    }
    // 2 CNOTs per ZZ edge (CNOT–Rz–CNOT)
    return 2 * cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass implementation
// ─────────────────────────────────────────────────────────────────────────────

struct NoiseModelPreservationPass
    : impl::NoiseModelPreservationPassBase<NoiseModelPreservationPass> {
    using NoiseModelPreservationPassBase::NoiseModelPreservationPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx  = &getContext();
        Builder builder(ctx);

        double t1   = noiseT1Ns;
        double t2   = noiseT2Ns;
        double fid  = noiseCxFidelity;
        double cxT  = noiseCxTimeNs;
        unsigned maxC = expectedMaxCnots;

        func.walk([&](FreezePartitionOp fpOp) {

            // ── Extract graph ────────────────────────────────────────────
            unsigned numNodes = 0;
            auto J = nmpExtractJ(fpOp, numNodes);
            if (numNodes == 0) {
                fpOp->emitWarning()
                    << "doqaoa-noise-preserve: missing h_quad, skipping";
                return;
            }

            auto hotspotAttr = fpOp.getHotspotIndices();
            unsigned m       = static_cast<unsigned>(hotspotAttr.size());
            unsigned numSP   = 1u << m;

            std::vector<int32_t> hotspots;
            for (auto idx : hotspotAttr)
                hotspots.push_back(static_cast<int32_t>(idx));

            // ── Compute CNOT counts per sub-problem ───────────────────────
            // For p=1, the free-free edge count is the same for all k
            // (frozen qubits only change effective bias, not the free sub-graph).
            int repCnots = cnotCountForSubproblem(J, numNodes, hotspots, 0);

            std::vector<int32_t> cnotCounts(numSP, repCnots);
            int32_t maxCnots = repCnots;

            // ── T1 budget check ───────────────────────────────────────────
            double circuitTimeNs = static_cast<double>(maxCnots) * cxT;
            bool depthOk = (circuitTimeNs < t1);

            if (!depthOk) {
                fpOp->emitWarning()
                    << "doqaoa-noise-preserve: circuit_time="
                    << llvm::format("%.0f", circuitTimeNs)
                    << "ns exceeds T1=" << llvm::format("%.0f", t1)
                    << "ns for max_cnots=" << maxCnots
                    << "; decoherence error budget exceeded";
            }

            // ── expected-max-cnots regression gate ───────────────────────
            if (maxC > 0 && static_cast<unsigned>(maxCnots) > maxC) {
                fpOp->emitError()
                    << "doqaoa-noise-preserve: max_cnots=" << maxCnots
                    << " exceeds expected-max-cnots=" << maxC
                    << " (regression failure)";
                signalPassFailure();
                return;
            }

            // ── Annotate ──────────────────────────────────────────────────
            auto f64Ty = Float64Type::get(ctx);
            fpOp->setAttr("noise_t1_ns",       FloatAttr::get(f64Ty, t1));
            fpOp->setAttr("noise_t2_ns",       FloatAttr::get(f64Ty, t2));
            fpOp->setAttr("noise_cx_fidelity", FloatAttr::get(f64Ty, fid));
            fpOp->setAttr("noise_cx_time_ns",  FloatAttr::get(f64Ty, cxT));
            fpOp->setAttr("noise_cnot_counts",
                          DenseI32ArrayAttr::get(ctx, cnotCounts));
            fpOp->setAttr("noise_max_cnots",
                          builder.getI32IntegerAttr(maxCnots));
            fpOp->setAttr("noise_depth_ok",
                          builder.getI32IntegerAttr(depthOk ? 1 : 0));

            // ── Remark ────────────────────────────────────────────────────
            llvm::SmallString<220> info;
            llvm::raw_svector_ostream ss(info);
            ss << "doqaoa-noise-preserve: max_cnots=" << maxCnots
               << " circuit_time=" << llvm::format("%.0f", circuitTimeNs) << "ns"
               << " T1=" << llvm::format("%.0f", t1) << "ns"
               << " depth_ok=" << (depthOk ? "1" : "0")
               << " | cx_fidelity=" << llvm::format("%.4f", fid);
            fpOp->emitRemark() << info;
        });
    }
};

} // namespace quantum
} // namespace catalyst
