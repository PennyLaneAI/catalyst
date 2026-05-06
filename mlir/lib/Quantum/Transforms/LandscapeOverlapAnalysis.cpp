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

// LandscapeOverlapAnalysis — Phase 2, Task 1 (wired to EnergyEval backend)
//
// For each quantum.freeze_partition op:
//   1. Extract J_ij from h_quad and h_lin from the op attributes.
//   2. Call energy::buildLandscapeVector() for each of the 2^m sub-problems.
//      The backend selects exact statevector (N_free ≤ 20) or sample-based
//      (N_free > 20, 512 shots) automatically.
//   3. Compute pairwise cosine similarity S_kl (Eq. 2.6).
//   4. Annotate the op with landscape_overlap_q (f64) and recommended_k (i32).

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
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

#define GEN_PASS_DECL_LANDSCAPEOVERLAPANALYSISPASS
#define GEN_PASS_DEF_LANDSCAPEOVERLAPANALYSISPASS
#include "Quantum/Transforms/Passes.h.inc"

// ─────────────────────────────────────────────────────────────────────────────
// Attribute extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the N×N J matrix from either DenseGraphAttr or SparseGraphAttr.
/// Returns empty vector on failure (attribute missing or unknown type).
static std::vector<double> extractJMatrix(Operation *op, unsigned &numNodes)
{
    auto hquadAttr = op->getAttr("h_quad");
    if (!hquadAttr)
        return {};

    if (auto dense = dyn_cast<DenseGraphAttr>(hquadAttr)) {
        numNodes = dense.getNumNodes();
        std::vector<double> J(numNodes * numNodes, 0.0);
        unsigned idx = 0;
        for (double v : dense.getWeights().getValues<double>())
            J[idx++] = v;
        return J;
    }

    if (auto sparse = dyn_cast<SparseGraphAttr>(hquadAttr)) {
        numNodes = sparse.getNumNodes();
        std::vector<double> J(numNodes * numNodes, 0.0);
        auto rows = sparse.getRowIndices();
        auto cols = sparse.getColIndices();
        unsigned e = 0;
        for (double v : sparse.getWeights().getValues<double>()) {
            unsigned r = rows[e], c = cols[e];
            J[r * numNodes + c] = v;
            J[c * numNodes + r] = v;
            ++e;
        }
        return J;
    }

    return {};
}

/// Extract linear bias vector h from h_lin attribute (tensor<N×f64>).
static std::vector<double> extractHLin(Operation *op, unsigned numNodes)
{
    std::vector<double> h(numNodes, 0.0);
    auto hlinAttr = op->getAttr("h_lin");
    if (!hlinAttr)
        return h;
    if (auto dea = dyn_cast<DenseElementsAttr>(hlinAttr)) {
        unsigned i = 0;
        for (double v : dea.getValues<double>())
            h[i++] = v;
    }
    return h;
}

/// Dot product of two equal-length L2-normalised vectors = cosine similarity.
static double cosineSimilarity(const std::vector<double> &a, const std::vector<double> &b)
{
    assert(a.size() == b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// K-means clustering helpers  (Task 4)
// ─────────────────────────────────────────────────────────────────────────────

struct KMeansResult {
    int32_t k;
    std::vector<int32_t> assignments; // one entry per sub-problem
    std::vector<std::vector<double>> centroids;
};

/// Squared L2 distance between two equal-length vectors.
static double vecDist2(const std::vector<double> &a, const std::vector<double> &b)
{
    double s = 0.0;
    for (std::size_t d = 0; d < a.size(); ++d) {
        double diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

/// Lloyd's K-means with K-means++ initialisation.
/// Returns {inertia, assignments, centroids}.
static std::tuple<double, std::vector<int32_t>, std::vector<std::vector<double>>>
runKMeans(const std::vector<std::vector<double>> &vecs, unsigned k, unsigned seed)
{
    unsigned N = static_cast<unsigned>(vecs.size());
    unsigned D = vecs.empty() ? 0u : static_cast<unsigned>(vecs[0].size());

    std::mt19937 rng(seed);

    // K-means++ init
    std::vector<std::vector<double>> centroids;
    centroids.reserve(k);
    {
        std::uniform_int_distribution<unsigned> uni(0, N - 1);
        centroids.push_back(vecs[uni(rng)]);
        for (unsigned c = 1; c < k; ++c) {
            std::vector<double> minD(N, std::numeric_limits<double>::max());
            for (unsigned i = 0; i < N; ++i)
                for (const auto &cen : centroids)
                    minD[i] = std::min(minD[i], vecDist2(vecs[i], cen));
            std::discrete_distribution<unsigned> w(minD.begin(), minD.end());
            centroids.push_back(vecs[w(rng)]);
        }
    }

    std::vector<int32_t> asgn(N, 0);

    for (unsigned iter = 0; iter < 100; ++iter) {
        bool changed = false;

        // Assignment step
        for (unsigned i = 0; i < N; ++i) {
            double best = std::numeric_limits<double>::max();
            int32_t bestC = 0;
            for (unsigned c = 0; c < k; ++c) {
                double d = vecDist2(vecs[i], centroids[c]);
                if (d < best) {
                    best = d;
                    bestC = static_cast<int32_t>(c);
                }
            }
            if (bestC != asgn[i]) {
                asgn[i] = bestC;
                changed = true;
            }
        }

        if (!changed)
            break;

        // Update step: recompute centroids
        std::vector<std::vector<double>> next(k, std::vector<double>(D, 0.0));
        std::vector<unsigned> cnt(k, 0u);
        for (unsigned i = 0; i < N; ++i) {
            unsigned c = static_cast<unsigned>(asgn[i]);
            for (unsigned d = 0; d < D; ++d)
                next[c][d] += vecs[i][d];
            ++cnt[c];
        }
        for (unsigned c = 0; c < k; ++c)
            if (cnt[c] > 0)
                for (unsigned d = 0; d < D; ++d)
                    next[c][d] /= cnt[c];
            else
                next[c] = centroids[c]; // keep old centroid if cluster empty
        centroids = std::move(next);
    }

    double inertia = 0.0;
    for (unsigned i = 0; i < N; ++i)
        inertia += vecDist2(vecs[i], centroids[static_cast<unsigned>(asgn[i])]);

    return {inertia, asgn, centroids};
}

/// Run K-means for K=1..maxK, select K via elbow:
/// first K where cumulative variance explained ≥ 90%.
static KMeansResult kmeansElbow(const std::vector<std::vector<double>> &vecs, unsigned maxK,
                                unsigned seed = 42)
{
    unsigned N = static_cast<unsigned>(vecs.size());
    if (N <= 1 || maxK <= 1)
        return {1, std::vector<int32_t>(N, 0), {}};

    maxK = std::min(maxK, N);

    std::vector<double> inertias(maxK + 1, 0.0);
    std::vector<std::vector<int32_t>> allAsgn(maxK + 1);
    std::vector<std::vector<std::vector<double>>> allCen(maxK + 1);

    for (unsigned k = 1; k <= maxK; ++k) {
        auto [inertia, asgn, cen] = runKMeans(vecs, k, seed);
        inertias[k] = inertia;
        allAsgn[k] = std::move(asgn);
        allCen[k] = std::move(cen);
    }

    // Elbow: first K where (I_1 - I_K) / (I_1 - I_max) >= 0.9
    double totalVar = inertias[1] - inertias[maxK];
    if (totalVar < 1e-10) // all landscapes identical
        return {1, allAsgn[1], allCen[1]};

    int32_t bestK = static_cast<int32_t>(maxK);
    for (unsigned k = 1; k <= maxK; ++k) {
        double cumVar = (inertias[1] - inertias[k]) / totalVar;
        if (cumVar >= 0.9) {
            bestK = static_cast<int32_t>(k);
            break;
        }
    }

    return {bestK, allAsgn[bestK], allCen[bestK]};
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase-transition detector helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the graph diameter via BFS from every node.
/// An edge (u,v) exists when |J[u*N+v]| > 1e-12.
/// Disconnected node pairs are treated as distance N-1 (conservative bound).
/// Returns 0 for N ≤ 1 (trivially connected).
static unsigned computeGraphDiameter(const std::vector<double> &J, unsigned N)
{
    if (N <= 1)
        return 0;

    unsigned diameter = 0;
    std::vector<int> dist(N);

    for (unsigned src = 0; src < N; ++src) {
        std::fill(dist.begin(), dist.end(), -1);
        dist[src] = 0;

        std::queue<unsigned> q;
        q.push(src);

        while (!q.empty()) {
            unsigned u = q.front();
            q.pop();
            for (unsigned v = 0; v < N; ++v) {
                if (v != u && dist[v] < 0 && std::abs(J[u * N + v]) > 1e-12) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        // Eccentricity of src: max dist, unreachable nodes → N-1
        for (unsigned i = 0; i < N; ++i) {
            unsigned d = (dist[i] < 0) ? (N - 1) : static_cast<unsigned>(dist[i]);
            if (d > diameter)
                diameter = d;
        }
    }

    return diameter;
}

/// Estimate the effective phase-transition parameter s from the graph diameter.
///
/// Heuristic (Sang et al.):  s_eff = 2 / (1 + diameter)
///   diameter = 1  (complete graph)  → s_eff = 1.0   (concentrated regime)
///   diameter = 2  (small-world)     → s_eff = 0.667  (above sc = 0.6)
///   diameter = 3  (sparse)          → s_eff = 0.5    (below sc → warn)
///   diameter ≥ 4  (path-like)       → s_eff ≤ 0.4    (fragmented)
static double estimateSEff(unsigned diameter)
{
    return 2.0 / (1.0 + static_cast<double>(diameter));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct LandscapeOverlapAnalysisPass
    : impl::LandscapeOverlapAnalysisPassBase<LandscapeOverlapAnalysisPass> {
    using LandscapeOverlapAnalysisPassBase::LandscapeOverlapAnalysisPassBase;

    void runOnOperation() final
    {
        func::FuncOp func = getOperation();
        MLIRContext *ctx = &getContext();
        Builder builder(ctx);

        // Flush the energy eval cache at the start of each function —
        // different functions may have different graphs.
        energy::flushCache();

        func.walk([&](FreezePartitionOp op) {
            // ── 1. Extract graph data ─────────────────────────────────────
            unsigned numNodes = 0;
            std::vector<double> J = extractJMatrix(op, numNodes);
            if (J.empty()) {
                op->emitWarning() << "doqaoa-landscape-overlap: missing h_quad attribute, skipping";
                return;
            }
            std::vector<double> h = extractHLin(op, numNodes);

            // ── 2. Get hotspot indices ────────────────────────────────────
            auto hotspotAttr = op.getHotspotIndices();
            std::vector<int32_t> hotspotIndices(hotspotAttr.begin(), hotspotAttr.end());
            unsigned m = static_cast<unsigned>(hotspotIndices.size());
            unsigned numSubProblems = 1u << m;

            if (m > 10) {
                op->emitWarning() << "doqaoa-landscape-overlap: m=" << m
                                  << " > 10, skipping (too expensive for analysis pass)";
                return;
            }

            // ── 3. Build GraphDesc and landscape vectors via EnergyEval ──
            energy::GraphDesc graph;
            graph.numNodes = numNodes;
            graph.J = J;
            graph.h = h;
            graph.hotspotIndices = hotspotIndices;

            std::vector<std::vector<double>> landscapes(numSubProblems);
            for (unsigned k = 0; k < numSubProblems; ++k)
                landscapes[k] = energy::buildLandscapeVector(graph, k, gridSize);

            // ── 4. Pairwise cosine similarity S_kl (Eq. 2.6) ─────────────
            double sumSim = 0.0;
            unsigned numPairs = 0;
            for (unsigned k = 0; k < numSubProblems; ++k) {
                for (unsigned l = k + 1; l < numSubProblems; ++l) {
                    sumSim += cosineSimilarity(landscapes[k], landscapes[l]);
                    ++numPairs;
                }
            }

            double q = (numPairs > 0) ? (sumSim / numPairs) : 1.0;

            // ── 5. Recommend K (threshold) ────────────────────────────────
            int32_t recK = (q >= overlapThreshold) ? 1 : static_cast<int32_t>(numSubProblems);

            // ── 5b. K-means + elbow (Task 4) ─────────────────────────────
            // For concentrated regime (q ≥ threshold) K=1 is already known;
            // skip K-means for efficiency.  For fragmented regime, run elbow
            // to find the finest natural grouping K ∈ [1, 2^m].
            int32_t clusterK;
            std::vector<int32_t> clusterAssignments;

            if (q >= overlapThreshold || numSubProblems <= 1) {
                clusterK = 1;
                clusterAssignments.assign(numSubProblems, 0);
            }
            else {
                KMeansResult km = kmeansElbow(landscapes, numSubProblems);
                clusterK = km.k;
                clusterAssignments = std::move(km.assignments);
            }

            // ── 6. Phase-transition detector ──────────────────────────────
            unsigned diameter = computeGraphDiameter(J, numNodes);
            double sEff = estimateSEff(diameter);
            double scVal = scThreshold; // extract from Option<double> before format

            if (sEff < scVal) {
                llvm::SmallString<128> msg;
                llvm::raw_svector_ostream ss(msg);
                ss << llvm::format("doqaoa-landscape-overlap: fragmented landscape regime"
                                   " (s_eff=%.3f < sc=%.3f, diameter=%u);"
                                   " DO-QAOA parameter transfer may not achieve K=1",
                                   sEff, scVal, diameter);
                op->emitWarning() << msg;
            }

            // ── 7. Annotate op ────────────────────────────────────────────
            op->setAttr("landscape_overlap_q", builder.getF64FloatAttr(q));
            op->setAttr("recommended_k", builder.getI32IntegerAttr(recK));
            op->setAttr("s_eff", builder.getF64FloatAttr(sEff));
            op->setAttr("cluster_k", builder.getI32IntegerAttr(clusterK));
            op->setAttr("cluster_assignments", DenseI32ArrayAttr::get(ctx, clusterAssignments));
        });
    }
};

} // namespace quantum
} // namespace catalyst
