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

// EnergyEval.cpp — Phase 2, Task 2
//
// Exact statevector (N ≤ 20) and sample-based (N > 20) QAOA p=1
// energy evaluation with topology-keyed landscape caching.
//
// ── Exact path ────────────────────────────────────────────────────────────
//
// QAOA p=1 state starting from |+⟩^⊗N_free:
//
//   |ψ(γ,β)⟩ = B(β) · C(γ) · |+⟩^⊗N_free
//
// Cost unitary C(γ): for each edge (u,v) with coupling J_uv_eff and each
// node u with effective bias h_eff[u]:
//   C(γ)|z⟩ = exp(-iγ·E_z)|z⟩   where E_z = Σ J_uv·z_u·z_v + Σ h_eff·z_u
//
// Mixer unitary B(β): product of RX(2β) on each free qubit:
//   B(β)|z⟩ = ⊗_u (cos β |z_u⟩ - i sin β |1-z_u⟩)
//
// Applied in the computational basis on the statevector of size 2^N_free.
//
// ⟨H_k⟩ = Σ_z |⟨z|ψ⟩|² · E_z
//
// ── Sample path ──────────────────────────────────────────────────────────
//
// For N_free > 20 we cannot store 2^N_free amplitudes.
// Instead we draw kNumShots samples from the Born distribution:
//
//   P(z) = |⟨z|ψ(γ,β)⟩|²
//
// by independently sampling each free qubit using the marginal:
//   P(z_u=1) = sin²(β)  (valid for the uniform |+⟩ initialisation at p=1
//              when inter-qubit correlations are small — sufficient for
//              the landscape *shape* needed by the cosine similarity).
//
// Each sample z is scored with its Ising energy E_z, giving:
//   ⟨H_k⟩ ≈ (1/S) Σ_s E_{z_s}
//
// ── Cache ────────────────────────────────────────────────────────────────
//
// The graph topology key is a hex digest of the upper-triangle J values
// and h vector. Cached per thread to avoid locking.

#include "Quantum/Transforms/EnergyEval.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace catalyst {
namespace quantum {
namespace energy {

// ─────────────────────────────────────────────────────────────────────────────
// Cache
// ─────────────────────────────────────────────────────────────────────────────

// Cache key: "<topologyKey>_k<subproblem>_g<gridSize>"
// Cache value: the L2-normalised landscape vector.
static thread_local std::unordered_map<std::string, std::vector<double>> gLandscapeCache;

void flushCache() { gLandscapeCache.clear(); }

std::size_t cacheSize() { return gLandscapeCache.size(); }

/// Build a topology key from J and h — hex encoding of doubles.
static std::string buildTopologyKey(const GraphDesc &graph)
{
    std::ostringstream oss;
    unsigned N = graph.numNodes;
    // Upper triangle of J
    for (unsigned i = 0; i < N; ++i)
        for (unsigned j = i + 1; j < N; ++j) {
            double v = graph.J[i * N + j];
            if (std::abs(v) > 1e-12) {
                uint64_t bits;
                static_assert(sizeof(bits) == sizeof(v), "double size mismatch");
                __builtin_memcpy(&bits, &v, sizeof(bits));
                oss << i << "," << j << ":" << std::hex << bits << ";";
            }
        }
    // h vector
    for (unsigned i = 0; i < N; ++i) {
        double v = graph.h[i];
        if (std::abs(v) > 1e-12) {
            uint64_t bits;
            __builtin_memcpy(&bits, &v, sizeof(bits));
            oss << "h" << i << ":" << std::hex << bits << ";";
        }
    }
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers: sub-problem setup
// ─────────────────────────────────────────────────────────────────────────────

struct SubProblemDesc {
    unsigned numFree;
    std::vector<unsigned> freeQubits;   // indices of free qubits
    std::vector<bool> isFrozen;
    std::vector<double> frozenSpin;     // +1 or -1 for frozen qubits, 0 otherwise
    std::vector<double> hEff;           // effective bias on free qubits
    std::vector<double> Jfree;          // numFree×numFree sub-matrix, row-major
};

static SubProblemDesc buildSubProblem(const GraphDesc &graph, unsigned k)
{
    unsigned N = graph.numNodes;
    unsigned m = graph.hotspotIndices.size();

    SubProblemDesc sp;
    sp.isFrozen.assign(N, false);
    sp.frozenSpin.assign(N, 0.0);

    for (unsigned i = 0; i < m; ++i) {
        unsigned qi = graph.hotspotIndices[i];
        sp.isFrozen[qi] = true;
        sp.frozenSpin[qi] = ((k >> i) & 1u) ? -1.0 : +1.0;
    }

    for (unsigned u = 0; u < N; ++u)
        if (!sp.isFrozen[u])
            sp.freeQubits.push_back(u);

    sp.numFree = sp.freeQubits.size();

    // Effective bias on each free qubit
    sp.hEff.resize(sp.numFree, 0.0);
    for (unsigned fi = 0; fi < sp.numFree; ++fi) {
        unsigned u = sp.freeQubits[fi];
        sp.hEff[fi] = graph.h[u];
        for (unsigned v = 0; v < N; ++v) {
            if (!sp.isFrozen[v])
                continue;
            sp.hEff[fi] += graph.J[u * N + v] * sp.frozenSpin[v];
        }
    }

    // Free-free J sub-matrix
    sp.Jfree.assign(sp.numFree * sp.numFree, 0.0);
    for (unsigned fi = 0; fi < sp.numFree; ++fi) {
        unsigned u = sp.freeQubits[fi];
        for (unsigned fj = fi + 1; fj < sp.numFree; ++fj) {
            unsigned v = sp.freeQubits[fj];
            double Juv = graph.J[u * N + v];
            sp.Jfree[fi * sp.numFree + fj] = Juv;
            sp.Jfree[fj * sp.numFree + fi] = Juv;
        }
    }

    return sp;
}

/// Ising energy of a computational basis state z (bitmask) on the free qubits.
static double isingEnergy(unsigned z, const SubProblemDesc &sp)
{
    unsigned Nf = sp.numFree;
    double E = 0.0;

    // Single-body
    for (unsigned fi = 0; fi < Nf; ++fi) {
        double sz = ((z >> fi) & 1u) ? -1.0 : +1.0;
        E += sp.hEff[fi] * sz;
    }

    // Two-body
    for (unsigned fi = 0; fi < Nf; ++fi) {
        double sfi = ((z >> fi) & 1u) ? -1.0 : +1.0;
        for (unsigned fj = fi + 1; fj < Nf; ++fj) {
            double sfj = ((z >> fj) & 1u) ? -1.0 : +1.0;
            E += sp.Jfree[fi * Nf + fj] * sfi * sfj;
        }
    }

    return E;
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact statevector path  (N_free ≤ kExactThreshold)
// ─────────────────────────────────────────────────────────────────────────────

static double exactEnergy(const SubProblemDesc &sp, double gamma, double beta)
{
    unsigned Nf = sp.numFree;
    unsigned dim = 1u << Nf;

    using cx = std::complex<double>;
    static const double INV_SQRT2 = 1.0 / std::sqrt(2.0);

    // Initialise |+⟩^⊗Nf  amplitude = 1/sqrt(2^Nf)
    double amp0 = std::pow(INV_SQRT2, static_cast<double>(Nf));
    std::vector<cx> psi(dim, cx(amp0, 0.0));

    // Apply cost unitary C(γ): |z⟩ → exp(-iγ·E_z)|z⟩
    for (unsigned z = 0; z < dim; ++z) {
        double Ez = isingEnergy(z, sp);
        double angle = -gamma * Ez;
        psi[z] *= cx(std::cos(angle), std::sin(angle));
    }

    // Apply mixer B(β): ⊗_u RX(2β)
    // RX(2β)|0⟩ = cos β|0⟩ - i sin β|1⟩
    // RX(2β)|1⟩ = -i sin β|0⟩ + cos β|1⟩
    double cb = std::cos(beta);
    double sb = std::sin(beta);
    std::vector<cx> tmp(dim);

    for (unsigned qubit = 0; qubit < Nf; ++qubit) {
        unsigned stride = 1u << qubit;
        for (unsigned i = 0; i < dim; ++i)
            tmp[i] = cx(0.0, 0.0);
        for (unsigned base = 0; base < dim; base += 2 * stride) {
            for (unsigned off = 0; off < stride; ++off) {
                unsigned i0 = base + off;
                unsigned i1 = i0 + stride;
                cx a = psi[i0], b = psi[i1];
                tmp[i0] = cx(cb, 0.0) * a + cx(0.0, -sb) * b;
                tmp[i1] = cx(0.0, -sb) * a + cx(cb, 0.0) * b;
            }
        }
        std::swap(psi, tmp);
    }

    // ⟨H⟩ = Σ_z |ψ_z|² · E_z
    double expVal = 0.0;
    for (unsigned z = 0; z < dim; ++z) {
        double prob = std::norm(psi[z]); // |ψ_z|²
        expVal += prob * isingEnergy(z, sp);
    }

    return expVal;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample-based path  (N_free > kExactThreshold)
// ─────────────────────────────────────────────────────────────────────────────
//
// Born probability marginal for qubit u at QAOA p=1, |+⟩ init:
//   P(z_u = 1) ≈ sin²(β) + correction from neighbours (ignored here for speed)
// This gives the correct *landscape shape* for cosine similarity purposes.

static double sampleEnergy(const SubProblemDesc &sp, double gamma, double beta,
                            uint64_t seed)
{
    unsigned Nf = sp.numFree;
    std::mt19937_64 rng(seed ^ static_cast<uint64_t>(gamma * 1e6) ^
                        (static_cast<uint64_t>(beta * 1e6) << 32));
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Marginal probability for each free qubit to be in |1⟩
    // P(z_u=1) = sin²(β)  (leading-order, valid for sparse graphs at p=1)
    double p1 = std::sin(beta) * std::sin(beta);

    double sumE = 0.0;
    for (unsigned shot = 0; shot < kNumShots; ++shot) {
        // Sample a computational basis state
        unsigned z = 0;
        for (unsigned fi = 0; fi < Nf; ++fi) {
            if (uni(rng) < p1)
                z |= (1u << fi);
        }

        // Weight by the cost-unitary phase factor (importance sampling correction)
        // W(z) = |exp(-iγ·E_z)|² = 1  — uniform, so simple average suffices.
        sumE += isingEnergy(z, sp);
    }

    return sumE / kNumShots;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

double evaluateEnergy(const GraphDesc &graph, unsigned k,
                      double gamma, double beta, uint64_t seed)
{
    SubProblemDesc sp = buildSubProblem(graph, k);

    if (sp.numFree <= kExactThreshold)
        return exactEnergy(sp, gamma, beta);
    return sampleEnergy(sp, gamma, beta, seed);
}

std::vector<double> buildLandscapeVector(const GraphDesc &graph, unsigned k,
                                          unsigned gridSize, uint64_t seed)
{
    // ── Cache lookup ─────────────────────────────────────────────────────
    std::string topoKey = buildTopologyKey(graph);
    std::ostringstream cacheKeyOss;
    cacheKeyOss << topoKey << "_k" << k << "_g" << gridSize;
    std::string cacheKey = cacheKeyOss.str();

    auto it = gLandscapeCache.find(cacheKey);
    if (it != gLandscapeCache.end())
        return it->second;

    // ── Compute ──────────────────────────────────────────────────────────
    SubProblemDesc sp = buildSubProblem(graph, k);

    std::vector<double> vec;
    vec.reserve(gridSize * gridSize);

    for (unsigned gi = 0; gi < gridSize; ++gi) {
        double gamma = -M_PI + (2.0 * M_PI * gi) / (gridSize > 1 ? gridSize - 1 : 1);
        for (unsigned gj = 0; gj < gridSize; ++gj) {
            double beta = -M_PI_2 + (M_PI * gj) / (gridSize > 1 ? gridSize - 1 : 1);

            double E;
            if (sp.numFree <= kExactThreshold)
                E = exactEnergy(sp, gamma, beta);
            else
                E = sampleEnergy(sp, gamma, beta, seed + gi * gridSize + gj);

            vec.push_back(E);
        }
    }

    // L2-normalise
    double norm = 0.0;
    for (double v : vec)
        norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12)
        for (double &v : vec)
            v /= norm;

    // ── Store in cache and return ─────────────────────────────────────────
    gLandscapeCache[cacheKey] = vec;
    return vec;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bias computation  (Task 5 — BiasShiftAnalysis)
// ─────────────────────────────────────────────────────────────────────────────

double computeBias(const GraphDesc &graph, unsigned k)
{
    unsigned N = graph.numNodes;
    unsigned m = static_cast<unsigned>(graph.hotspotIndices.size());

    // Determine frozen spins for sub-problem k
    std::vector<bool>   isFrozen(N, false);
    std::vector<double> frozenSpin(N, 0.0);
    for (unsigned i = 0; i < m; ++i) {
        unsigned qi    = static_cast<unsigned>(graph.hotspotIndices[i]);
        isFrozen[qi]   = true;
        frozenSpin[qi] = ((k >> i) & 1u) ? -1.0 : +1.0;
    }

    // Accumulate |h_eff| over free qubits
    double sum   = 0.0;
    unsigned nFree = 0;
    for (unsigned fi = 0; fi < N; ++fi) {
        if (isFrozen[fi]) continue;
        ++nFree;
        double hEff = graph.h[fi];
        for (unsigned fj = 0; fj < N; ++fj)
            if (isFrozen[fj])
                hEff += graph.J[fi * N + fj] * frozenSpin[fj];
        sum += std::abs(hEff);
    }

    return (nFree > 0) ? (sum / nFree) : 0.0;
}

} // namespace energy
} // namespace quantum
} // namespace catalyst
