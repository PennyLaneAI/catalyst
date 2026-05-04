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

// EnergyEval.h — Phase 2, Task 2
//
// Exact and sample-based QAOA p=1 expectation-value evaluation backend
// for the DO-QAOA landscape analysis passes.
//
// Two execution paths are selected automatically by numFreeQubits:
//   N ≤ exactThreshold (default 20): statevector-exact path
//   N  > exactThreshold             : sample-based path (512 shots)
//
// Results are cached by graph topology key so that repeated evaluations
// of the same graph structure (different (γ,β) points for the same
// sub-problem) reuse the cached unitary product.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace catalyst {
namespace quantum {
namespace energy {

// ─────────────────────────────────────────────────────────────────────────────
// Graph descriptor passed to the energy evaluator.
// ─────────────────────────────────────────────────────────────────────────────

struct GraphDesc {
    unsigned numNodes;                    // total qubit count N
    std::vector<double> J;                // N×N coupling matrix, row-major
    std::vector<double> h;                // N linear bias vector
    std::vector<int32_t> hotspotIndices;  // frozen qubit positions
};

// ─────────────────────────────────────────────────────────────────────────────
// Thresholds and constants
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of free qubits for the exact statevector path.
static constexpr unsigned kExactThreshold = 20;

/// Number of shots for the sample-based path.
static constexpr unsigned kNumShots = 512;

// ─────────────────────────────────────────────────────────────────────────────
// Cache control
// ─────────────────────────────────────────────────────────────────────────────

/// Flush the thread-local landscape cache. Call between unrelated graphs.
void flushCache();

/// Return the number of cached entries (for testing).
std::size_t cacheSize();

// ─────────────────────────────────────────────────────────────────────────────
// Primary API
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the QAOA p=1 expectation value E_k(γ, β) for sub-problem k.
///
/// sub-problem k: bit i of k → frozen qubit hotspotIndices[i] takes spin
///   +1 (bit=0) or -1 (bit=1).
///
/// Selects exact or sample-based path based on the number of free qubits.
/// Results for the same (graphKey, k) are cached across (γ,β) calls.
///
/// @param graph   Graph descriptor (J, h, hotspotIndices, numNodes).
/// @param k       Sub-problem index in [0, 2^m).
/// @param gamma   QAOA cost unitary angle γ.
/// @param beta    QAOA mixer unitary angle β.
/// @param seed    RNG seed for sample path (ignored for exact path).
/// @return        ⟨H_k⟩ under the QAOA p=1 state.
double evaluateEnergy(const GraphDesc &graph, unsigned k,
                      double gamma, double beta,
                      uint64_t seed = 42);

/// Build the full landscape vector for sub-problem k on a gridSize×gridSize
/// (γ,β) grid and L2-normalise it.
///
/// γ ∈ [−π, π],  β ∈ [−π/2, π/2] (standard QAOA ranges for p=1).
std::vector<double> buildLandscapeVector(const GraphDesc &graph, unsigned k,
                                         unsigned gridSize,
                                         uint64_t seed = 42);

/// Compute the effective linear-bias norm for sub-problem k:
///   B_k = (1 / N_free) × Σ_{free qubits i} |h_eff[i]|
/// where h_eff[i] = h[i] + Σ_{frozen j} J[i,j] × s_j.
///
/// Used by BiasShiftAnalysis to compute ΔB = |B_target − B_rep|.
double computeBias(const GraphDesc &graph, unsigned k);

} // namespace energy
} // namespace quantum
} // namespace catalyst
