// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// A small, exact tensor-network engine.
//
// This header is deliberately independent of Catalyst: it knows nothing about
// QuantumDevice, so it can be unit-tested on its own. It provides
//
//   * Tensor            - dense tensor with named (integer-labelled) indices
//   * contractPair      - exact pairwise contraction over shared indices
//   * contractNetwork   - full contraction using a greedy ordering heuristic
//
// Conventions
// -----------
// * Every index carries an int64_t label. A label appearing in exactly one
//   tensor is an OPEN index; a label appearing in exactly two tensors is a BOND
//   and gets summed over during contraction. A label must never appear three or
//   more times.
// * Tensor data is row-major: the LAST index varies fastest.
//
// There is no truncation anywhere in this file. Contraction is exact; the only
// approximation-free knob is the *order* in which pairs are contracted, which
// changes cost but never the result.

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Exception.hpp"

namespace ExactTN {

using cd = std::complex<double>;

// Labels at or above this value are reserved for the "bra" copy of a network
// (see ExactTNDevice::expvalSandwich). Fresh labels must stay below it.
constexpr int64_t kPrimeOffset = 1LL << 40;

/**
 * @brief A dense tensor with integer-labelled indices.
 */
struct Tensor {
    std::vector<int64_t> idx; ///< index labels, one per rank
    std::vector<size_t> dim;  ///< dimension of each index
    std::vector<cd> data;     ///< row-major payload, size == prod(dim)

    Tensor() = default;

    Tensor(std::vector<int64_t> indices, std::vector<size_t> dims, std::vector<cd> values)
        : idx(std::move(indices)), dim(std::move(dims)), data(std::move(values)) {
        RT_FAIL_IF(idx.size() != dim.size(), "TensorNetwork: idx/dim rank mismatch");
        RT_FAIL_IF(data.size() != numel(), "TensorNetwork: data size does not match dims");
    }

    [[nodiscard]] auto rank() const -> size_t { return idx.size(); }

    /// Number of elements implied by `dim` (1 for a scalar).
    [[nodiscard]] auto numel() const -> size_t {
        size_t n = 1;
        for (size_t d : dim) {
            n *= d;
        }
        return n;
    }

    /// Row-major strides: stride[k] = prod(dim[k+1 ...]).
    [[nodiscard]] auto strides() const -> std::vector<size_t> {
        std::vector<size_t> s(dim.size(), 1);
        for (size_t k = dim.size(); k-- > 0;) {
            s[k] = (k + 1 == dim.size()) ? 1 : s[k + 1] * dim[k + 1];
        }
        return s;
    }

    /// Position of a label, or npos.
    [[nodiscard]] auto find(int64_t label) const -> size_t {
        for (size_t k = 0; k < idx.size(); k++) {
            if (idx[k] == label) {
                return k;
            }
        }
        return std::numeric_limits<size_t>::max();
    }

    [[nodiscard]] auto has(int64_t label) const -> bool {
        return find(label) != std::numeric_limits<size_t>::max();
    }
};

/**
 * @brief Enumerate all mixed-radix offsets for a subset of a tensor's indices.
 *
 * Given the strides and dimensions of some subset of positions, returns a table
 * of length prod(dims) whose entry j is the flat offset contributed by the j-th
 * combination (first entry varying slowest, i.e. row-major).
 */
inline auto offsetTable(const std::vector<size_t> &strides, const std::vector<size_t> &dims)
    -> std::vector<size_t> {
    size_t total = 1;
    for (size_t d : dims) {
        total *= d;
    }
    std::vector<size_t> table(total, 0);
    // Build incrementally: digit k has place value prod(dims[k+1...]).
    for (size_t j = 0; j < total; j++) {
        size_t rest = j;
        size_t off = 0;
        for (size_t k = dims.size(); k-- > 0;) {
            const size_t digit = rest % dims[k];
            rest /= dims[k];
            off += digit * strides[k];
        }
        table[j] = off;
    }
    return table;
}

/**
 * @brief Exact contraction of two tensors over their shared index labels.
 *
 * Output index order is: A's unshared indices, then B's unshared indices.
 * Cost is O(prod(out) * prod(shared)).
 */
inline auto contractPair(const Tensor &A, const Tensor &B) -> Tensor {
    const auto sA = A.strides();
    const auto sB = B.strides();

    std::vector<size_t> posSharedA, posSharedB, posOutA, posOutB;
    for (size_t k = 0; k < A.rank(); k++) {
        if (B.has(A.idx[k])) {
            posSharedA.push_back(k);
        } else {
            posOutA.push_back(k);
        }
    }
    // Shared order is fixed by A so that the two offset tables stay aligned.
    for (size_t k : posSharedA) {
        const size_t p = B.find(A.idx[k]);
        RT_FAIL_IF(B.dim[p] != A.dim[k], "TensorNetwork: bond dimension mismatch");
        posSharedB.push_back(p);
    }
    for (size_t k = 0; k < B.rank(); k++) {
        if (!A.has(B.idx[k])) {
            posOutB.push_back(k);
        }
    }

    auto gather = [](const std::vector<size_t> &src, const std::vector<size_t> &pos) {
        std::vector<size_t> out;
        out.reserve(pos.size());
        for (size_t p : pos) {
            out.push_back(src[p]);
        }
        return out;
    };

    const auto dimsOutA = gather(A.dim, posOutA);
    const auto dimsOutB = gather(B.dim, posOutB);
    const auto dimsShared = gather(A.dim, posSharedA);

    const auto offOutA = offsetTable(gather(sA, posOutA), dimsOutA);
    const auto offOutB = offsetTable(gather(sB, posOutB), dimsOutB);
    const auto offShA = offsetTable(gather(sA, posSharedA), dimsShared);
    const auto offShB = offsetTable(gather(sB, posSharedB), dimsShared);

    std::vector<int64_t> outIdx;
    std::vector<size_t> outDim;
    for (size_t p : posOutA) {
        outIdx.push_back(A.idx[p]);
        outDim.push_back(A.dim[p]);
    }
    for (size_t p : posOutB) {
        outIdx.push_back(B.idx[p]);
        outDim.push_back(B.dim[p]);
    }

    const size_t nA = offOutA.size();
    const size_t nB = offOutB.size();
    const size_t nS = offShA.size();

    std::vector<cd> out(nA * nB, cd{0.0, 0.0});
    for (size_t a = 0; a < nA; a++) {
        for (size_t b = 0; b < nB; b++) {
            cd acc{0.0, 0.0};
            for (size_t s = 0; s < nS; s++) {
                acc += A.data[offOutA[a] + offShA[s]] * B.data[offOutB[b] + offShB[s]];
            }
            out[a * nB + b] = acc;
        }
    }
    return Tensor(std::move(outIdx), std::move(outDim), std::move(out));
}

/**
 * @brief Contract an entire network down to a single tensor.
 *
 * Uses a greedy heuristic: repeatedly contract the connected pair whose result
 * is smallest, breaking ties by contraction cost. Disconnected components are
 * only joined (as outer products) once nothing shares an index, which keeps
 * intermediates as small as this heuristic can manage.
 *
 * The ordering affects speed and peak memory but never the numerical result.
 *
 * @param nodes            network to contract (consumed)
 * @param maxIntermediate  abort if an intermediate would exceed this many
 *                         elements; 0 disables the guard
 */
inline auto contractNetwork(std::vector<Tensor> nodes, size_t maxIntermediate = 0) -> Tensor {
    RT_FAIL_IF(nodes.empty(), "TensorNetwork: cannot contract an empty network");

    while (nodes.size() > 1) {
        size_t bestI = 0, bestJ = 1;
        bool foundConnected = false;
        double bestResultSize = std::numeric_limits<double>::max();
        double bestCost = std::numeric_limits<double>::max();

        for (size_t i = 0; i < nodes.size(); i++) {
            for (size_t j = i + 1; j < nodes.size(); j++) {
                // Result size and cost of contracting (i, j), in elements.
                double resultSize = 1.0;
                double sharedSize = 1.0;
                for (size_t k = 0; k < nodes[i].rank(); k++) {
                    if (nodes[j].has(nodes[i].idx[k])) {
                        sharedSize *= static_cast<double>(nodes[i].dim[k]);
                    } else {
                        resultSize *= static_cast<double>(nodes[i].dim[k]);
                    }
                }
                for (size_t k = 0; k < nodes[j].rank(); k++) {
                    if (!nodes[i].has(nodes[j].idx[k])) {
                        resultSize *= static_cast<double>(nodes[j].dim[k]);
                    }
                }
                const bool connected = sharedSize > 1.0;
                const double cost = resultSize * sharedSize;

                // Prefer any connected pair over any disconnected pair.
                if (connected && !foundConnected) {
                    foundConnected = true;
                    bestResultSize = resultSize;
                    bestCost = cost;
                    bestI = i;
                    bestJ = j;
                    continue;
                }
                if (connected != foundConnected) {
                    continue; // never downgrade from connected to disconnected
                }
                if (resultSize < bestResultSize ||
                    (resultSize == bestResultSize && cost < bestCost)) {
                    bestResultSize = resultSize;
                    bestCost = cost;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (maxIntermediate != 0 && bestResultSize > static_cast<double>(maxIntermediate)) {
            RT_FAIL(("TensorNetwork: contraction needs an intermediate of ~" +
                     std::to_string(static_cast<unsigned long long>(bestResultSize)) +
                     " elements, exceeding the configured limit of " +
                     std::to_string(maxIntermediate) +
                     ". Raise max_intermediate_log2 or simplify the circuit.")
                        .c_str());
        }

        Tensor merged = contractPair(nodes[bestI], nodes[bestJ]);
        // Erase the higher index first so the lower one stays valid.
        nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(bestJ));
        nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(bestI));
        nodes.push_back(std::move(merged));
    }
    return nodes.front();
}

/**
 * @brief Permute a tensor so its indices appear in the requested label order.
 */
inline auto permuteTo(const Tensor &T, const std::vector<int64_t> &order) -> Tensor {
    RT_FAIL_IF(order.size() != T.rank(), "TensorNetwork: permutation rank mismatch");
    if (T.idx == order) {
        return T;
    }

    const auto s = T.strides();
    std::vector<size_t> newStrides;
    std::vector<size_t> newDims;
    newStrides.reserve(order.size());
    newDims.reserve(order.size());
    for (int64_t label : order) {
        const size_t p = T.find(label);
        RT_FAIL_IF(p == std::numeric_limits<size_t>::max(),
                   "TensorNetwork: permutation references an unknown index");
        newStrides.push_back(s[p]);
        newDims.push_back(T.dim[p]);
    }

    const auto table = offsetTable(newStrides, newDims);
    std::vector<cd> out(table.size());
    for (size_t j = 0; j < table.size(); j++) {
        out[j] = T.data[table[j]];
    }
    return Tensor(order, newDims, std::move(out));
}

} // namespace ExactTN
