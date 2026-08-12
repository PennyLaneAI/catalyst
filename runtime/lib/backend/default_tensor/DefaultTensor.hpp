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

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataView.hpp"
#include "Exception.hpp"
#include "QuantumDevice.hpp"
#include "TensorNetwork.hpp"
#include "Types.h"
#include "Utils.hpp" // Catalyst::Runtime::parse_kwargs (in-tree only)

namespace Catalyst::Runtime::Devices {

using ExactTN::cd;
using ExactTN::Tensor;

struct ObsRecord {
    ObsType kind{Basic};
    ObsId basic_id{Identity};
    std::vector<QubitIdType> wires;
    std::vector<cd> matrix;       ///< Hermitian only
    std::vector<ObsIdType> terms; ///< TensorProd / Hamiltonian
    std::vector<double> coeffs;   ///< Hamiltonian only
};

struct DefaultTensor final : public QuantumDevice {
    explicit DefaultTensor(const std::string &kwargs = "{}")
    {
        device_kwargs_ = Catalyst::Runtime::parse_kwargs(kwargs);
        if (auto it = device_kwargs_.find("max_intermediate_log2");
            it != device_kwargs_.end()) {
            const int bits = std::stoi(it->second);
            RT_FAIL_IF(bits < 1 || bits > 62,
                       "DefaultTensor: max_intermediate_log2 must be in [1, 62]");
            max_intermediate_ = size_t{1} << bits;
        }
    }
    ~DefaultTensor() override = default;

    auto AllocateQubit() -> QubitIdType override
    {
        const QubitIdType id = next_qubit_id_++;
        const int64_t leg = freshLabel();
        // A brand new qubit is the rank-1 tensor |0> = (1, 0).
        nodes_.push_back(Tensor({leg}, {2}, {cd{1.0, 0.0}, cd{0.0, 0.0}}));
        frontier_[id] = leg;
        order_.push_back(id);
        return id;
    }

    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override
    {
        std::vector<QubitIdType> ids;
        ids.reserve(num_qubits);
        for (size_t i = 0; i < num_qubits; i++) {
            ids.push_back(AllocateQubit());
        }
        return ids;
    }

    /**
     * @brief Release one qubit.
     *
     * The qubit may still be entangled with the ones that survive. Simply
     * dropping its open leg would leave the network representing an unnormalised
     * (and generally wrong) state on the remainder, so instead we do the
     * physically meaningful thing: measure it, which projects and renormalises,
     * then cap the resulting product leg to remove it from the network.
     *
     * This satisfies the interface contract -- a released qubit can no longer
     * influence any later result -- and on an unentangled qubit it is exactly a
     * no-op on the rest of the state. The observable consequence, matching an
     * unread measurement, is that releasing an entangled qubit collapses the
     * survivors into one branch rather than leaving them in a superposition.
     *
     * Cost note: this is the one genuinely expensive release path, because
     * measuring requires knowing a probability, which requires contracting.
     * Bulk release of every live qubit (i.e. program teardown) is special-cased
     * in ReleaseQubits and costs nothing.
     */
    void ReleaseQubit(QubitIdType qubit) override
    {
        auto it = frontier_.find(qubit);
        RT_FAIL_IF(it == frontier_.end(),
                   "DefaultTensor: releasing an unknown or already-released qubit");

        if (frontier_.size() == 1) {
            // Last qubit standing: nothing can observe it afterwards.
            teardown();
            return;
        }

        // Collapse it so no correlation with the remaining qubits survives.
        // measureImpl contracts the network down to a single node as a side
        // effect, so no separate compress() is needed here.
        const bool one = measureImpl(qubit, std::nullopt);

        // Cap the (now product-state) leg with <0| or <1|, removing it from the
        // network while keeping the remaining state exactly normalised.
        const int64_t leg = frontier_[qubit];
        nodes_.push_back(Tensor({leg}, {2},
                                {one ? cd{0.0, 0.0} : cd{1.0, 0.0},
                                 one ? cd{1.0, 0.0} : cd{0.0, 0.0}}));

        frontier_.erase(qubit);
        order_.erase(std::remove(order_.begin(), order_.end(), qubit), order_.end());
    }

    void ReleaseQubits(const std::vector<QubitIdType> &qubits) override
    {
        // Fast path: releasing every live qubit is program teardown. Nothing can
        // observe the state afterwards, so skip the contraction entirely. This
        // matters -- the runtime always ends a program this way, and collapsing
        // qubit-by-qubit would make cleanup cost as much as a full simulation.
        bool releasesEverything = qubits.size() >= frontier_.size();
        if (releasesEverything) {
            for (const auto &[q, _] : frontier_) {
                if (std::find(qubits.begin(), qubits.end(), q) == qubits.end()) {
                    releasesEverything = false;
                    break;
                }
            }
        }
        if (releasesEverything) {
            for (QubitIdType q : qubits) {
                RT_FAIL_IF(frontier_.find(q) == frontier_.end(),
                           "DefaultTensor: releasing an unknown or already-released qubit");
            }
            teardown();
            return;
        }

        for (QubitIdType q : qubits) {
            ReleaseQubit(q);
        }
    }

    auto GetNumQubits() const -> size_t override { return frontier_.size(); }

    void SetDeviceShots(size_t shots) override { shots_ = shots; }
    auto GetDeviceShots() const -> size_t override { return shots_; }
    void SetDevicePRNG(std::mt19937 *gen) override { gen_ = gen; }

    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::string> &) override
    {
        RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
                   "DefaultTensor: control wires/values length mismatch");

        if (name == "CNOT" || name == "CZ") {
            RT_FAIL_IF(wires.size() != 2, "DefaultTensor: CNOT/CZ need exactly 2 wires");
            // CNOT/CZ are just controlled-X / controlled-Z; fold the explicit
            // control wire in with any extra controls Catalyst supplied.
            std::vector<QubitIdType> ctrl{wires[0]};
            std::vector<bool> vals{true};
            ctrl.insert(ctrl.end(), controlled_wires.begin(), controlled_wires.end());
            vals.insert(vals.end(), controlled_values.begin(), controlled_values.end());
            const auto m = (name == "CNOT") ? matX() : matZ();
            applyControlled(m, wires[1], ctrl, vals);
            return;
        }

        RT_FAIL_IF(wires.size() != 1,
                   ("DefaultTensor: '" + name + "' is not a supported 1-qubit gate").c_str());
        auto m = singleQubitMatrix(name, params);
        if (inverse) {
            m = adjoint(m);
        }
        applyControlled(m, wires[0], controlled_wires, controlled_values);
    }

    auto Observable(ObsId id, const std::vector<cd> &matrix,
                    const std::vector<QubitIdType> &wires) -> ObsIdType override
    {
        if (id == Hermitian) {
            RT_FAIL_IF(wires.size() != 1, "DefaultTensor: Hermitian limited to 1 wire");
            RT_FAIL_IF(matrix.size() != 4, "DefaultTensor: Hermitian expects a 2x2 matrix");
        }
        else {
            RT_FAIL_IF(wires.size() != 1, "DefaultTensor: named observable acts on 1 wire");
        }
        ObsRecord rec;
        rec.kind = Basic;
        rec.basic_id = id;
        rec.wires = wires;
        rec.matrix = matrix;
        obs_.push_back(std::move(rec));
        return static_cast<ObsIdType>(obs_.size() - 1);
    }

    auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType override
    {
        ObsRecord rec;
        rec.kind = TensorProd;
        rec.terms = obs;
        obs_.push_back(std::move(rec));
        return static_cast<ObsIdType>(obs_.size() - 1);
    }

    auto HamiltonianObservable(const std::vector<double> &coeffs,
                               const std::vector<ObsIdType> &obs) -> ObsIdType override
    {
        RT_FAIL_IF(coeffs.size() != obs.size(),
                   "DefaultTensor: coefficient/observable length mismatch");
        ObsRecord rec;
        rec.kind = Hamiltonian;
        rec.coeffs = coeffs;
        rec.terms = obs;
        obs_.push_back(std::move(rec));
        return static_cast<ObsIdType>(obs_.size() - 1);
    }

    auto Expval(ObsIdType key) -> double override
    {
        // <psi|O|psi> as a single closed network: no statevector is formed.
        return expvalSandwich(key, /*squared=*/false);
    }

    auto Var(ObsIdType key) -> double override
    {
        const double ev = expvalSandwich(key, false);
        const double ev2 = expvalSandwich(key, true);
        return ev2 - ev * ev;
    }

    void State(DataView<cd, 1> &state) override
    {
        const auto amps = amplitudes();
        RT_FAIL_IF(state.size() != amps.size(), "DefaultTensor: state buffer size mismatch");
        size_t i = 0;
        for (auto &out : state) {
            out = amps[i++];
        }
    }

    void Probs(DataView<double, 1> &probs) override { PartialProbs(probs, activeWires()); }

    void PartialProbs(DataView<double, 1> &probs,
                      const std::vector<QubitIdType> &wires) override
    {
        const auto p = marginal(wires);
        RT_FAIL_IF(probs.size() != p.size(), "DefaultTensor: probs buffer size mismatch");
        size_t i = 0;
        for (auto &out : probs) {
            out = p[i++];
        }
    }

    void Sample(DataView<double, 2> &samples) override { PartialSample(samples, activeWires()); }

    void PartialSample(DataView<double, 2> &samples,
                       const std::vector<QubitIdType> &wires) override
    {
        RT_FAIL_IF(shots_ == 0, "DefaultTensor: sampling requires shots > 0");
        const auto p = marginal(wires);
        auto it = samples.begin();
        for (size_t s = 0; s < shots_; s++) {
            const size_t outcome = draw(p);
            for (size_t w = 0; w < wires.size(); w++) {
                RT_FAIL_IF(it == samples.end(), "DefaultTensor: samples buffer too small");
                const size_t shift = wires.size() - 1 - w; // wires[0] is the MSB
                *it = static_cast<double>((outcome >> shift) & 1ULL);
                ++it;
            }
        }
    }

    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts) override
    {
        PartialCounts(eigvals, counts, activeWires());
    }

    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires) override
    {
        RT_FAIL_IF(shots_ == 0, "DefaultTensor: counts requires shots > 0");
        const size_t dim = size_t{1} << wires.size();
        RT_FAIL_IF(eigvals.size() != dim || counts.size() != dim,
                   "DefaultTensor: counts buffers must have 2^n entries");

        const auto p = marginal(wires);
        std::vector<int64_t> tally(dim, 0);
        for (size_t s = 0; s < shots_; s++) {
            tally[draw(p)]++;
        }
        // Every entry must be written or the caller reads uninitialised memory.
        size_t i = 0;
        for (auto &e : eigvals) {
            e = static_cast<double>(i++);
        }
        i = 0;
        for (auto &c : counts) {
            c = tally[i++];
        }
    }

    /// Mid-circuit measurement: projects the network and renormalises.
    auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result override
    {
        const bool outcome = measureImpl(wire, postselect);
        // The runtime never frees this pointer, so return a static constant
        // rather than malloc'ing (which would leak once per measurement).
        static constexpr bool kTrue = true;
        static constexpr bool kFalse = false;
        return const_cast<Result>(outcome ? &kTrue : &kFalse);
    }

  private:
    std::vector<Tensor> nodes_;                 ///< the tensor network
    std::map<QubitIdType, int64_t> frontier_;   ///< qubit -> its open index label
    std::vector<QubitIdType> order_;            ///< allocation order (wire ordering)
    std::vector<ObsRecord> obs_;
    QubitIdType next_qubit_id_{0};              ///< never reused
    int64_t next_label_{0};
    size_t shots_{0};
    bool last_outcome_{false};
    size_t max_intermediate_{size_t{1} << 27};  ///< ~134M elements (~2 GiB complex)
    std::mt19937 *gen_{nullptr};
    std::mt19937 fallback_gen_{std::random_device{}()};
    std::unordered_map<std::string, std::string> device_kwargs_;

    auto freshLabel() -> int64_t
    {
        RT_FAIL_IF(next_label_ >= ExactTN::kPrimeOffset,
                   "DefaultTensor: exhausted index labels");
        return next_label_++;
    }

    /// Drop the whole network. Qubit IDs still never get reused.
    void teardown()
    {
        nodes_.clear();
        frontier_.clear();
        order_.clear();
        obs_.clear();
        next_label_ = 0;
    }

    /// Currently allocated qubits, in allocation order (wire 0 first).
    auto activeWires() const -> std::vector<QubitIdType> { return order_; }

    auto uniform() -> double
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return gen_ ? dist(*gen_) : dist(fallback_gen_);
    }

    auto draw(const std::vector<double> &p) -> size_t
    {
        const double r = uniform();
        double acc = 0.0;
        for (size_t i = 0; i < p.size(); i++) {
            acc += p[i];
            if (r < acc) {
                return i;
            }
        }
        return p.size() - 1; // guard against floating-point rounding
    }

    static auto matX() -> std::array<cd, 4> { return {0, 1, 1, 0}; }
    static auto matZ() -> std::array<cd, 4> { return {1, 0, 0, -1}; }

    static auto adjoint(const std::array<cd, 4> &m) -> std::array<cd, 4>
    {
        return {std::conj(m[0]), std::conj(m[2]), std::conj(m[1]), std::conj(m[3])};
    }

    static auto singleQubitMatrix(const std::string &name,
                                  const std::vector<double> &p) -> std::array<cd, 4>
    {
        const cd I{0.0, 1.0};
        const double s = M_SQRT1_2;
        if (name == "Identity") {
            return {1, 0, 0, 1};
        }
        if (name == "PauliX") {
            return matX();
        }
        if (name == "PauliY") {
            return {0, -I, I, 0};
        }
        if (name == "PauliZ") {
            return matZ();
        }
        if (name == "Hadamard") {
            return {s, s, s, -s};
        }
        RT_FAIL_IF(p.empty(),
                   ("DefaultTensor: '" + name + "' requires a rotation parameter").c_str());
        const double t = p[0];
        if (name == "RX") {
            return {std::cos(t / 2), -I * std::sin(t / 2), -I * std::sin(t / 2),
                    std::cos(t / 2)};
        }
        if (name == "RY") {
            return {std::cos(t / 2), -std::sin(t / 2), std::sin(t / 2), std::cos(t / 2)};
        }
        if (name == "RZ") {
            return {std::exp(-I * (t / 2)), 0, 0, std::exp(I * (t / 2))};
        }
        RT_FAIL(("DefaultTensor: unsupported gate '" + name + "'").c_str());
    }

    static auto observableMatrix(const ObsRecord &rec) -> std::array<cd, 4>
    {
        switch (rec.basic_id) {
        case Identity:
            return {1, 0, 0, 1};
        case PauliX:
            return matX();
        case PauliY:
            return {0, cd{0, -1}, cd{0, 1}, 0};
        case PauliZ:
            return matZ();
        case Hadamard:
            return {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2};
        case Hermitian:
            return {rec.matrix[0], rec.matrix[1], rec.matrix[2], rec.matrix[3]};
        default:
            RT_FAIL("DefaultTensor: unsupported observable");
        }
    }

    auto frontierOf(QubitIdType q) -> int64_t
    {
        auto it = frontier_.find(q);
        RT_FAIL_IF(it == frontier_.end(),
                   "DefaultTensor: operation references an unallocated or released qubit");
        return it->second;
    }

    /**
     * @brief Append a (possibly controlled) single-qubit gate to the network.
     *
     * With c control wires this adds one rank-(2c+2) tensor: the controls are
     * passed through diagonally (a "copy" tensor) and the target gets `m`
     * applied only on the selected control branch.
     */
    void applyControlled(const std::array<cd, 4> &m, QubitIdType target,
                         const std::vector<QubitIdType> &ctrl_wires,
                         const std::vector<bool> &ctrl_values)
    {
        // Reject duplicate wires early; silently aliasing legs would produce a
        // wrong answer rather than an error.
        std::vector<QubitIdType> all{target};
        all.insert(all.end(), ctrl_wires.begin(), ctrl_wires.end());
        std::vector<QubitIdType> sorted = all;
        std::sort(sorted.begin(), sorted.end());
        RT_FAIL_IF(std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end(),
                   "DefaultTensor: target and control wires must be distinct");

        const size_t nc = ctrl_wires.size();
        const int64_t tOld = frontierOf(target);
        const int64_t tNew = freshLabel();

        std::vector<int64_t> cOld, cNew;
        cOld.reserve(nc);
        cNew.reserve(nc);
        for (QubitIdType c : ctrl_wires) {
            cOld.push_back(frontierOf(c));
            cNew.push_back(freshLabel());
        }

        // Index order: [ctrlOut..., ctrlIn..., targetOut, targetIn]
        std::vector<int64_t> idx;
        std::vector<size_t> dims;
        for (size_t k = 0; k < nc; k++) {
            idx.push_back(cNew[k]);
            dims.push_back(2);
        }
        for (size_t k = 0; k < nc; k++) {
            idx.push_back(cOld[k]);
            dims.push_back(2);
        }
        idx.push_back(tNew);
        dims.push_back(2);
        idx.push_back(tOld);
        dims.push_back(2);

        const size_t ctrlDim = size_t{1} << nc; // combinations of control values
        std::vector<cd> data(ctrlDim * ctrlDim * 4, cd{0.0, 0.0});

        // The control block is diagonal: out-combination == in-combination.
        size_t activeCombination = 0;
        for (size_t k = 0; k < nc; k++) {
            if (ctrl_values[k]) {
                activeCombination |= (size_t{1} << (nc - 1 - k));
            }
        }
        for (size_t c = 0; c < ctrlDim; c++) {
            const size_t base = (c * ctrlDim + c) * 4;
            if (c == activeCombination) {
                data[base + 0] = m[0];
                data[base + 1] = m[1];
                data[base + 2] = m[2];
                data[base + 3] = m[3];
            }
            else {
                data[base + 0] = cd{1.0, 0.0}; // identity on the target
                data[base + 3] = cd{1.0, 0.0};
            }
        }

        nodes_.push_back(Tensor(std::move(idx), std::move(dims), std::move(data)));
        frontier_[target] = tNew;
        for (size_t k = 0; k < nc; k++) {
            frontier_[ctrl_wires[k]] = cNew[k];
        }
    }

    /**
     * @brief Contract the network to the full amplitude tensor, in wire order.
     *
     * This is the exponential step, and is only ever reached because some
     * measurement asked for it.
     */
    auto amplitudes() -> std::vector<cd>
    {
        RT_FAIL_IF(frontier_.empty(), "DefaultTensor: no qubits are allocated");
        Tensor full = ExactTN::contractNetwork(nodes_, max_intermediate_);

        std::vector<int64_t> want;
        want.reserve(order_.size());
        for (QubitIdType q : order_) {
            want.push_back(frontier_.at(q));
        }
        RT_FAIL_IF(full.rank() != want.size(),
                   "DefaultTensor: contracted rank does not match qubit count");
        full = ExactTN::permuteTo(full, want);

        // Collapse the network to a single node so repeated measurements do not
        // re-contract the whole history.
        nodes_.clear();
        nodes_.push_back(full);
        return full.data;
    }

    /// Fold the network into one tensor without changing the state it encodes.
    void compress()
    {
        if (nodes_.size() <= 1 || frontier_.empty()) {
            return;
        }
        Tensor full = ExactTN::contractNetwork(nodes_, max_intermediate_);
        nodes_.clear();
        nodes_.push_back(std::move(full));
    }

    /**
     * @brief Marginal distribution over `wires`; wires[0] is the most
     *        significant bit.
     *
     * Computed as the diagonal of the reduced density matrix, obtained by
     * closing the ket against its own conjugate:
     *
     *   - wires NOT requested are traced out by joining bra and ket directly on
     *     the same leg (a sum over that index),
     *   - wires that ARE requested keep a separate bra and ket leg, giving
     *     rho[out, in]; the diagonal is then the probability.
     *
     * The point is that no full amplitude vector is ever formed. A marginal on
     * k wires costs an intermediate of order 4^k rather than 2^n, so partial
     * probabilities stay affordable on registers far too wide to write down.
     */
    auto marginal(const std::vector<QubitIdType> &wires) -> std::vector<double>
    {
        for (QubitIdType w : wires) {
            (void)frontierOf(w); // validate before doing expensive work
        }
        RT_FAIL_IF(wires.size() > 30, "DefaultTensor: too many wires requested at once");

        // Requested wires keep a distinct bra leg; everything else is contracted
        // straight through, which performs the partial trace.
        std::unordered_map<int64_t, int64_t> braLeg;
        std::vector<int64_t> ketOpen, braOpen;
        ketOpen.reserve(wires.size());
        braOpen.reserve(wires.size());
        for (QubitIdType w : wires) {
            const int64_t leg = frontier_.at(w);
            const int64_t primed = leg + ExactTN::kPrimeOffset;
            braLeg[leg] = primed;
            ketOpen.push_back(leg);
            braOpen.push_back(primed);
        }

        std::vector<Tensor> full = nodes_;
        for (const Tensor &src : nodes_) {
            Tensor t = src;
            for (auto &z : t.data) {
                z = std::conj(z);
            }
            for (auto &lab : t.idx) {
                const auto it = braLeg.find(lab);
                if (it != braLeg.end()) {
                    lab = it->second; // stays open: this wire is measured
                }
                else if (!isTraced(lab)) {
                    lab += ExactTN::kPrimeOffset; // internal bra label
                }
                // else: an open leg of an unrequested wire -> shared with the
                // ket, which contracts it and traces that wire out.
            }
            full.push_back(std::move(t));
        }

        Tensor rho = ExactTN::contractNetwork(std::move(full), max_intermediate_);

        std::vector<int64_t> want = ketOpen;
        want.insert(want.end(), braOpen.begin(), braOpen.end());
        RT_FAIL_IF(rho.rank() != want.size(),
                   "DefaultTensor: reduced density matrix has unexpected rank");
        rho = ExactTN::permuteTo(rho, want);

        // Diagonal entries rho[i, i] are the probabilities.
        const size_t dim = size_t{1} << wires.size();
        std::vector<double> p(dim, 0.0);
        for (size_t i = 0; i < dim; i++) {
            p[i] = rho.data[i * dim + i].real();
            if (p[i] < 0.0) {
                p[i] = 0.0; // clamp round-off noise
            }
        }
        // Renormalise against accumulated floating-point error so that samplers
        // downstream always see a valid distribution.
        double total = 0.0;
        for (double v : p) {
            total += v;
        }
        RT_FAIL_IF(total <= 0.0, "DefaultTensor: marginal has zero total probability");
        for (double &v : p) {
            v /= total;
        }
        return p;
    }

    /// True if `label` is the open frontier leg of some currently live qubit.
    auto isTraced(int64_t label) const -> bool
    {
        for (const auto &[q, leg] : frontier_) {
            if (leg == label) {
                return true;
            }
        }
        return false;
    }

    /// One summand of an observable: a scalar coefficient, the operator tensors
    /// it contributes, and where each wire's leg ends up after them.
    struct Term {
        double coeff{1.0};
        std::vector<Tensor> tensors;
        std::map<QubitIdType, int64_t> front;
    };

    /**
     * @brief Compute <psi|O|psi> (or <psi|O^2|psi>) as one closed network.
     *
     * Layout, per summand of O:
     *
     *   ket  ...--L_q---[ O ]---M_q---...  bra (conjugated)
     *
     * The ket keeps its own labels. The observable consumes the ket's open leg
     * L_q and produces M_q. The bra is a conjugated copy of the ket whose
     * *internal* labels are shifted by kPrimeOffset but whose open leg for wire
     * q is renamed L_q -> M_q, so the network closes into a scalar. If the
     * observable is identity on a wire then M_q == L_q and the two sides meet
     * directly.
     *
     * No statevector is materialised; the contractor picks the order.
     */
    auto expvalSandwich(ObsIdType key, bool squared) -> double
    {
        RT_FAIL_IF(frontier_.empty(), "DefaultTensor: no qubits are allocated");
        validateObs(key);

        std::vector<Term> terms{Term{1.0, {}, frontier_}};
        const int reps = squared ? 2 : 1;
        for (int r = 0; r < reps; r++) {
            terms = applyObsToTerms(key, terms);
        }

        double total = 0.0;
        for (const Term &term : terms) {
            // ket + operator tensors
            std::vector<Tensor> full = nodes_;
            full.insert(full.end(), term.tensors.begin(), term.tensors.end());

            // Map for the bra's open legs: original frontier -> post-operator leg.
            std::unordered_map<int64_t, int64_t> openRemap;
            for (const auto &[q, leg] : frontier_) {
                openRemap[leg] = term.front.at(q);
            }

            for (const Tensor &src : nodes_) {
                Tensor t = src;
                for (auto &z : t.data) {
                    z = std::conj(z);
                }
                for (auto &lab : t.idx) {
                    const auto it = openRemap.find(lab);
                    lab = (it != openRemap.end()) ? it->second : lab + ExactTN::kPrimeOffset;
                }
                full.push_back(std::move(t));
            }

            Tensor scalar = ExactTN::contractNetwork(full, max_intermediate_);
            RT_FAIL_IF(scalar.rank() != 0,
                       "DefaultTensor: expectation network did not close to a scalar");
            total += term.coeff * scalar.data[0].real();
        }
        return total;
    }

    void validateObs(ObsIdType key) const
    {
        RT_FAIL_IF(key < 0 || static_cast<size_t>(key) >= obs_.size(),
                   "DefaultTensor: invalid observable id");
        const ObsRecord &rec = obs_[static_cast<size_t>(key)];
        for (ObsIdType t : rec.terms) {
            validateObs(t);
        }
    }

    /**
     * @brief Multiply every accumulated term by observable `key`.
     *
     * A tensor product multiplies each term by all its factors in sequence; a
     * Hamiltonian expands the list (distributing over the sum). Each Term
     * carries its own frontier, so summands acting on different wires never
     * interfere -- which is why no frontier "harmonisation" is needed.
     *
     * Labels are drawn from the device counter so they can never collide with
     * the ket's existing labels.
     */
    auto applyObsToTerms(ObsIdType key, const std::vector<Term> &terms) -> std::vector<Term>
    {
        const ObsRecord &rec = obs_[static_cast<size_t>(key)];

        if (rec.kind == TensorProd) {
            std::vector<Term> acc = terms;
            for (ObsIdType t : rec.terms) {
                acc = applyObsToTerms(t, acc);
            }
            return acc;
        }

        if (rec.kind == Hamiltonian) {
            std::vector<Term> out;
            for (size_t k = 0; k < rec.terms.size(); k++) {
                std::vector<Term> scaled = terms;
                for (Term &t : scaled) {
                    t.coeff *= rec.coeffs[k];
                }
                std::vector<Term> part = applyObsToTerms(rec.terms[k], scaled);
                out.insert(out.end(), std::make_move_iterator(part.begin()),
                           std::make_move_iterator(part.end()));
            }
            return out;
        }

        // Basic single-wire observable. Identity needs no tensor at all, which
        // keeps Hamiltonian summands on disjoint wires cheap.
        const QubitIdType w = rec.wires[0];
        const auto m = observableMatrix(rec);
        if (rec.basic_id == Identity) {
            return terms;
        }

        std::vector<Term> acc = terms;
        for (Term &term : acc) {
            auto it = term.front.find(w);
            RT_FAIL_IF(it == term.front.end(),
                       "DefaultTensor: observable references an unallocated qubit");
            const int64_t in = it->second;
            const int64_t out = freshLabel();
            term.tensors.push_back(Tensor({out, in}, {2, 2}, {m[0], m[1], m[2], m[3]}));
            term.front[w] = out;
        }
        return acc;
    }

    /// Shared implementation for Measure and ReleaseQubit.
    auto measureImpl(QubitIdType wire, std::optional<int32_t> postselect) -> bool
    {
        const int64_t leg = frontierOf(wire);
        const auto p = marginal({wire});
        const double p1 = p[1];

        bool outcome;
        if (postselect.has_value()) {
            RT_FAIL_IF(postselect.value() != 0 && postselect.value() != 1,
                       "DefaultTensor: postselect must be 0 or 1");
            outcome = (postselect.value() == 1);
        }
        else {
            outcome = (uniform() < p1);
        }

        const double norm = outcome ? p1 : (1.0 - p1);
        RT_FAIL_IF(norm <= 0.0,
                   "DefaultTensor: postselected on a zero-probability outcome");

        // Project: attach |outcome><outcome| / sqrt(norm) to the wire.
        const double inv = 1.0 / std::sqrt(norm);
        const int64_t out = freshLabel();
        std::vector<cd> proj(4, cd{0.0, 0.0});
        if (outcome) {
            proj[3] = cd{inv, 0.0}; // |1><1|
        }
        else {
            proj[0] = cd{inv, 0.0}; // |0><0|
        }
        nodes_.push_back(Tensor({out, leg}, {2, 2}, std::move(proj)));
        frontier_[wire] = out;

        last_outcome_ = outcome;
        return outcome;
    }
};

} // namespace Catalyst::Runtime::Devices
