// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <numeric>
#include <utility>
#include <vector>

#include "BitUtil.hpp"
#include "LinearAlgebra.hpp"
#include "Util.hpp"

#include "Error.hpp"

#include <StateVectorCPU.hpp>

#include <iostream>

namespace Pennylane {
/**
 * @brief State-vector dynamic class.
 *
 * This class allocates and deallocates qubits/wires dynamically,
 * and defines all operations to manipulate the statevector data for
 * quantum circuit simulation.
 *
 */
template <class PrecisionT = double>
class StateVectorDynamicCPU : public StateVectorCPU<PrecisionT, StateVectorDynamicCPU<PrecisionT>> {
  public:
    using BaseType = StateVectorCPU<PrecisionT, StateVectorDynamicCPU<PrecisionT>>;

    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> data_;

    static constexpr PrecisionT epsilon_ = std::numeric_limits<PrecisionT>::epsilon() * 100;

    template <class IIter, class OIter>
    inline OIter _move_data_elements(IIter first, size_t distance, OIter second) {
        *second++ = std::move(*first);
        for (size_t i = 1; i < distance; i++) {
            *second++ = std::move(*++first);
        }
        return second;
    }

    template <class IIter, class OIter>
    inline OIter _shallow_move_data_elements(IIter first, size_t distance, OIter second) {
        for (size_t i = 0; i < distance; i++) {
            *second++ = std::move(*first);
            *first = Util::ZERO<PrecisionT>();
            first++;
        }
        return second;
    }

    inline void _scalar_mul_data(
        std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> &data,
        ComplexPrecisionT scalar) {
        std::transform(data.begin(), data.end(), data.begin(),
                       [scalar](const ComplexPrecisionT &elem) { return elem * scalar; });
    }

    inline void _normalize_data(
        std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> &data) {
        _scalar_mul_data(data, Util::ONE<PrecisionT>() /
                                   std::sqrt(Util::squaredNorm(data.data(), data.size())));
    }

  public:
    /**
     * @brief Create a new statevector
     *
     * @param num_qubits Number of qubits
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    explicit StateVectorDynamicCPU(size_t num_qubits, Threading threading = Threading::SingleThread,
                                   CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model},
          data_{Util::exp2(num_qubits), Util::ZERO<PrecisionT>(),
                getAllocator<ComplexPrecisionT>( // LCOV_EXCL_LINE
                    this->memory_model_)} {
        data_[0] = Util::ONE<PrecisionT>();
    }

    /**
     * @brief Construct a statevector from another statevector
     *
     * @tparam OtherDerived A derived type of StateVectorCPU to use for
     * construction.
     * @param other Another statevector to construct the statevector from
     */
    template <class OtherDerived>
    explicit StateVectorDynamicCPU(const StateVectorCPU<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(), other.memoryModel()),
          data_{other.getData(), other.getData() + other.getLength(),
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {}

    /**
     * @brief Construct a statevector from data pointer
     *
     * @param other_data Data pointer to construct the statvector from.
     * @param other_size Size of the data
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    StateVectorDynamicCPU(const ComplexPrecisionT *other_data, size_t other_size,
                          Threading threading = Threading::SingleThread,
                          CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(Util::log2PerfectPower(other_size), threading, memory_model),
          data_{other_data, other_data + other_size,
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
    }

    /**
     * @brief Construct a statevector from a data vector
     *
     * @tparam Alloc Allocator type of std::vector to use for constructing
     * statevector.
     * @param other Data to construct the statevector from
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    template <class Alloc>
    explicit StateVectorDynamicCPU(const std::vector<std::complex<PrecisionT>, Alloc> &other,
                                   Threading threading = Threading::SingleThread,
                                   CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorDynamicCPU(other.data(), other.size(), threading, memory_model) {}

    StateVectorDynamicCPU(const StateVectorDynamicCPU &rhs) = default;
    StateVectorDynamicCPU(StateVectorDynamicCPU &&) noexcept = default;

    StateVectorDynamicCPU &operator=(const StateVectorDynamicCPU &) = default;
    StateVectorDynamicCPU &operator=(StateVectorDynamicCPU &&) noexcept = default;

    ~StateVectorDynamicCPU() = default;

    /**
     * @brief Update data of the class to new_data
     *
     * @tparam Alloc Allocator type of std::vector to use for updating data.
     * @param new_data std::vector contains data.
     */
    template <class Alloc> void updateData(const std::vector<ComplexPrecisionT, Alloc> &new_data) {
        assert(data_.size() == new_data.size());
        std::copy(new_data.data(), new_data.data() + new_data.size(), data_.data());
    }

    Util::AlignedAllocator<ComplexPrecisionT> allocator() const { return data_.get_allocator(); }

    [[nodiscard]] auto isValidWire(size_t wire) -> bool { return wire < this->getNumQubits(); }

    /**
     * @brief Compute the purity of the system after releasing (a qubit) `wire`.
     *
     * This traces out the complement of the wire for a more efficient
     * computation of the purity in O(N) with calculating the reduced density
     * matrix after tracing out the complement of qubit `wire`.
     *
     * @param wire Index of the wire.
     * @return ComplexPrecisionT
     */
    auto getSubsystemPurity(size_t wire) -> ComplexPrecisionT {
        PL_ABORT_IF_NOT(isValidWire(wire), "Invalid wire: The wire must be in the range of wires");

        const size_t sv_size = data_.size();

        // With `k` indexing the subsystem on n-1 qubits, we need to insert an
        // addtional bit into the index of the full state-vector at position
        // `wire`. These masks enable us to split the bits of the index `k` into
        // those above and below `wire`.
        const size_t lower_mask = (1UL << wire) - 1;
        const size_t upper_mask = sv_size - lower_mask - 1;

        // The resulting 2x2 reduced density matrix of the complement system to
        // qubit `wire`.
        std::vector<ComplexPrecisionT> rho(4, {0, 0});

        for (uint8_t i = 0; i < 2; i++) {
            for (uint8_t j = 0; j < 2; j++) {
                ComplexPrecisionT sum{0, 0};
                for (size_t k = 0; k < (sv_size / 2); k++) {
                    size_t idx_wire_0 = (/* upper_bits: */ (upper_mask & k) << 1UL) +
                                        /* lower_bits: */ (lower_mask & k);
                    size_t idx_i = idx_wire_0 + (i << wire);
                    size_t idx_j = idx_wire_0 + (j << wire);

                    // This computes <00..i..00|psi><psi|00..j..00> on the first
                    // iteration, with the last iteration computing
                    // <11..i..11|psi><psi|11..j..11>.
                    sum += data_[idx_i] * std::conj(data_[idx_j]);
                }
                rho[2 * i + j] = sum;
            }
        }

        // Compute/Return the trace of rho**2
        return (rho[0] * rho[0]) + (ComplexPrecisionT{2, 0} * rho[1] * rho[2]) + (rho[3] * rho[3]);
    }

    /**
     * @brief Check the purity of a system after releasing/disabling `wire`.
     *
     * @param wire Index of the wire.
     * @param eps The comparing precision threshold.
     * @return bool
     */
    [[nodiscard]] auto checkSubsystemPurity(size_t wire, double eps = epsilon_) -> bool {
        ComplexPrecisionT purity = getSubsystemPurity(wire);
        return (std::abs(1.0 - purity.real()) < eps) && (purity.imag() < eps);
    }

    /**
     * @brief Allocate a new wire.
     *
     * @return It updates the state-vector and the number of qubits,
     * and returns index of the activated wire.
     */
    auto allocateWire() -> size_t {
        const size_t next_idx = this->getNumQubits();
        const size_t dsize = data_.size();
        data_.resize(dsize << 1UL);

        auto src = data_.begin();
        std::advance(src, dsize - 1);
        for (auto dst = data_.end() - 2; src != data_.begin();
             std::advance(src, -1), std::advance(dst, -2)) {
            _shallow_move_data_elements(src, 1, dst);
        }

        this->setNumQubits(next_idx + 1);
        return next_idx;
    }

    /**
     * @brief Release a given wire.
     *
     * @param wire Index of the wire to be released
     */
    void releaseWire(size_t wire) {
        PL_ABORT_IF_NOT(checkSubsystemPurity(wire),
                        "Invalid wire: "
                        "The state-vector must remain pure after releasing a wire")

        const size_t distance = 1UL << wire;

        auto dst = data_.begin();

        // Check if the reduced state-vector is the first-half
        bool is_first_half = false;
        for (auto src = dst; src < data_.end(); std::advance(src, 2 * distance)) {
            is_first_half =
                std::any_of(src, src + static_cast<long long>(distance),
                            [](ComplexPrecisionT &e) { return e != Util::ZERO<PrecisionT>(); });
            if (is_first_half) {
                break;
            }
        }

        auto src = dst;
        if (!is_first_half) {
            std::advance(src, distance);
        }

        for (; src < data_.end(); std::advance(src, 2 * distance), std::advance(dst, distance)) {
            _move_data_elements(src, distance, dst);
        }
        data_.resize(data_.size() / 2);
        // normalize state-vector
        _normalize_data(data_);

        this->setNumQubits(this->getNumQubits() - 1);
    }

    /**
     * @brief Update the state-vector to the initial state with 0-qubit.
     */
    void clearData() {
        data_.clear();
        this->setNumQubits(0);

        // the init state-vector
        data_.push_back(Util::ONE<PrecisionT>());
    }

    /**
     * @brief Get underlying C-style data of the state-vector.
     */
    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.data(); }

    /**
     * @brief Get underlying C-style data of the state-vector.
     */
    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * { return data_.data(); }

    /**
     * @brief Get underlying data vector.
     */
    [[nodiscard]] auto getDataVector()
        -> std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> & {
        return data_;
    }

    /**
     * @brief Get underlying data vector.
     */
    [[nodiscard]] auto getDataVector() const
        -> const std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> & {
        return data_;
    }
};

} // namespace Pennylane
