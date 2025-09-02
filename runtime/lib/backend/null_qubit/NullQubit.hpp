// Copyright 2022 Xanadu Quantum Technologies Inc.

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

#include <algorithm> // generate_n
#include <chrono>
#include <complex>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataView.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "ResourceTracker.hpp"
#include "Types.h"
#include "Utils.hpp"

namespace Catalyst::Runtime::Devices {

/**
 * @brief struct API for a null backend quantum device.
 *
 * This device API contains several NULL methods:
 * - a set of methods to manage qubit allocations and deallocations, device shot
 *   noise, and quantum tape recording as well as reference values for the result
 *   data-type; these are used to implement Quantum Runtime (QR) instructions.
 *
 * - a set of methods for quantum operations, observables, measurements, and gradient
 *   of the device; these are used to implement Quantum Instruction Set (QIS) instructions.
 */
struct NullQubit final : public Catalyst::Runtime::QuantumDevice {
    std::unordered_map<std::string, std::string> device_kwargs;

    NullQubit(const std::string &kwargs = "{}")
    {
        this->device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);
        if (device_kwargs.find("track_resources") != device_kwargs.end()) {
            this->track_resources_ = device_kwargs["track_resources"] == "True";
        }
        if (device_kwargs.find("resources_fname") != device_kwargs.end()) {
            this->resource_tracker_.SetResourcesFname(device_kwargs["resources_fname"]);
        }
        if (device_kwargs.find("compute_depth") != device_kwargs.end()) {
            this->resource_tracker_.SetComputeDepth(device_kwargs["compute_depth"] == "True");
        }
    }
    ~NullQubit() {} // LCOV_EXCL_LINE

    NullQubit &operator=(const NullQubit &) = delete;
    NullQubit(const NullQubit &) = delete;
    NullQubit(NullQubit &&) = delete;
    NullQubit &operator=(NullQubit &&) = delete;

    /**
     * @brief Get the device kwargs as a map.
     */
    std::unordered_map<std::string, std::string> GetDeviceKwargs() { return this->device_kwargs; }

    /**
     * @brief Allocate a "null" qubit.
     *
     * @return `QubitIdType`
     */
    auto AllocateQubit() -> QubitIdType
    {
        num_qubits_++; // next_id
        if (this->track_resources_) {
            this->resource_tracker_.SetMaxWires(num_qubits_);
        }
        return this->qubit_manager.Allocate(num_qubits_);
    }

    /**
     * @brief Allocate a vector of "null" qubits.
     *
     * @param num_qubits The number of qubits to allocate.
     *
     * @return `std::vector<QubitIdType>`
     */
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
    {
        if (!num_qubits) {
            return {};
        }
        std::vector<QubitIdType> result(num_qubits);
        std::generate_n(result.begin(), num_qubits, [this]() { return AllocateQubit(); });
        return result;
    }

    /**
     * @brief Release a qubit.
     */
    void ReleaseQubit(QubitIdType q)
    {
        if (num_qubits_) {
            num_qubits_--;
            this->qubit_manager.Release(q);
        }
    }

    /**
     * @brief Release all qubits.
     */
    void ReleaseAllQubits()
    {
        num_qubits_ = 0;
        this->qubit_manager.ReleaseAll();
        if (this->track_resources_) {
            this->resource_tracker_.WriteOut();
        }
    }

    /**
     * @brief Get the number of allocated qubits.
     *
     * @return `size_t`
     */
    [[nodiscard]] auto GetNumQubits() const -> size_t { return num_qubits_; }

    /**
     * @brief Set the number of device shots.
     *
     * @param shots The number of noise shots
     */
    void SetDeviceShots(size_t shots) { device_shots_ = shots; }

    /**
     * @brief Get the number of device shots.
     *
     * @return `size_t`
     */
    [[nodiscard]] auto GetDeviceShots() const -> size_t { return device_shots_; }

    /**
     * @brief Doesn't set the PRNG of the device.
     *
     * The Catalyst runtime enables seeded program execution on non-hardware devices.
     * A random number generator instance is managed by the runtime to predictably
     * generate results for non-deterministic programs, such as those involving `Measure`
     * calls.
     * Devices implementing support for this feature do not need to use the provided
     * PRNG instance as their sole source of random numbers, but it is expected that the
     * the same instance state will predictable and reproducibly generate the same
     * program results. It is also expected that the provided PRNG state is evolved
     * sufficiently so that two device executions sharing the same instance do not produce
     * identical results.
     * The provided PRNG instance is not thread-locked, and devices wishing to share it
     * across threads will need to provide their own thread-safety.
     */
    void SetDevicePRNG([[maybe_unused]] std::mt19937 *gen) {}

    /**
     * @brief Doesn't start recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    void StartTapeRecording() {}

    /**
     * @brief Doesn't stop recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    void StopTapeRecording() {}

    /**
     * @brief Doesn't prepare subsystems using the given ket vector in the computational basis.
     */
    void SetState(DataView<std::complex<double>, 1> &, std::vector<QubitIdType> &) {}

    /**
     * @brief Doesn't prepare a single computational basis state.
     */
    void SetBasisState(DataView<int8_t, 1> &, std::vector<QubitIdType> &) {}

    /**
     * @brief Doesn't apply a single gate to the state vector of a device with its name if this is
     * supported.
     */
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires = {},
                        const std::vector<bool> &controlled_values = {})
    {
        if (this->track_resources_) {
            this->resource_tracker_.NamedOperation(name, inverse, wires, controlled_wires);
        }
    }

    /**
     * @brief Doesn't apply a given matrix directly to the state vector of a device.
     *
     */
    void MatrixOperation(const std::vector<std::complex<double>> &,
                         const std::vector<QubitIdType> &wires, bool inverse,
                         const std::vector<QubitIdType> &controlled_wires = {},
                         const std::vector<bool> &controlled_values = {})
    {
        if (this->track_resources_) {
            this->resource_tracker_.MatrixOperation(inverse, wires, controlled_wires);
        }
    }

    /**
     * @brief Doesn't construct a named (Identity, PauliX, PauliY, PauliZ, and Hadamard)
     * or Hermitian observable.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto Observable(ObsId, const std::vector<std::complex<double>> &,
                    const std::vector<QubitIdType> &) -> ObsIdType
    {
        return 0.0;
    }

    /**
     * @brief Doesn't construct a tensor product of observables.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType { return 0.0; }

    /**
     * @brief Doesn't construct a Hamiltonian observable.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType
    {
        return 0.0;
    }

    /**
     * @brief Doesn't compute the expected value of an observable.
     *
     * Always return 0.
     *
     * @return `double` The expected value
     */
    auto Expval(ObsIdType) -> double { return 0.0; }

    /**
     * @brief Doesn't compute the variance of an observable.
     *
     * Always return 0.
     *
     * @return `double` The variance
     */
    auto Var(ObsIdType) -> double { return 0.0; }

    /**
     * @brief Doesn't get the state-vector of a device.
     *
     * Always fills the state vector corresponding to the pure ground state, e.g. [1, 0] for a
     * one-qubit system, [1, 0, 0, 0] for a two-qubit system, etc.
     */
    void State(DataView<std::complex<double>, 1> &state)
    {
        auto iter = state.begin();
        *iter = 1.0;
        if (num_qubits_ > 0) {
            ++iter;
            std::fill(iter, state.end(), 0.0);
        }
    }

    /**
     * @brief Doesn't compute the probabilities of each computational basis state.
     *
     * Always fills a probability of 1 for the ground basis state, and 0 for all other basis states,
     * e.g. [1, 0] for a one-qubit system, [1, 0, 0, 0] for a two-qubit system, etc.
     */
    void Probs(DataView<double, 1> &probs)
    {
        auto iter = probs.begin();
        *iter = 1.0;
        if (num_qubits_ > 0) {
            ++iter;
            std::fill(iter, probs.end(), 0.0);
        }
    }

    /**
     * @brief Doesn't compute the probabilities for a subset of the full system.
     *
     * Same behaviour as Probs().
     */
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &)
    {
        Probs(probs);
    }

    /**
     * @brief Doesn't compute samples with the number of shots on the entire wires,
     * returning raw samples.
     *
     * Always fills array of samples with 0.
     */
    void Sample(DataView<double, 2> &samples)
    {
        // If num_qubits == 0, the samples array is unallocated (shape=(shots, 0)), so don't fill
        if (num_qubits_ > 0) {
            std::fill(samples.begin(), samples.end(), 0.0);
        }
    }

    /**
     * @brief Doesn't Compute partial samples with the number of shots on `wires`,
     * returning raw samples.
     *
     * Same behaviour as Sample().
     *
     * @param samples The pre-allocated `DataView<double, 2>`representing a matrix of
     * shape `shots * numWires`. The built-in iterator in `DataView<double, 2>`
     * iterates over all elements of `samples` row-wise.
     */
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &)
    {
        Sample(samples);
    }

    /**
     * @brief Doesn't sample with the number of shots on the entire wires, returning the
     * number of counts for each sample.
     *
     * Always fills eigenvalues with integers ranging from 0, 1, ..., 2**num_qubits
     * Always sets the first element of counts to `shots` and fills the rest with 0.
     */
    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts)
    {
        auto iter_eigvals = eigvals.begin();
        *iter_eigvals = 0.0;
        ++iter_eigvals;

        auto iter_counts = counts.begin();
        *iter_counts = GetDeviceShots();
        ++iter_counts;

        if (num_qubits_ > 0) {
            std::iota(iter_eigvals, eigvals.end(), 1.0);
            std::fill(iter_counts, counts.end(), 0);
        }
    }

    /**
     * @brief Doesn't Partial sample with the number of shots on `wires`, returning the
     * number of counts for each sample.
     *
     * Same behaviour as Counts().
     */
    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &)
    {
        Counts(eigvals, counts);
    }

    /**
     * @brief This is not A general measurement method that acts on a single wire.
     *
     * @return `Result` The measurement result
     */
    auto Measure(QubitIdType, std::optional<int32_t>) -> Result
    {
        return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
    }

    /**
     * @brief Doesn't Compute the gradient of a quantum tape, that is cached using
     * `Catalyst::Runtime::Simulator::CacheManager`, for a specific set of trainable
     * parameters.
     */
    void Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &) {}

    auto CacheManagerInfo()
        -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>
    {
        return {0, 0, 0, {}, {}};
    }

    /**
     * @brief Returns the filename where resource tracking information is dumped. Only works if
     * resource tracking is enabled
     */
    auto GetResourcesFilename() const -> std::string
    {
        return this->resource_tracker_.GetFilename();
    }

    /**
     * @brief Returns whether the device is tracking resources or not.
     */
    auto IsTrackingResources() const -> bool { return track_resources_; }

  private:
    bool track_resources_{false};
    ResourceTracker resource_tracker_;
    std::size_t num_qubits_{0};
    std::size_t device_shots_{0};
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};

    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;
};
} // namespace Catalyst::Runtime::Devices
