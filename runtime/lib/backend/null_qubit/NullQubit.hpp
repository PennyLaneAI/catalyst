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
 * @brief A null backend quantum device.
 *
 * This device API provides a complete null backend implementation that:
 * - Manages qubit allocations and deallocations without actual quantum state
 * - Supports device shot configuration
 * - Implements all Quantum Runtime (QR) and Quantum Instruction Set (QIS) methods as no-ops
 * - Optionally tracks resource usage including gate counts, wire usage, and circuit depth
 * - Returns mock results for all measurements and observations
 *
 * The null device is particularly useful for:
 * - Testing quantum program compilation and execution without quantum simulation overhead
 * - Resource tracking & resource estimation
 * - Validating quantum program structure and control flow
 */
struct NullQubit final : public Catalyst::Runtime::QuantumDevice {
    std::unordered_map<std::string, std::string> device_kwargs;

    /**
     * @brief Constructs a NullQubit device with optional configuration parameters
     *
     * Parses device configuration from JSON-like string format and sets up
     * resource tracking options if specified. By default, resource tracking is turned off.
     * Supported parameters:
     * - "track_resources": Enable/disable resource tracking ("True"/"False")
     * - "resources_filename": Static filename for resource output [requires resource tracking]
     * - "compute_depth": Enable/disable circuit depth computation ("True"/"False") [requires
     * resource tracking]
     *
     * @param kwargs non-nested JSON-like string containing device configuration parameters
     */
    NullQubit(const std::string &kwargs = "{}")
    {
        this->device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);
        if (device_kwargs.contains("track_resources")) {
            this->track_resources_ = device_kwargs["track_resources"] == "True";
        }
        if (device_kwargs.contains("resources_filename")) {
            this->resource_tracker_.SetResourcesFilename(device_kwargs["resources_filename"]);
        }
        if (device_kwargs.contains("compute_depth")) {
            this->resource_tracker_.SetComputeDepth(device_kwargs["compute_depth"] == "True");
        }
    }
    ~NullQubit()
    {
        // We always want to gather resources that were used for an *entire* execution end-to-end
        // A device is guaranteed to live as long as its ExecutionContext, so its destructor is a
        // safe place to write out resource tracking data

        if (this->track_resources_) {
            this->resource_tracker_.WriteOut();
        }
    }

    NullQubit &operator=(const NullQubit &) = delete;
    NullQubit(const NullQubit &) = delete;
    NullQubit(NullQubit &&) = delete;
    NullQubit &operator=(NullQubit &&) = delete;

    /**
     * @brief Returns the device configuration parameters as a key-value map
     *
     * @return Map containing all device configuration parameters passed during construction
     */
    std::unordered_map<std::string, std::string> GetDeviceKwargs() { return this->device_kwargs; }

    /**
     * @brief Allocates a new "null" qubit and returns its qubit ID
     *
     * @return QubitIdType The qubit ID of the newly allocated qubit
     */
    auto AllocateQubit() -> QubitIdType
    {
        QubitIdType new_qubit = this->qubit_manager.Allocate(num_qubits_);
        if (this->track_resources_) {
            this->resource_tracker_.AllocateQubit(new_qubit);
        }
        num_qubits_++; // next_id
        return new_qubit;
    }

    /**
     * @brief Allocates a vector of "null" qubits and returns their qubit IDs
     *
     * @param num_qubits The number of qubits to allocate
     * @return std::vector<QubitIdType> Vector containing qubit IDs of the newly allocated qubits
     */
    auto AllocateQubits(std::size_t num_qubits) -> std::vector<QubitIdType>
    {
        if (!num_qubits) {
            return {};
        }
        std::vector<QubitIdType> result(num_qubits);
        std::generate_n(result.begin(), num_qubits, [this]() { return AllocateQubit(); });
        return result;
    }

    /**
     * @brief Releases a previously allocated qubit
     *
     * Decrements the qubit counter and releases the qubit through the qubit manager.
     *
     * @param q The qubit ID of the qubit to release
     */
    void ReleaseQubit(QubitIdType q)
    {
        if (num_qubits_) {
            num_qubits_--;
            this->qubit_manager.Release(q);
        }

        if (this->track_resources_) {
            this->resource_tracker_.ReleaseQubit(q);
        }
    }

    /**
     * @brief Releases qubits and optionally writes resource tracking data
     *
     * Decrements the qubit counter and releases the specified qubits through the qubit manager
     *
     * @param qubits A vector of the qubit IDs of the qubits to release
     */
    void ReleaseQubits(const std::vector<QubitIdType> &qubits)
    {
        for (auto q : qubits) {
            this->ReleaseQubit(q);
        }
    }

    /**
     * @brief Returns the current number of allocated qubits
     *
     * @return size_t The number of currently allocated qubits
     */
    [[nodiscard]] auto GetNumQubits() const -> std::size_t { return num_qubits_; }

    /**
     * @brief Sets the number of shots for measurement sampling
     *
     * @param shots The number of measurement shots to configure
     */
    void SetDeviceShots(std::size_t shots) { device_shots_ = shots; }

    /**
     * @brief Returns the current number of configured device shots
     *
     * @return size_t The number of shots configured for measurements
     */
    [[nodiscard]] auto GetDeviceShots() const -> std::size_t { return device_shots_; }

    /**
     * @brief No-op implementation for setting device PRNG state
     *
     * The null device doesn't perform any quantum simulation, so PRNG configuration
     * is not required. This method exists to satisfy the QuantumDevice interface.
     *
     * @param gen Pointer to the random number generator (ignored in null device)
     */
    void SetDevicePRNG([[maybe_unused]] std::mt19937 *gen) {}

    /**
     * @brief No-op implementation for starting quantum tape recording
     *
     * The null device doesn't maintain quantum state, so tape recording is not applicable.
     * This method exists to satisfy the QuantumDevice interface for compatibility with
     * caching and gradient computation systems.
     */
    void StartTapeRecording() {}

    /**
     * @brief No-op implementation for stopping quantum tape recording
     *
     * The null device doesn't maintain quantum state, so tape recording is not applicable.
     * This method exists to satisfy the QuantumDevice interface for compatibility with
     * caching and gradient computation systems.
     */
    void StopTapeRecording() {}

    /**
     * @brief No-op implementation for state preparation using a statevector
     *
     * The null device doesn't maintain quantum state, so state preparation is ignored.
     * This method exists to satisfy the QuantumDevice interface.
     *
     * @param state The state vector data (ignored)
     * @param wires The qubits to prepare (ignored)
     */
    void SetState(DataView<std::complex<double>, 1> &, std::vector<QubitIdType> &wires)
    {
        if (this->track_resources_) {
            this->resource_tracker_.SetState(wires);
        }
    }

    /**
     * @brief No-op implementation for computational basis state preparation
     *
     * The null device doesn't maintain quantum state, so basis state preparation is ignored.
     * This method exists to satisfy the QuantumDevice interface.
     *
     * @param basis_state The computational basis state (ignored)
     * @param wires The qubits to prepare (ignored)
     */
    void SetBasisState(DataView<int8_t, 1> &, std::vector<QubitIdType> &wires)
    {
        if (this->track_resources_) {
            this->resource_tracker_.SetBasisState(wires);
        }
    }

    /**
     * @brief No-op implementation for a named quantum operation
     *
     * If resource tracking is enabled, records the operation details including name,
     * parameters, and wire usage.
     *
     * @param name The name of the quantum operation (e.g., "PauliX", "CNOT", "RZ")
     * @param params Parameters for parametric gates (ignored)
     * @param wires The target qubits for the operation
     * @param inverse Whether this is an adjoint (inverse) operation
     * @param controlled_wires Control qubits for controlled operations
     * @param controlled_values Control values for multi-controlled operations
     */
    void NamedOperation(const std::string &name, [[maybe_unused]] const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires = {},
                        [[maybe_unused]] const std::vector<bool> &controlled_values = {},
                        [[maybe_unused]] const std::vector<std::string> &optional_params = {})
    {
        if (this->track_resources_) {
            this->resource_tracker_.NamedOperation(name, inverse, wires, controlled_wires);
        }
    }

    /**
     * @brief No-op implementation for a generic matrix operation
     *
     * If resource tracking is enabled, records the operation as a general unitary matrix
     * operation.
     *
     * @param matrix The unitary matrix defining the operation (ignored)
     * @param wires The target qubits for the operation
     * @param inverse Whether this is an adjoint (inverse) operation
     * @param controlled_wires Control qubits for controlled operations
     * @param controlled_values Control values for multi-controlled operations
     */
    void MatrixOperation([[maybe_unused]] const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse,
                         const std::vector<QubitIdType> &controlled_wires = {},
                         const std::vector<bool> &controlled_values = {})
    {
        if (this->track_resources_) {
            this->resource_tracker_.MatrixOperation(inverse, wires, controlled_wires);
        }
    }

    /**
     * @brief Creates a null observable and returns a dummy identifier
     *
     * The null device doesn't perform actual observable construction or measurement.
     * This method always returns 0 to satisfy the interface requirements.
     *
     * @param obs_id The type of observable (Identity, PauliX, PauliY, PauliZ, Hadamard, or
     * Hermitian)
     * @param matrix The matrix representation for Hermitian observables (ignored)
     * @param wires The qubits the observable acts on (ignored)
     * @return ObsIdType Always returns 0
     */
    auto Observable(ObsId, const std::vector<std::complex<double>> &,
                    const std::vector<QubitIdType> &) -> ObsIdType
    {
        return 0;
    }

    /**
     * @brief Creates a null tensor product observable and returns a dummy identifier
     *
     * The null device doesn't perform actual tensor product construction.
     * This method always returns 0 to satisfy the interface requirements.
     *
     * @param obs_ids Vector of observable identifiers to combine (ignored)
     * @return ObsIdType Always returns 0
     */
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType { return 0; }

    /**
     * @brief Creates a null Hamiltonian observable and returns a dummy identifier
     *
     * The null device doesn't perform actual Hamiltonian construction.
     * This method always returns 0 to satisfy the interface requirements.
     *
     * @param coeffs Coefficients for the Hamiltonian terms (ignored)
     * @param obs_ids Observable identifiers for the Hamiltonian terms (ignored)
     * @return ObsIdType Always returns 0
     */
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType
    {
        return 0;
    }

    /**
     * @brief Returns a dummy expectation value (always 0)
     *
     * The null device doesn't compute actual expectation values since it maintains
     * no quantum state. Always returns 0 for consistency.
     *
     * @param obs_id The observable identifier (ignored)
     * @return double Always returns 0
     */
    auto Expval(ObsIdType) -> double { return 0; }

    /**
     * @brief Returns a dummy variance value (always 0)
     *
     * The null device doesn't compute actual variances since it maintains
     * no quantum state. Always returns 0 for consistency.
     *
     * @param obs_id The observable identifier (ignored)
     * @return double Always returns 0
     */
    auto Var(ObsIdType) -> double { return 0; }

    /**
     * @brief Fills the state vector with the computational ground state
     *
     * Since the null device doesn't maintain actual quantum state, this method
     * simulates the pure ground state: |0⟩^⊗n. The first element is set to 1.0
     * and all remaining elements are set to 0.0.
     *
     * @param state The state vector to fill with ground state values
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
     * @brief Fills the probability array with ground state probabilities
     *
     * Since the null device simulates the ground state |0⟩^⊗n, this method sets
     * probability 1.0 for the ground basis state and 0.0 for all other states.
     * Example: [1, 0] for 1 qubit, [1, 0, 0, 0] for 2 qubits.
     *
     * @param probs The probability array to fill
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
     * @brief Fills the partial probability array with ground state probabilities
     *
     * Behaves identically to Probs() since the null device always simulates
     * the ground state regardless of which subset of qubits is measured.
     *
     * @param probs The probability array to fill
     * @param wires The subset of qubits to compute probabilities for (ignored)
     */
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &)
    {
        Probs(probs);
    }

    /**
     * @brief Fills the sample array with ground state measurements (all zeros)
     *
     * Since the null device simulates the ground state |0⟩^⊗n, all measurements
     * return 0. The samples array is filled with zeros for all shots and qubits.
     *
     * @param samples The 2D sample array to fill (shape: shots × num_qubits)
     */
    void Sample(DataView<double, 2> &samples)
    {
        // If num_qubits == 0, the samples array is unallocated (shape=(shots, 0)), so don't fill
        if (num_qubits_ > 0) {
            std::fill(samples.begin(), samples.end(), 0.0);
        }
    }

    /**
     * @brief Fills the partial sample array with ground state measurements (all zeros)
     *
     * Behaves identically to Sample() since the null device always simulates
     * the ground state regardless of which subset of qubits is measured.
     *
     * @param samples The 2D sample array to fill (shape: shots × num_target_wires)
     * @param wires The subset of qubits to sample from (ignored)
     */
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &)
    {
        Sample(samples);
    }

    /**
     * @brief Generates measurement count statistics for ground state (all shots in state 0)
     *
     * Since the null device simulates the ground state |0⟩^⊗n, all measurements yield 0.
     * The eigenvalues array is filled with integers 0, 1, ..., 2^num_qubits - 1,
     * while the counts array has all shots in the first position (state 0) and zeros elsewhere.
     *
     * @param eigvals Array to fill with possible computational basis measurement outcomes (0 to 2^n
     * - 1, inclusive)
     * @param counts Array to fill with count statistics (all shots in position 0)
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
     * @brief Generates partial measurement count statistics for ground state
     *
     * Behaves identically to Counts() since the null device always simulates
     * the ground state regardless of which subset of qubits is measured.
     *
     * @param eigvals Array to fill with possible measurement outcomes
     * @param counts Array to fill with count statistics
     * @param wires The subset of qubits to measure (ignored)
     */
    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &)
    {
        Counts(eigvals, counts);
    }

    /**
     * @brief Performs a dummy measurement that always returns false
     *
     * Since the null device simulates the ground state |0⟩^⊗n, all single-qubit
     * measurements in the computational basis always yield 0 (false).
     *
     * @param wire The qubit to measure (ignored)
     * @param postselect Optional postselection value (ignored)
     * @return Result Always returns a reference to the global false constant
     */
    auto Measure(QubitIdType, std::optional<int32_t>) -> Result
    {
        return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
    }

    /**
     * @brief No-op implementation for gradient computation
     *
     * The null device doesn't maintain quantum state or support gradient computation.
     * This method exists to satisfy the QuantumDevice interface.
     *
     * @param gradients Vector of gradient data views (not modified)
     * @param trainable_params Indices of trainable parameters (ignored)
     */
    void Gradient(std::vector<DataView<double, 1>> &, const std::vector<std::size_t> &) {}

    /**
     * @brief Returns dummy cache manager information
     *
     * The null device doesn't support caching, so this returns zero counts and empty vectors.
     *
     * @return Tuple containing cache statistics (all zeros/empty)
     */
    auto CacheManagerInfo() -> std::tuple<std::size_t, std::size_t, std::size_t,
                                          std::vector<std::string>, std::vector<ObsIdType>>
    {
        return {0, 0, 0, {}, {}};
    }

    /**
     * @brief Returns the filename where resource tracking data is written
     *
     * Only meaningful when resource tracking is enabled. Returns the configured
     * or auto-generated filename for resource output.
     *
     * @return std::string The resource tracking output filename
     */
    auto GetResourcesFilename() const -> std::string
    {
        return this->resource_tracker_.GetFilename();
    }

    /**
     * @brief Returns whether resource tracking is currently enabled
     *
     * @return bool True if resource tracking is enabled, false otherwise
     */
    auto IsTrackingResources() const -> bool { return track_resources_; }

  private:
    bool track_resources_{false};
    ResourceTracker resource_tracker_;
    std::size_t num_qubits_{0};
    std::size_t device_shots_{0};
    Catalyst::Runtime::QubitManager<QubitIdType, std::size_t> qubit_manager{};

    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;
};
} // namespace Catalyst::Runtime::Devices
