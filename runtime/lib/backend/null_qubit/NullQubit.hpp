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
#include <complex>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "DataView.hpp"
#include "InstructionStrBuilder.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "Types.h"

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
    NullQubit(const std::string &kwargs = "{}")
    {
        std::unordered_map<std::string, std::string> device_kwargs =
            Catalyst::Runtime::parse_kwargs(kwargs);
        print_instructions = device_kwargs["print_instructions"] == "True" ? 1 : 0;
    }
    ~NullQubit() = default; // LCOV_EXCL_LINE

    NullQubit &operator=(const NullQubit &) = delete;
    NullQubit(const NullQubit &) = delete;
    NullQubit(NullQubit &&) = delete;
    NullQubit &operator=(NullQubit &&) = delete;

    /**
     * @brief Doesn't Allocate a qubit.
     *
     * @return `QubitIdType`
     */
    auto AllocateQubit() -> QubitIdType
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("AllocateQubit", "")
                      << std::endl;
        }

        num_qubits_++; // next_id
        return this->qubit_manager.Allocate(num_qubits_);
    }

    /**
     * @brief Allocate a vector of qubits.
     *
     * @param num_qubits The number of qubits to allocate.
     *
     * @return `std::vector<QubitIdType>`
     */
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
    {
        bool prev_print_instructions = print_instructions;
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("AllocateQubits", num_qubits)
                      << std::endl;
        }
        print_instructions = false; // set to false so we do not print instructions for each call to
                                    // AllocateQubit() below.

        if (!num_qubits) {
            return {};
        }
        std::vector<QubitIdType> result(num_qubits);
        std::generate_n(result.begin(), num_qubits, [this]() { return AllocateQubit(); });

        print_instructions = prev_print_instructions; // restore actual value of print_instructions
        return result;
    }

    /**
     * @brief Doesn't Release a qubit.
     */
    void ReleaseQubit(QubitIdType q)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("ReleaseQubit", q) << std::endl;
        }

        if (!num_qubits_) {
            num_qubits_--;
            this->qubit_manager.Release(q);
        }
    }

    /**
     * @brief Doesn't Release all qubits.
     */
    void ReleaseAllQubits()
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("ReleaseAllQubits", "")
                      << std::endl;
        }

        num_qubits_ = 0;
        this->qubit_manager.ReleaseAll();
    }

    /**
     * @brief Doesn't Get the number of allocated qubits.
     *
     * @return `size_t`
     */
    [[nodiscard]] auto GetNumQubits() const -> size_t { return num_qubits_; }

    /**
     * @brief Doesn't Set the number of device shots.
     *
     * @param shots The number of noise shots
     */
    void SetDeviceShots(size_t shots) {}

    /**
     * @brief Doesn't Get the number of device shots.
     *
     * @return `size_t`
     */
    [[nodiscard]] auto GetDeviceShots() const -> size_t { return 0; }

    /**
     * @brief Doesn't Set the PRNG of the device.
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
     * @brief Doesn't Start recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    void StartTapeRecording() {}

    /**
     * @brief Doesn't Stop recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    void StopTapeRecording() {}

    /**
     * @brief Not the Result value for "Zero" used in the measurement process.
     *
     * @return `Result`
     */
    [[nodiscard]] auto Zero() const -> Result { return NULL; }

    /**
     * @brief Not the Result value for "One"  used in the measurement process.
     *
     * @return `Result`
     */
    [[nodiscard]] auto One() const -> Result { return NULL; }

    /**
     * @brief Not A helper method to print the state vector of a device.
     */
    void PrintState() {}

    /**
     * @brief Doesn't Prepare subsystems using the given ket vector in the computational basis.
     */
    void SetState(DataView<std::complex<double>, 1> &, std::vector<QubitIdType> &) {}

    /**
     * @brief Doesn't Prepare a single computational basis state.
     */
    void SetBasisState(DataView<int8_t, 1> &, std::vector<QubitIdType> &) {}

    /**
     * @brief Doesn't Apply a single gate to the state vector of a device with its name if this is
     * supported.
     */
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires = {},
                        const std::vector<bool> &controlled_values = {})
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_named_op_str(
                             name, params, wires, inverse, controlled_wires, controlled_values)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Apply a given matrix directly to the state vector of a device.
     *
     */
    void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse,
                         const std::vector<QubitIdType> &controlled_wires = {},
                         const std::vector<bool> &controlled_values = {})
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_matrix_op_str(
                             matrix, wires, inverse, controlled_wires, controlled_values)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Construct a named (Identity, PauliX, PauliY, PauliZ, and Hadamard)
     * or Hermitian observable.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto Observable(ObsId obs_id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires) -> ObsIdType
    {
        if (print_instructions) {
            return instruction_str_builder.create_obs_str(obs_id, matrix, wires);
        }

        return 0;
    }

    /**
     * @brief Doesn't Construct a tensor product of observables.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto TensorObservable(const std::vector<ObsIdType> &obs_keys) -> ObsIdType
    {
        if (print_instructions) {
            return instruction_str_builder.create_tensor_obs_str(obs_keys);
        }

        return 0;
    }

    /**
     * @brief Doesn't Construct a Hamiltonian observable.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto HamiltonianObservable(const std::vector<double> &matrix,
                               const std::vector<ObsIdType> &obs_keys) -> ObsIdType
    {
        if (print_instructions) {
            return instruction_str_builder.create_hamiltonian_obs_str(matrix, obs_keys);
        }

        return 0;
    }

    /**
     * @brief Doesn't Compute the expected value of an observable.
     *
     * @return `double` The expected value
     */
    auto Expval(ObsIdType o) -> double
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_op_with_obs_str("Expval", o) << std::endl;
        }
        return 0.0;
    }

    /**
     * @brief Doesn't Compute the variance of an observable.
     *
     * @return `double` The variance
     */
    auto Var(ObsIdType o) -> double
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_op_with_obs_str("Var", o) << std::endl;
        }
        return 0.0;
    }

    /**
     * @brief Doesn't Get the state-vector of a device.
     */
    void State(DataView<std::complex<double>, 1> &state)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("State", "") << std::endl;
        }
    }

    /**
     * @brief Doesn't Compute the probabilities of each computational basis state.
     */
    void Probs(DataView<double, 1> &probs)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_op_with_view_str("Probs", probs) << std::endl;
        }
    }

    /**
     * @brief Doesn't Compute the probabilities for a subset of the full system.
     */
    void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &wires)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_op_with_view_str("PartialProbs", probs, wires)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Compute samples with the number of shots on the entire wires,
     * returing raw samples.
     */
    void Sample(DataView<double, 2> &samples, size_t shots)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_samples_str("Sample", samples, shots)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Compute partial samples with the number of shots on `wires`,
     * returing raw samples.
     *
     * @param samples The pre-allocated `DataView<double, 2>`representing a matrix of
     * shape `shots * numWires`. The built-in iterator in `DataView<double, 2>`
     * iterates over all elements of `samples` row-wise.
     */
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires,
                       size_t shots)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_samples_str("PartialSample", samples, shots,
                                                                 wires)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Sample with the number of shots on the entire wires, returning the
     * number of counts for each sample.
     */
    void Counts(DataView<double, 1> &eigen_vals, DataView<int64_t, 1> &counts, size_t shots)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_counts_str("Counts", eigen_vals, counts, shots)
                      << std::endl;
        }
    }

    /**
     * @brief Doesn't Partial sample with the number of shots on `wires`, returning the
     * number of counts for each sample.
     */
    void PartialCounts(DataView<double, 1> &eigen_vals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires, size_t shots)
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_counts_str("PartialCounts", eigen_vals, counts,
                                                                shots, wires)
                      << std::endl;
        }
    }

    /**
     * @brief This is not A general measurement method that acts on a single wire.
     *
     * @return `Result` The measurement result
     */
    auto Measure(QubitIdType q, std::optional<int32_t> postselect) -> Result
    {
        if (print_instructions) {
            std::cout << instruction_str_builder.get_simple_op_str("Measure", q) << std::endl;
        }

        bool *ret = (bool *)malloc(sizeof(bool));
        *ret = true;
        return ret;
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

  private:
    bool print_instructions;
    std::size_t num_qubits_{0};
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
    Catalyst::Runtime::InstructionStrBuilder instruction_str_builder{};
};
} // namespace Catalyst::Runtime::Devices
