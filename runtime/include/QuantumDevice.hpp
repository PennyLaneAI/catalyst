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

#pragma once

#include <complex>
#include <memory>
#include <vector>

#include "Types.h"

namespace Catalyst::Runtime {

/**
 * @brief struct API for backend quantum devices.
 *
 * This device API contains,
 * - a set of methods to manage qubit allocations and deallocations, device shot
 *   noise, and quantum tape recording as well as reference values for the result
 *   data-type; these are used to implement Quantum Runtime (QR) instructions.
 *
 * - a set of methods for quantum operations, observables, measurements, and gradient
 *   of the device; these are used to implement Quantum Instruction Set (QIS) instructions.
 *
 */
struct QuantumDevice {
    QuantumDevice() = default;  // LCOV_EXCL_LINE
    virtual ~QuantumDevice() {} // LCOV_EXCL_LINE

    QuantumDevice &operator=(const QuantumDevice &) = delete;
    QuantumDevice(const QuantumDevice &) = delete;

    /**
     * @brief Allocate a qubit.
     *
     * @return QubitIdType
     */
    virtual auto AllocateQubit() -> QubitIdType = 0;

    /**
     * @brief Allocate a vector of qubits.
     *
     * @param num_qubits The number of qubits to allocate.
     *
     * @return `std::vector<QubitIdType>`
     */
    virtual auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> = 0;

    /**
     * @brief Release a qubit.
     *
     * @param qubit The id of the qubit
     */
    virtual void ReleaseQubit(QubitIdType qubit) = 0;

    /**
     * @brief Release all qubits.
     */
    virtual void ReleaseAllQubits() = 0;

    /**
     * @brief Get the number of allocated qubits.
     *
     * @return `size_t`
     */
    virtual auto GetNumQubits() -> size_t = 0;

    /**
     * @brief Set the number of device shots.
     *
     * @param shots The number of noise shots
     */
    virtual void SetDeviceShots(size_t shots) = 0;

    /**
     * @brief Get the number of device shots.
     *
     * @return `size_t`
     */
    virtual auto GetDeviceShots() -> size_t = 0;

    /**
     * @brief Start recording a quantum tape if provided.
     */
    virtual void StartTapeRecording() = 0;

    /**
     * @brief Stop recording a quantum tape if provided.
     */
    virtual void StopTapeRecording() = 0;

    /**
     * @brief Result value for "Zero" used in the measurement process.
     *
     * @return `Result`
     */
    virtual auto Zero() -> Result = 0;

    /**
     * @brief Result value for "One"  used in the measurement process.
     *
     * @return `Result`
     */
    virtual auto One() -> Result = 0;

    /**
     * @brief A helper method to print the state vector of a device.
     */
    virtual void PrintState() = 0;

    /**
     * @brief Apply a single gate to the state vector of a device with its name if this is
     * supported.
     *
     * @param name The name of the gate to apply
     * @param params Optional parameter list for parametric gates
     * @param wires Wires to apply gate to
     * @param inverse Indicates whether to use inverse of gate
     */
    virtual void NamedOperation(const std::string &name, const std::vector<double> &params,
                                const std::vector<QubitIdType> &wires, bool inverse) = 0;

    /**
     * @brief Apply a given matrix directly to the state vector of a device.
     *
     * @param matrix The matrix of data in row-major format
     * @param wires Wires to apply gate to
     * @param inverse Indicates whether to use inverse of gate
     */
    virtual void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                 const std::vector<QubitIdType> &wires, bool inverse) = 0;

    /**
     * @brief Construct a named (Identity, PauliX, PauliY, PauliZ, and Hadamard)
     * or Hermitian observable.
     *
     * @param id The type of the observable
     * @param matrix The matrix of data to construct a hermitian observable
     * @param wires Wires to apply observable to
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    virtual auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                            const std::vector<QubitIdType> &wires) -> ObsIdType = 0;

    /**
     * @brief Construct a tensor product of observables.
     *
     * @param obs The vector of observables indices of type ObsIdType
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    virtual auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType = 0;

    /**
     * @brief Construct a Hamiltonian observable.
     *
     * @param coeffs The vector of coefficients
     * @param obs The vector of observables indices of size `coeffs`
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    virtual auto HamiltonianObservable(const std::vector<double> &coeffs,
                                       const std::vector<ObsIdType> &obs) -> ObsIdType = 0;

    /**
     * @brief Compute the expected value of an observable.
     *
     * @param obsKey The index of the constructed observable
     *
     * @return `double` The expected value
     */
    virtual auto Expval(ObsIdType obsKey) -> double = 0;

    /**
     * @brief Compute the variance of an observable.
     *
     * @param obsKey The index of the constructed observable
     *
     * @return `double` The variance
     */
    virtual auto Var(ObsIdType obsKey) -> double = 0;

    /**
     * @brief Compute the probabilities of each computational basis state.
     *
     * @return `std::vector<double>`
     */
    virtual auto Probs() -> std::vector<double> = 0;

    /**
     * @brief Compute the probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset of the full system
     *
     * @return `std::vector<double>`
     */
    virtual auto PartialProbs(const std::vector<QubitIdType> &wires) -> std::vector<double> = 0;

    /**
     * @brief Get the state-vector of a device.
     *
     * @return `std::vector<std::complex<double>>`
     */
    virtual auto State() -> std::vector<std::complex<double>> = 0;

    /**
     * @brief Compute samples with the number of shots on the entire wires,
     * returing raw samples.
     *
     * @param shots The number of shots
     *
     * @return `std::vector<double>`
     */
    virtual auto Sample(size_t shots) -> std::vector<double> = 0;

    /**
     * @brief Compute partial samples with the number of shots on `wires`,
     * returing raw samples.
     *
     * @param wires Wires to compute samples on
     * @param shots The number of shots
     *
     * @return `std::vector<double>`
     */
    virtual auto PartialSample(const std::vector<QubitIdType> &wires, size_t shots)
        -> std::vector<double> = 0;

    /**
     * @brief Sample with the number of shots on the entire wires, returning the
     * number of counts for each sample.
     *
     * @param shots The number of shots
     *
     * @return `std::tuple<std::vector<double>, std::vector<int64_t>>` (eigvals, counts)
     */
    virtual auto Counts(size_t shots) -> std::tuple<std::vector<double>, std::vector<int64_t>> = 0;

    /**
     * @brief Partial sample with the number of shots on `wires`, returning the
     * number of counts for each sample.
     *
     * @param wires Wires to compute samples on
     * @param shots The number of shots
     *
     * @return `std::tuple<std::vector<double>, std::vector<int64_t>>` (eigvals, counts)
     */
    virtual auto PartialCounts(const std::vector<QubitIdType> &wires, size_t shots)
        -> std::tuple<std::vector<double>, std::vector<int64_t>> = 0;

    /**
     * @brief A general measurement method that acts on a single wire.
     *
     * @param wire The wire to compute Measure on

     * @return `Result` The measurement result
     */
    virtual auto Measure(QubitIdType wire) -> Result = 0;

    /**
     * @brief Compute the gradient of a quantum tape, that is cached using
     * `Catalyst::Runtime::Simulator::CacheManager`, for a specific set of trainable
     * parameters.
     *
     * @param trainParams The vector of trainable parameters; if none, all parameters
     * would be assumed trainable
     *
     * @return `std::vector<std::vector<double>>` A vector of jacobians for each observables
     */
    virtual auto Gradient(const std::vector<size_t> &trainParams)
        -> std::vector<std::vector<double>> = 0;
};

/**
 * Create a pointer of type `QuantumDevice` for an implementation of this device API.
 *
 * @return std::unique_ptr<QuantumDevice>
 */
std::unique_ptr<QuantumDevice> CreateQuantumDevice();

} // namespace Catalyst::Runtime
