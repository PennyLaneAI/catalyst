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
#include <optional>
#include <vector>

#include "DataView.hpp"
#include "Types.h"

// A helper template macro to generate the <IDENTIFIER>Factory method by
// calling <CONSTRUCTOR>(kwargs). Check the Custom Devices guideline for details:
// https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html
#define GENERATE_DEVICE_FACTORY(IDENTIFIER, CONSTRUCTOR)                                           \
    extern "C" Catalyst::Runtime::QuantumDevice *IDENTIFIER##Factory(const char *kwargs)           \
    {                                                                                              \
        return new CONSTRUCTOR(std::string(kwargs));                                               \
    }

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
    QuantumDevice() = default;          // LCOV_EXCL_LINE
    virtual ~QuantumDevice() = default; // LCOV_EXCL_LINE

    QuantumDevice &operator=(const QuantumDevice &) = delete;
    QuantumDevice(const QuantumDevice &) = delete;
    QuantumDevice(QuantumDevice &&) = delete;
    QuantumDevice &operator=(QuantumDevice &&) = delete;

    /**
     * @brief Allocate a qubit.
     *
     * @return `QubitIdType`
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
    [[nodiscard]] virtual auto GetNumQubits() const -> size_t = 0;

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
    [[nodiscard]] virtual auto GetDeviceShots() const -> size_t = 0;

    /**
     * @brief Start recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    virtual void StartTapeRecording() = 0;

    /**
     * @brief Stop recording a quantum tape if provided.
     *
     * @note This is backed by the `Catalyst::Runtime::CacheManager<ComplexT>` property in
     * the device implementation.
     */
    virtual void StopTapeRecording() = 0;

    /**
     * @brief Result value for "Zero" used in the measurement process.
     *
     * @return `Result`
     */
    [[nodiscard]] virtual auto Zero() const -> Result = 0;

    /**
     * @brief Result value for "One"  used in the measurement process.
     *
     * @return `Result`
     */
    [[nodiscard]] virtual auto One() const -> Result = 0;

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
     * @param controlled_wires Optional controlled wires applied to the operation
     * @param controlled_values Optional controlled values applied to the operation
     */
    virtual void
    NamedOperation(const std::string &name, const std::vector<double> &params,
                   const std::vector<QubitIdType> &wires, [[maybe_unused]] bool inverse = false,
                   [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires = {},
                   [[maybe_unused]] const std::vector<bool> &controlled_values = {}) = 0;

    /**
     * @brief Apply a given matrix directly to the state vector of a device.
     *
     * @param matrix The matrix of data in row-major format
     * @param wires Wires to apply gate to
     * @param inverse Indicates whether to use inverse of gate
     * @param controlled_wires Controlled wires applied to the operation
     * @param controlled_values Controlled values applied to the operation
     */
    virtual void
    MatrixOperation(const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires, [[maybe_unused]] bool inverse = false,
                    [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires = {},
                    [[maybe_unused]] const std::vector<bool> &controlled_values = {}) = 0;

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
     * @brief Get the state-vector of a device.
     *
     * @param state The pre-allocated `DataView<complex<double>, 1>`
     */
    virtual void State(DataView<std::complex<double>, 1> &state) = 0;

    /**
     * @brief Compute the probabilities of each computational basis state.

     * @param probs The pre-allocated `DataView<double, 1>`
     */
    virtual void Probs(DataView<double, 1> &probs) = 0;

    /**
     * @brief Compute the probabilities for a subset of the full system.
     *
     * @param probs The pre-allocated `DataView<double, 1>`
     * @param wires Wires will restrict probabilities to a subset of the full system
     */
    virtual void PartialProbs(DataView<double, 1> &probs,
                              const std::vector<QubitIdType> &wires) = 0;

    /**
     * @brief Compute samples with the number of shots on the entire wires,
     * returing raw samples.
     *
     * @param samples The pre-allocated `DataView<double, 2>`representing a matrix of
     * shape `shots * numQubits`. The built-in iterator in `DataView<double, 2>`
     * iterates over all elements of `samples` row-wise.
     * @param shots The number of shots
     */
    virtual void Sample(DataView<double, 2> &samples, size_t shots) = 0;

    /**
     * @brief Compute partial samples with the number of shots on `wires`,
     * returing raw samples.
     *
     * @param samples The pre-allocated `DataView<double, 2>`representing a matrix of
     * shape `shots * numWires`. The built-in iterator in `DataView<double, 2>`
     * iterates over all elements of `samples` row-wise.
     * @param wires Wires to compute samples on
     * @param shots The number of shots
     */
    virtual void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires,
                               size_t shots) = 0;

    /**
     * @brief Sample with the number of shots on the entire wires, returning the
     * number of counts for each sample.
     *
     * @param eigvals The pre-allocated `DataView<double, 1>`
     * @param counts The pre-allocated `DataView<int64_t, 1>`
     * @param shots The number of shots
     */
    virtual void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                        size_t shots) = 0;

    /**
     * @brief Partial sample with the number of shots on `wires`, returning the
     * number of counts for each sample.
     *
     * @param eigvals The pre-allocated `DataView<double, 1>`
     * @param counts The pre-allocated `DataView<int64_t, 1>`
     * @param wires Wires to compute samples on
     * @param shots The number of shots
     */
    virtual void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                               const std::vector<QubitIdType> &wires, size_t shots) = 0;

    /**
     * @brief A general measurement method that acts on a single wire.
     *
     * @param wire The wire to compute Measure on
     * @param postselect Which basis state to postselect after a mid-circuit measurement (-1 denotes
     no post-selection)

     * @return `Result` The measurement result
     */
    virtual auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result = 0;

    /**
     * @brief Compute the gradient of a quantum tape, that is cached using
     * `Catalyst::Runtime::Simulator::CacheManager`, for a specific set of trainable
     * parameters.
     *
     * @param gradients The vector of pre-allocated `DataView<double, 1>*`
     * to store gradients resutls for the list of cached observables.
     * @param trainParams The vector of trainable parameters; if none, all parameters
     * would be assumed trainable
     *
     */
    virtual void Gradient(std::vector<DataView<double, 1>> &gradients,
                          const std::vector<size_t> &trainParams) = 0;
};
} // namespace Catalyst::Runtime
