// Copyright 2022-2025 Xanadu Quantum Technologies Inc.

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
#include <random>
#include <vector>

#include "DataView.hpp"
#include "Types.h"

// A helper template macro to generate the <IDENTIFIER>Factory function by
// calling <CONSTRUCTOR>(kwargs). Check the Custom Devices guideline for details:
// https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html
#define GENERATE_DEVICE_FACTORY(IDENTIFIER, CONSTRUCTOR)                                           \
    extern "C" Catalyst::Runtime::QuantumDevice *IDENTIFIER##Factory(const char *kwargs)           \
    {                                                                                              \
        return new CONSTRUCTOR(std::string(kwargs));                                               \
    }

namespace Catalyst::Runtime {

/**
 * @brief Interface class for Catalyst Runtime device backends.
 *
 * All Catalyst device plugins must implement this class and distribute it as a shared library.
 * See https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html for details.
 *
 * The `QuantumDevice` interface methods can broadly be categorized into device management functions
 * that do not directly impact the computation or quantum state (think qubit management,
 * execution configuration, or recording functionality), and computation functions like quantum
 * gates and measurement procedures.
 * In addition, not all methods are required to be implemented by a plugin, as some features are
 * opt-in or not applicable to all device types. Optional features will be marked as such in the
 * method description and contain a stub implementation.
 */
struct QuantumDevice {
    QuantumDevice() = default;          // LCOV_EXCL_LINE
    virtual ~QuantumDevice() = default; // LCOV_EXCL_LINE

    QuantumDevice &operator=(const QuantumDevice &) = delete;
    QuantumDevice(const QuantumDevice &) = delete;
    QuantumDevice(QuantumDevice &&) = delete;
    QuantumDevice &operator=(QuantumDevice &&) = delete;

    // ----------------------------------------
    //  QUBIT MANAGEMENT
    // ----------------------------------------

    /**
     * @brief Allocate an array of qubits.
     *
     * The operation should perform the necessary steps to make the given number of qubits
     * available in a clean |0> state. At minimum, a device must be able to execute this
     * call once before any quantum operations have been run. Handling multiple allocation calls
     * is optional and only useful if the device intends to support dynamic qubit allocations
     * (that is allocations during quantum program execution).
     *
     * The values returned by this operation are integer IDs that will be used to address the
     * allocated qubits in subsequent operations (like gates or measurements). There are no
     * restrictions on the values these IDs can take, but the IDs of all active qubits must not
     * overlap, and once an ID is mapped to a particular qubit that ID must not change until the
     * qubit is explicitly freed.
     *
     * While devices may choose not to distinguish between logical and device IDs, having a logical
     * qubit labeling system can help catch program errors such as addressing previously freed
     * qubits. An example implementation can be found in `QubitManager.hpp`. Devices are also free
     * to disable a resource without physically freeing it, allowing for faster dynamic allocations.
     *
     * @param num_qubits The number of qubits to allocate.
     *
     * @return `std::vector<QubitIdType>` Array of qubit IDs.
     */
    virtual auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> = 0;

    /**
     * @brief Release an array of qubits.
     *
     * For devices without dynamic allocation support it is expected that this function
     * only succeed if the ID array contains the same values as those produced by the
     * initial `AllocateQubits` call, otherwise the device is encouraged to raise an error.
     *
     * See `ReleaseQubit` for caveats around releasing / resetting entangled qubits.
     *
     * Opposite of `AllocateQubits`.
     *
     * @param qubits array of IDs of the qubits to release.
     */
    virtual void ReleaseQubits(const std::vector<QubitIdType> &qubits) = 0;

    /**
     * @brief Get the number of currently allocated qubits.
     *
     * @return `size_t`
     */
    virtual auto GetNumQubits() const -> size_t = 0;

    /**
     * @brief (Optional) Allocate a qubit.
     *
     * Allocate a new qubit in the |0> state. This method is only needed for devices with dynamic
     * qubit allocation.
     *
     * See `AllocateQubits` for more details on allocation semantics.
     *
     * @return `QubitIdType` Qubit ID.
     */
    virtual auto AllocateQubit() -> QubitIdType
    {
        RT_FAIL("Dynamic qubit allocation is unsupported by device");
    }

    /**
     * @brief (Optional) Release a qubit.
     *
     * Release the provided qubit. This method is only needed for devices with dynamic qubit
     * allocation.
     *
     * Note that the interface does not require the qubit to be in a particular state. The behaviour
     * for releasing an entangled qubit is left up to the device, and could include:
     *  - raising a runtime error (assuming the device can detect impure states)
     *  - continuing execution with an entangled but inaccessible qubit
     *  - resetting the qubit with or without measuring its state
     * In any case, the deallocated qubit must not be accessible by any future instructions or its
     * state considered in any future results.
     *
     * Opposite of `AllocateQubit`.
     *
     * @param qubit ID of the qubit to release.
     */
    virtual void ReleaseQubit(QubitIdType qubit)
    {
        RT_FAIL("Dynamic qubit release is unsupported by device");
    }

    // ----------------------------------------
    //  EXECUTION MANAGEMENT
    // ----------------------------------------

    /**
     * @brief Set the number of execution shots.
     *
     * The Runtime will call this function to set the number of times the quantum execution is to
     * be repeated. Generally, it will be called once before any quantum instructions are run, but
     * some devices may choose to support it at arbitrary points in the program if they have the
     * capability to simulate shot noise.
     *
     * Devices with no or restricted support are encouraged to raise a runtime error.
     *
     * @param shots Shot number.
     */
    virtual void SetDeviceShots(size_t shots) = 0;

    /**
     * @brief Get the number of execution shots.
     *
     * @return `size_t` Shot number.
     */
    virtual auto GetDeviceShots() const -> size_t = 0;

    /**
     * @brief (Optional) Set the PRNG of the device.
     *
     * The Catalyst runtime enables seeded program execution on non-hardware devices.
     * A random number generator instance is managed by the runtime to predictably
     * generate results for non-deterministic programs, such as those involving `Measure`
     * calls.
     * Devices implementing support for this feature do not need to use the provided
     * PRNG instance as their sole source of randomness, but it is expected that the
     * the same instance state will predictably and reproducibly generate the same
     * program results (considering all random processes in the device).
     * It is also expected that the provided PRNG state is evolved sufficiently so that two copies
     * of a device provided with the same PRNG instance do not produce identical results when run
     * in succession.
     * Note that the provided PRNG instance is not thread-locked, and devices wishing to use it
     * across internal threads will need to provide their own thread-safety.
     *
     * @param gen Pointer to a Catalyst-managed Mersenne Twister instance.
     */
    virtual void SetDevicePRNG(std::mt19937 *gen) {};

    // ----------------------------------------
    //  QUANTUM OPERATIONS
    // ----------------------------------------

    /**
     * @brief Apply an arbitrary quantum gate to the device.
     *
     * This instruction is opaque to the chosen quantum operation, which is specified via a
     * string identifier. Additionally, an array of floating point parameters can be supplied,
     * as well certain quantum modifiers, like the Hermitian adjoint (inverse) and control qubits.
     * If the supplied combination of parameters is invalid or unsupported, the device should
     * raise a runtime error.
     *
     * @param name A string identifier for the operation to apply.
     * @param params Float parameters for parametric gates (may be empty).
     * @param wires Qubits to apply the operation to.
     * @param inverse Apply the inverse (Hermitian adjoint) of the operation.
     * @param controlled_wires Control qubits applied to the operation.
     * @param controlled_values Control values associated to the control qubits (equal length).
     */
    virtual void NamedOperation(const std::string &name, const std::vector<double> &params,
                                const std::vector<QubitIdType> &wires, bool inverse = false,
                                const std::vector<QubitIdType> &controlled_wires = {},
                                const std::vector<bool> &controlled_values = {}) = 0;
    /**
     * @brief Perform a computational-basis measurement on one qubit.
     *
     * This instruction is generally used for mid-circuit measurements, implementing an immediate,
     * random projective measurement and producing a classical result as output. However, it may
     * also be used in terminal measurements where the measurement processes below are unsupported.
     *
     * @param wire The qubit to measure.
     * @param postselect Optional parameter to force the result to the provided state (roughly
     *                   equivalent to post-selection).
     *
     * @return `Result` The measurement result.
     */
    virtual auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result = 0;

    /**
     * @brief (Optional) Apply an arbitrary unitary matrix to the device.
     *
     * Instead of identifying an operation by name, this instruction uses the mathematical
     * representation of a quantum operator as a unitary matrix in the computational basis.
     *
     * See `NamedOperation` for additional gate semantics.
     *
     * @param matrix A 1D array representation of the matrix in row-major format.
     * @param wires Qubits to apply the operation to.
     * @param inverse Apply the inverse of the operation.
     * @param controlled_wires Control qubits applied to the operation.
     * @param controlled_values Control values associated to the control qubits (equal length).
     */
    virtual void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                 const std::vector<QubitIdType> &wires, bool inverse = false,
                                 const std::vector<QubitIdType> &controlled_wires = {},
                                 const std::vector<bool> &controlled_values = {})
    {
        RT_FAIL("MatrixOperation is unsupported by device");
    }

    /**
     * @brief (Optional) Initialize qubits to a computational basis state.
     *
     * This instruction initializes a set of qubits to the provided computational basis state.
     * Although the instruction is typically used for state initialization at the beginning of
     * a quantum program, devices are encouraged to support arbitrary re-initialization if
     * capable. Reinitialization on a subset of qubits is equivalent to deallocation -> allocation
     * -> initialization on the same set of qubits.
     * See `ReleaseQubit` for caveats around releasing / resetting entangled qubits.
     *
     * @param n Bitstring representation of the basis state |n>, stored as a Byte-array.
     * @param wires The qubits to initialize.
     */
    virtual void SetBasisState(DataView<int8_t, 1> &n, std::vector<QubitIdType> &wires)
    {
        RT_FAIL("SetBasisState is unsupported by device");
    }

    /**
     * @brief (Optional) Initialize qubits to an arbitrary quantum state.
     *
     * Like `SetBasisState`, but instead of initializing to a single computational basis state
     * this instruction works with a vector of complex amplitudes, one for each possible basis
     * state.
     *
     * @param state Quantum state vector of size 2^len(wires).
     * @param wires The qubits to initialize.
     */
    virtual void SetState(DataView<std::complex<double>, 1> &state, std::vector<QubitIdType> &wires)
    {
        RT_FAIL("SetState is unsupported by device");
    }

    // ----------------------------------------
    //  QUANTUM OBSERVABLES
    // ----------------------------------------

    /**
     * @brief (Optional) Construct a named observable.
     *
     * This operation instructs the device to generate a named observable based on the provided enum
     * `id`, which will be one of {Identity, PauliX, PauliY, PauliZ, Hermitian}. The Hermitian kind
     * uses additional data to define the observable from a 2D complex matrix, which is expected to
     * be Hermitian. If this is not the case, the device is free to raise an error or produce
     * undefined behaviour. Additionally, the operation accepts a list of target qubits whose length
     * is expected to match the dimensionality of the observable.
     *
     * The observable system in the Catalyst Runtime relies primarily on the device's implementation
     * for support. As such, the device is free to represent observables in any way they wish.
     * Before using an observable in measurement processes, Catalyst will invoke one or more
     * observable methods from the interface. The device is expected to process and cache the
     * information in such a way that it is ready to use in a subsequent measurement process call.
     * The device must return an ID than unambiguously identifies the requested observable, but this
     * ID is not required to be unique across different calls supplied with the same information.
     *
     * @param id The name of the observable.
     * @param matrix The matrix of data to use for a Hermitian observable (unused otherwise).
     * @param wires Qubits the observable applies to.
     *
     * @return `ObsIdType` ID of the constructed observable.
     */
    virtual auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                            const std::vector<QubitIdType> &wires) -> ObsIdType
    {
        RT_FAIL("Observable is unsupported by device");
    }

    /**
     * @brief (Optional) Construct a tensor product of existing observables (prod).
     *
     * Given a list of observable IDs, construct the tensor product of all supplied observables.
     * If wires overlap across observables, the device should raise an error unless it can produce
     * sensible results.
     *
     * See `Observable` for additional details.
     *
     * @param obs The list of observables IDs.
     *
     * @return `ObsIdType` ID of the constructed observable.
     */
    virtual auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
    {
        RT_FAIL("TensorObservable is unsupported by device");
    }

    /**
     * @brief (Optional) Construct a linear combination of existing observables (sum).
     *
     * Given a list of observable IDs and associated coefficients, construct the linear combination
     * the supplied terms. The coefficients and observables are expected to have the same length.
     *
     *
     * See `Observable` for additional details.
     *
     * @param coeffs The list of coefficients.
     * @param obs The list of observables IDs.
     *
     * @return `ObsIdType` ID of the constructed observable.
     */
    virtual auto HamiltonianObservable(const std::vector<double> &coeffs,
                                       const std::vector<ObsIdType> &obs) -> ObsIdType
    {
        RT_FAIL("HamiltonianObservable is unsupported by device");
    }

    // ----------------------------------------
    //  MEASUREMENT PROCESSES
    // ----------------------------------------

    /**
     * @brief (Optional) Compute raw samples on all qubits.
     *
     * Perform measurement sampling in the computational basis on all qubits. The number of samples
     * to take is the current active shot count in the device (see also `SetDeviceShots`). When
     * possible, the result should be produced without affecting the internal quantum state.
     *
     * The samples are taken individually on each qubit and stored in a 2D array with the shots
     * being the outer dimension and the qubits being the inner dimension. For example, a 4-shot
     * 2-qubit sample call might produce the following result:
     *    [[0, 1],
     *     [0, 0],
     *     [1, 1],
     *     [0, 0]]
     * The samples are stored as double-precision floating-point numbers; for computational basis
     * measurements this might be overkill, but allows for expanding this operation in the future
     * to sample arbitrary observables and produce eigenvalues as sample results.
     *
     * The result of this operation must be written into the `samples` argument buffer.
     *
     * @param samples The pre-allocated buffer for the measurement samples.
     */
    virtual void Sample(DataView<double, 2> &samples)
    {
        RT_FAIL("Sample is unsupported by device");
    }

    /**
     * @brief (Optional) Compute raw samples for a quantum subsystem.
     *
     * Like `Sample`, but for a subset of currently allocated qubits.
     *
     * @param samples The pre-allocated buffer for the measurement samples.
     * @param wires Qubits to compute samples for.
     */
    virtual void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires)
    {
        RT_FAIL("PartialSample is unsupported by device");
    }

    /**
     * @brief (Optional) Compute the sample counts on all qubits.
     *
     * Perform measurement sampling in the computational basis and sum up the occurrence of each
     * outcome. All currently allocated qubits should be sampled. The number of samples to take
     * is the current active shot count in the device (see also `SetDeviceShots`). When possible,
     * the result should be produced without affecting the internal quantum state.
     *
     * The potential measurement outcomes are returned as `eigvals`; currently the only
     * supported mode uses computational basis states whose bitstring representation "01010110..."
     * is first taken as an integer value, and then converted to a double-precision floating-point
     * number. This effectively limits the number of qubits supported by this operation to 53, after
     * which the basis states cannot be accurately represented by the floating-point format. In the
     * future, this operation may be expanded to support eigenvalues of arbitrary observables as
     * possible measurement outcomes, hence the floating-point datatype.
     *
     * The number of times of each measurement outcome is sampled is stored as a 64-bit integer in
     * the same location in the `counts` array as the corresponding basis state in the `eigvals`
     * array. The entries need not be in any particular order, since the measured states are
     * returned alongside the counts.
     *
     * In effect, the result of this operation is a dictionary from measurement outcomes to counts,
     * stored as a pair of dense 1D arrays, one for the keys and one for the values. It's important
     * that all 2^n elements of the `eigvals` and `counts` arrays are written to, where n is the
     * number of qubits, else the result will contain uninitialized data.
     *
     * The results of this operation must be written into the `eigvals` and `counts` argument
     * buffers.
     *
     * @param eigvals The pre-allocated buffer for all measured states.
     * @param counts The pre-allocated buffer for all measured counts.
     */
    virtual void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts)
    {
        RT_FAIL("Counts is unsupported by device");
    }

    /**
     * @brief (Optional) Compute the sample counts for a quantum subsystem.
     *
     * Like `Counts`, but for a subset of currently allocated qubits.
     *
     * @param eigvals The pre-allocated buffer for all measured states.
     * @param counts The pre-allocated buffer for all measured counts.
     * @param wires Qubits to compute sample counts for.
     */
    virtual void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                               const std::vector<QubitIdType> &wires)
    {
        RT_FAIL("PartialCounts is unsupported by device");
    }

    /**
     * @brief (Optional) Compute measurement probabilities on all qubits.
     *
     * The output of this operation is the probability distribution across computational basis
     * states for all currently allocated qubits. When possible, the result should be produced
     * without affecting the internal quantum state.
     *
     * The result of this operation must be written into the `probs` argument buffer.
     *
     * @param probs The pre-allocated buffer for the probabilities.
     */
    virtual void Probs(DataView<double, 1> &probs) { RT_FAIL("Probs is unsupported by device"); }

    /**
     * @brief (Optional) Compute measurement probabilities for a quantum subsystem.
     *
     * Like `Probs`, but for a subset of currently allocated qubits.
     *
     * @param probs The pre-allocated buffer for the probabilities.
     * @param wires Qubits to compute probabilities for.
     */
    virtual void PartialProbs(DataView<double, 1> &probs, const std::vector<QubitIdType> &wires)
    {
        RT_FAIL("PartialProbs is unsupported by device");
    }
    /**
     * @brief (Optional) Compute the expected value of an observable.
     *
     * The output of this operation is expectation value ⟨O⟩ of an observable O with respect to the
     * current quantum state. When possible, the result should be produced without affecting the
     * internal quantum state.
     *
     * See also `Observable` for observable semantics.
     *
     * @param obsKey The ID of the constructed observable.
     *
     * @return `double` The expectation value of the observable.
     */
    virtual auto Expval(ObsIdType obsKey) -> double { RT_FAIL("Expval is unsupported by device"); }

    /**
     * @brief (Optional) Compute the variance of an observable.
     *
     * The output of this operation is variance ⟨O²⟩−⟨O⟩² of an observable O with respect to the
     * current quantum state. When possible, the result should be produced without affecting the
     * internal quantum state.
     *
     * See also `Observable` for observable semantics.
     *
     * @param obsKey The ID of the constructed observable.
     *
     * @return `double` The variance of the observable.
     */
    virtual auto Var(ObsIdType obsKey) -> double { RT_FAIL("Var is unsupported by device"); }

    /**
     * @brief (Optional) Get the full quantum state of all qubits.
     *
     * Typically for devices with statevector simulation capabilities.
     *
     * The output of this operation is the quantum statevector in the computational basis
     * on all currently allocated qubits. When possible, the result should be produced without
     * affecting the internal quantum state.
     *
     * The result of this operation must be written into the `state` argument buffer.
     *
     * @param state Pre-allocated buffer for the quantum state.
     */
    virtual void State(DataView<std::complex<double>, 1> &state)
    {
        RT_FAIL("State is unsupported by device");
    }

    // ----------------------------------------
    //  QUANTUM DERIVATIVES
    // ----------------------------------------

    /**
     * @brief (Optional) Compute the gradient/Jacobian of a quantum circuit.
     *
     * For devices with internal differentiation capabilities (e.g. simulator with reverse-mode AD).
     *
     * Performing differentiation inside a device can be advantageous for performance compared to
     * device-agnostic methods like the Parameter-Shift technique. If supported, Catalyst will
     * communicate the start and end of the "forward pass" via the `StartTapeRecording` and
     * `StopTapeRecording` functions. The `Gradient` call then requests the "reverse pass" with
     * respect to the recorded circuit and observables. A sample implementation of circuit
     * recording is available in `CacheManager.hpp`.
     *
     * The output is an array of gradients (i.e. the Jacobian), one for each expectation value /
     * observable in the recorded program. The length of the gradient is determined by the number
     * of gate parameters in the recorded circuit, or by the optional `trainParams` argument. If
     * provided, `trainParams` can customize which parameter values participate in differentiation.
     *
     * The result of this operation must be written into the `gradients` argument buffer.
     *
     * @param gradients The vector of pre-allocated `DataView<double, 1>*`
     *                  to store the flattened Jacobian (one gradient per cached observable).
     * @param trainParams The vector of trainable parameters; if empty, all parameters
     *                    would be assumed trainable.
     */
    virtual void Gradient(std::vector<DataView<double, 1>> &gradients,
                          const std::vector<size_t> &trainParams)
    {
        RT_FAIL("Differentiation is unsupported by device");
    }

    /**
     * @brief (Optional) Start recording a quantum tape if provided.
     *
     * See `Gradient` for additional information.
     */
    virtual void StartTapeRecording() { RT_FAIL("Differentiation is unsupported by device"); }

    /**
     * @brief (Optional) Stop recording a quantum tape if provided.
     *
     * See `Gradient` for additional information.
     */
    virtual void StopTapeRecording() { RT_FAIL("Differentiation is unsupported by device"); }
};

} // namespace Catalyst::Runtime
