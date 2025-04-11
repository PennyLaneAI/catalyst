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
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "DataView.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "Types.h"
#include "Utils.hpp"

// The delimited to print before and after the JSON representing resource usage.
//   This is intended to make parsing the output easier.
//   Ideally, at some point resource usage would be passed back through as an object, and printing
//   at all would be unneeded.
#define RESOURCE_PRINT_DELIMETER "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="

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
        auto device_kwargs = Catalyst::Runtime::parse_kwargs(kwargs);
        if (device_kwargs.find("track_resources") != device_kwargs.end()) {
            track_resources_ = device_kwargs["track_resources"] == "True";
        }
    }
    ~NullQubit()
    {
        if (this->track_resources_) {
            PrintResourceUsage();
        }
    } // LCOV_EXCL_LINE

    NullQubit &operator=(const NullQubit &) = delete;
    NullQubit(const NullQubit &) = delete;
    NullQubit(NullQubit &&) = delete;
    NullQubit &operator=(NullQubit &&) = delete;

    /**
     * @brief Prints resources that would be used to execute this circuit as a JSON
     */
    void PrintResourceUsage()
    {
        std::cout << RESOURCE_PRINT_DELIMETER << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "  \"num_qubits\": " << resource_data_["num_qubits"] << ",\n";
        std::cout << "  \"num_gates\": " << resource_data_["num_gates"] << ",\n";
        std::cout << "  \"gate_types\": ";
        resource_data_.erase("num_gates");
        resource_data_.erase("num_qubits");
        pretty_print(resource_data_, 2);
        std::cout << "\n}" << std::endl;
        std::cout << RESOURCE_PRINT_DELIMETER << std::endl;
    }

    /**
     * @brief Doesn't Allocate a qubit.
     *
     * @return `QubitIdType`
     */
    auto AllocateQubit() -> QubitIdType
    {
        num_qubits_++; // next_id
        if (this->track_resources_) {
            // Store the highest number of qubits allocated at any time since device creation
            resource_data_["num_qubits"] = std::max(num_qubits_, resource_data_["num_qubits"]);
        }
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
        if (!num_qubits) {
            return {};
        }
        std::vector<QubitIdType> result(num_qubits);
        std::generate_n(result.begin(), num_qubits, [this]() { return AllocateQubit(); });
        return result;
    }

    /**
     * @brief Doesn't Release a qubit.
     */
    void ReleaseQubit(QubitIdType q)
    {
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
        if (this->track_resources_) {
            resource_data_["num_gates"]++;
            resource_data_[name]++;
        }
    }

    /**
     * @brief Doesn't Apply a given matrix directly to the state vector of a device.
     *
     */
    void MatrixOperation(const std::vector<std::complex<double>> &,
                         const std::vector<QubitIdType> &, bool,
                         const std::vector<QubitIdType> &controlled_wires = {},
                         const std::vector<bool> &controlled_values = {})
    {
        if (this->track_resources_) {
            resource_data_["num_gates"]++;
            resource_data_["QubitUnitary"]++;
        }
    }

    /**
     * @brief Doesn't Construct a named (Identity, PauliX, PauliY, PauliZ, and Hadamard)
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
     * @brief Doesn't Construct a tensor product of observables.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto TensorObservable(const std::vector<ObsIdType> &) -> ObsIdType { return 0.0; }

    /**
     * @brief Doesn't Construct a Hamiltonian observable.
     *
     * @return `ObsIdType` Index of the constructed observable
     */
    auto HamiltonianObservable(const std::vector<double> &, const std::vector<ObsIdType> &)
        -> ObsIdType
    {
        return 0.0;
    }

    /**
     * @brief Doesn't Compute the expected value of an observable.
     *
     * @return `double` The expected value
     */
    auto Expval(ObsIdType) -> double { return 0.0; }

    /**
     * @brief Doesn't Compute the variance of an observable.
     *
     * @return `double` The variance
     */
    auto Var(ObsIdType) -> double { return 0.0; }

    /**
     * @brief Doesn't Get the state-vector of a device.
     */
    void State(DataView<std::complex<double>, 1> &) {}

    /**
     * @brief Doesn't Compute the probabilities of each computational basis state.
     */
    void Probs(DataView<double, 1> &) {}

    /**
     * @brief Doesn't Compute the probabilities for a subset of the full system.
     */
    void PartialProbs(DataView<double, 1> &, const std::vector<QubitIdType> &) {}

    /**
     * @brief Doesn't Compute samples with the number of shots on the entire wires,
     * returing raw samples.
     */
    void Sample(DataView<double, 2> &, size_t) {}

    /**
     * @brief Doesn't Compute partial samples with the number of shots on `wires`,
     * returing raw samples.
     *
     * @param samples The pre-allocated `DataView<double, 2>`representing a matrix of
     * shape `shots * numWires`. The built-in iterator in `DataView<double, 2>`
     * iterates over all elements of `samples` row-wise.
     */
    void PartialSample(DataView<double, 2> &, const std::vector<QubitIdType> &, size_t) {}

    /**
     * @brief Doesn't Sample with the number of shots on the entire wires, returning the
     * number of counts for each sample.
     */
    void Counts(DataView<double, 1> &, DataView<int64_t, 1> &, size_t) {}

    /**
     * @brief Doesn't Partial sample with the number of shots on `wires`, returning the
     * number of counts for each sample.
     */
    void PartialCounts(DataView<double, 1> &, DataView<int64_t, 1> &,
                       const std::vector<QubitIdType> &, size_t)
    {
    }

    /**
     * @brief This is not A general measurement method that acts on a single wire.
     *
     * @return `Result` The measurement result
     */
    auto Measure(QubitIdType, std::optional<int32_t>) -> Result
    {
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
    bool track_resources_{false};
    std::size_t num_qubits_{0};
    std::unordered_map<std::string, size_t> resource_data_;
    Catalyst::Runtime::QubitManager<QubitIdType, size_t> qubit_manager{};
};
} // namespace Catalyst::Runtime::Devices
