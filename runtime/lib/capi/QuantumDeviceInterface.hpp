// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include "DataView.hpp"
#include "Types.h"

/**
 * @brief The namespace of `QuantumDeviceInterface`
 *
 * This splits the QuantumDevice API from device implementations.
 * It is used in the creation and execution of multiple devices
 * asynchronously via the device pool in `ExecutionContext`.
 *
 * @param device A `QuantumDevice` pointer of the device implementation
 */
namespace Catalyst::Runtime::QuantumDeviceInterface {

// TODO: Remove lcov skips after completing the end-to-end async execution support
// LCOV_EXCL_START
auto AllocateQubit(QuantumDevice *device) -> QubitIdType { return device->AllocateQubit(); }

auto AllocateQubits(QuantumDevice *device, size_t num_qubits) -> std::vector<QubitIdType>
{
    return device->AllocateQubits(num_qubits);
}

void ReleaseQubit(QuantumDevice *device, QubitIdType qubit) { device->ReleaseQubit(qubit); }

void ReleaseAllQubits(QuantumDevice *device) { device->ReleaseAllQubits(); }

[[nodiscard]] auto GetNumQubits(QuantumDevice *device) -> size_t { return device->GetNumQubits(); }

void SetDeviceShots(QuantumDevice *device, size_t shots) { device->SetDeviceShots(shots); }

auto GetDeviceShots(QuantumDevice *device) -> size_t { return device->GetDeviceShots(); }

void StartTapeRecording(QuantumDevice *device) { device->StartTapeRecording(); }

void StopTapeRecording(QuantumDevice *device) { device->StopTapeRecording(); }

[[nodiscard]] auto Zero(QuantumDevice *device) -> Result { return device->Zero(); }

[[nodiscard]] auto One(QuantumDevice *device) -> Result { return device->One(); }

void PrintState(QuantumDevice *device) { device->PrintState(); }

void NamedOperation(QuantumDevice *device, const std::string &name,
                    const std::vector<double> &params, const std::vector<QubitIdType> &wires,
                    bool inverse)
{
    device->NamedOperation(name, params, wires, inverse);
}

void MatrixOperation(QuantumDevice *device, const std::vector<std::complex<double>> &matrix,
                     const std::vector<QubitIdType> &wires, bool inverse)
{
    device->MatrixOperation(matrix, wires, inverse);
}

auto Observable(QuantumDevice *device, ObsId id, const std::vector<std::complex<double>> &matrix,
                const std::vector<QubitIdType> &wires) -> ObsIdType
{
    return device->Observable(id, matrix, wires);
}

auto TensorObservable(QuantumDevice *device, const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return device->TensorObservable(obs);
}

auto HamiltonianObservable(QuantumDevice *device, const std::vector<double> &coeffs,
                           const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return device->HamiltonianObservable(coeffs, obs);
}

auto Expval(QuantumDevice *device, ObsIdType obsKey) -> double { return device->Expval(obsKey); }

auto Var(QuantumDevice *device, ObsIdType obsKey) -> double { return device->Var(obsKey); }

void State(QuantumDevice *device, DataView<std::complex<double>, 1> &state)
{
    device->State(state);
}

void Probs(QuantumDevice *device, DataView<double, 1> &probs) { device->Probs(probs); }

void PartialProbs(QuantumDevice *device, DataView<double, 1> &probs,
                  const std::vector<QubitIdType> &wires)
{
    device->PartialProbs(probs, wires);
}

void Sample(QuantumDevice *device, DataView<double, 2> &samples, size_t shots)
{
    device->Sample(samples, shots);
}

void PartialSample(QuantumDevice *device, DataView<double, 2> &samples,
                   const std::vector<QubitIdType> &wires, size_t shots)
{
    device->PartialSample(samples, wires, shots);
}

void Counts(QuantumDevice *device, DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
            size_t shots)
{
    device->Counts(eigvals, counts, shots);
}

void PartialCounts(QuantumDevice *device, DataView<double, 1> &eigvals,
                   DataView<int64_t, 1> &counts, const std::vector<QubitIdType> &wires,
                   size_t shots)
{
    device->PartialCounts(eigvals, counts, wires, shots);
}

auto Measure(QuantumDevice *device, QubitIdType wire) -> Result { return device->Measure(wire); }

void Gradient(QuantumDevice *device, std::vector<DataView<double, 1>> &gradients,
              const std::vector<size_t> &trainParams)
{
    device->Gradient(gradients, trainParams);
}
// LCOV_EXCL_STOP

} // namespace Catalyst::Runtime::QuantumDeviceInterface
