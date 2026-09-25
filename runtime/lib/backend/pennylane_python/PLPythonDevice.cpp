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

#include "PLPythonDevice.hpp"

#include <algorithm>
#include <tuple>

#include "Exception.hpp"

namespace Catalyst::Runtime::Device {

using PLPython::Payload;

namespace {
auto obsName(ObsId id) -> const char *
{
    switch (id) {
    case ObsId::Identity:
        return "Identity";
    case ObsId::PauliX:
        return "PauliX";
    case ObsId::PauliY:
        return "PauliY";
    case ObsId::PauliZ:
        return "PauliZ";
    case ObsId::Hadamard:
        return "Hadamard";
    case ObsId::Hermitian:
        return "Hermitian";
    }
    RT_FAIL("Unsupported observable");
}
} // namespace

PLPythonDevice::PLPythonDevice(const std::string &kwargs)
    : runner(std::make_unique<PLPython::PLPythonRunner>()), device_kwargs(kwargs)
{
}

auto PLPythonDevice::getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<size_t>
{
    std::vector<size_t> res;
    res.reserve(wires.size());
    std::transform(wires.begin(), wires.end(), std::back_inserter(res),
                   [this](auto w) { return qubit_manager.getDeviceId(w); });
    return res;
}

auto PLPythonDevice::call(const std::string &method, Payload &payload) -> std::vector<double>
{
    payload.add("session", session);
    return runner->Call(method, payload.str());
}

auto PLPythonDevice::AllocateQubits(size_t num) -> std::vector<QubitIdType>
{
    if (!num) {
        return {};
    }
    RT_FAIL_IF(num_qubits, "Partial qubit allocation is not supported by PLPythonDevice");

    // A new execution of the QNode starts: open a new tape on the Python side.
    auto &&ids = runner->Call("init", Payload().add("kwargs", device_kwargs).str());
    session = static_cast<size_t>(ids.at(0));
    num_qubits = num;

    Payload shots;
    shots.add("shots", device_shots);
    std::ignore = call("shots", shots);
    Payload alloc;
    alloc.add("num_qubits", num);
    std::ignore = call("allocate", alloc);

    return qubit_manager.AllocateRange(0, num);
}

void PLPythonDevice::ReleaseQubits(const std::vector<QubitIdType> &qubits)
{
    RT_FAIL_IF(qubits.size() != num_qubits,
               "PLPythonDevice does not support dynamic qubit allocation");
    qubit_manager.ReleaseAll();
    num_qubits = 0;
    mcm_results.clear();

    Payload payload;
    std::ignore = call("release", payload);
}

auto PLPythonDevice::GetNumQubits() const -> size_t { return num_qubits; }

void PLPythonDevice::SetDeviceShots(size_t shots)
{
    device_shots = shots;
    if (num_qubits) {
        Payload payload;
        payload.add("shots", shots);
        std::ignore = call("shots", payload);
    }
}

auto PLPythonDevice::GetDeviceShots() const -> size_t { return device_shots; }

void PLPythonDevice::NamedOperation(const std::string &name, const std::vector<double> &params,
                                    const std::vector<QubitIdType> &wires, bool inverse,
                                    const std::vector<QubitIdType> &controlled_wires,
                                    const std::vector<bool> &controlled_values,
                                    const std::vector<std::string> &optional_params)
{
    Payload payload;
    payload.add("name", name)
        .add("params", params)
        .add("optional_params", optional_params)
        .add("wires", getDeviceWires(wires))
        .add("adjoint", inverse)
        .add("ctrl_wires", getDeviceWires(controlled_wires))
        .add("ctrl_values", controlled_values);
    std::ignore = call("gate", payload);
}

void PLPythonDevice::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                     const std::vector<QubitIdType> &wires, bool inverse,
                                     const std::vector<QubitIdType> &controlled_wires,
                                     const std::vector<bool> &controlled_values)
{
    Payload payload;
    payload.addComplex("re", "im", matrix)
        .add("wires", getDeviceWires(wires))
        .add("adjoint", inverse)
        .add("ctrl_wires", getDeviceWires(controlled_wires))
        .add("ctrl_values", controlled_values);
    std::ignore = call("matrix", payload);
}

void PLPythonDevice::SetState(DataView<std::complex<double>, 1> &state,
                              std::vector<QubitIdType> &wires)
{
    std::vector<std::complex<double>> values(state.begin(), state.end());
    Payload payload;
    payload.addComplex("re", "im", values).add("wires", getDeviceWires(wires));
    std::ignore = call("set_state", payload);
}

void PLPythonDevice::SetBasisState(DataView<int8_t, 1> &n, std::vector<QubitIdType> &wires)
{
    std::vector<int> values(n.begin(), n.end());
    Payload payload;
    payload.add("state", values).add("wires", getDeviceWires(wires));
    std::ignore = call("set_basis_state", payload);
}

auto PLPythonDevice::Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result
{
    Payload payload;
    payload.add("wire", getDeviceWires({wire})[0]);
    if (postselect.has_value()) {
        payload.add("postselect", static_cast<size_t>(postselect.value()));
    }
    std::ignore = call("measure", payload);

    // The outcome is only known once the device executes the tape: it is recorded as a
    // mid-circuit measurement, and the program receives a placeholder.
    mcm_results.push_back(false);
    return &mcm_results.back();
}

auto PLPythonDevice::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                const std::vector<QubitIdType> &wires) -> ObsIdType
{
    Payload payload;
    payload.add("kind", obsName(id)).add("wires", getDeviceWires(wires));
    if (id == ObsId::Hermitian) {
        payload.addComplex("re", "im", matrix);
    }
    return static_cast<ObsIdType>(call("observable", payload).at(0));
}

auto PLPythonDevice::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
{
    Payload payload;
    payload.add("obs", obs);
    return static_cast<ObsIdType>(call("tensor", payload).at(0));
}

auto PLPythonDevice::HamiltonianObservable(const std::vector<double> &coeffs,
                                           const std::vector<ObsIdType> &obs) -> ObsIdType
{
    Payload payload;
    payload.add("coeffs", coeffs).add("obs", obs);
    return static_cast<ObsIdType>(call("hamiltonian", payload).at(0));
}

auto PLPythonDevice::measurement(const std::string &method, Payload &&payload,
                                 size_t expected_size) -> std::vector<double>
{
    auto &&result = call(method, payload);
    RT_FAIL_IF(result.size() != expected_size,
               ("The Python device returned a result of unexpected size for " + method).c_str());
    return result;
}

auto PLPythonDevice::Expval(ObsIdType obsKey) -> double
{
    Payload payload;
    payload.add("obs", static_cast<size_t>(obsKey));
    return measurement("expval", std::move(payload), 1)[0];
}

auto PLPythonDevice::Var(ObsIdType obsKey) -> double
{
    Payload payload;
    payload.add("obs", static_cast<size_t>(obsKey));
    return measurement("var", std::move(payload), 1)[0];
}

void PLPythonDevice::Probs(DataView<double, 1> &probs)
{
    Payload payload;
    payload.addNull("wires");
    auto &&result = measurement("probs", std::move(payload), probs.size());
    std::copy(result.begin(), result.end(), probs.begin());
}

void PLPythonDevice::PartialProbs(DataView<double, 1> &probs,
                                  const std::vector<QubitIdType> &wires)
{
    Payload payload;
    payload.add("wires", getDeviceWires(wires));
    auto &&result = measurement("probs", std::move(payload), probs.size());
    std::copy(result.begin(), result.end(), probs.begin());
}

void PLPythonDevice::Sample(DataView<double, 2> &samples)
{
    Payload payload;
    payload.addNull("wires");
    auto &&result = measurement("sample", std::move(payload), samples.size());
    std::copy(result.begin(), result.end(), samples.begin());
}

void PLPythonDevice::PartialSample(DataView<double, 2> &samples,
                                   const std::vector<QubitIdType> &wires)
{
    Payload payload;
    payload.add("wires", getDeviceWires(wires));
    auto &&result = measurement("sample", std::move(payload), samples.size());
    std::copy(result.begin(), result.end(), samples.begin());
}

void PLPythonDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts)
{
    Payload payload;
    payload.addNull("wires");
    auto &&result = measurement("counts", std::move(payload), eigvals.size() + counts.size());
    std::copy(result.begin(), result.begin() + eigvals.size(), eigvals.begin());
    std::transform(result.begin() + eigvals.size(), result.end(), counts.begin(),
                   [](double c) { return static_cast<int64_t>(c); });
}

void PLPythonDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                   const std::vector<QubitIdType> &wires)
{
    Payload payload;
    payload.add("wires", getDeviceWires(wires));
    auto &&result = measurement("counts", std::move(payload), eigvals.size() + counts.size());
    std::copy(result.begin(), result.begin() + eigvals.size(), eigvals.begin());
    std::transform(result.begin() + eigvals.size(), result.end(), counts.begin(),
                   [](double c) { return static_cast<int64_t>(c); });
}

void PLPythonDevice::State(DataView<std::complex<double>, 1> &state)
{
    Payload payload;
    auto &&result = measurement("state", std::move(payload), 2 * state.size());
    auto it = state.begin();
    for (size_t i = 0; i < result.size(); i += 2) {
        *(it++) = std::complex<double>(result[i], result[i + 1]);
    }
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(PLPythonDevice, Catalyst::Runtime::Device::PLPythonDevice);
