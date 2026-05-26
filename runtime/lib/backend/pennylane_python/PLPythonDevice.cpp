// Copyright 2024 Xanadu Quantum Technologies Inc.

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

#include <bitset>
#include <sstream>

#include "Exception.hpp"

namespace Catalyst::Runtime::Device {

auto PLPythonDevice::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (!num_qubits) {
        return {};
    }

    const size_t cur_num_qubits = builder->getNumQubits();
    RT_FAIL_IF(cur_num_qubits,
               "Partial qubits allocation is not supported by PLPythonDevice");

    builder->setNumQubits(num_qubits);

    std::vector<QubitIdType> result = qubit_manager.AllocateRange(0, num_qubits);

    RT_FAIL_IF(!this->initial_allocated_QubitIds.empty(),
               "PLPythonDevice does not support dynamic qubit allocation")
    this->initial_allocated_QubitIds.insert(result.begin(), result.end());
    return result;
}

void PLPythonDevice::ReleaseQubits(const std::vector<QubitIdType> &qubits)
{
    std::set<QubitIdType> dealloc_Ids(qubits.begin(), qubits.end());
    RT_FAIL_IF(this->initial_allocated_QubitIds != dealloc_Ids,
               "PLPythonDevice does not support dynamic qubit allocation. Please ensure the "
               "deallocation qubit ID array contains the same values as those produced by the "
               "initial `AllocateQubits` call")
    this->initial_allocated_QubitIds.clear();

    // Refresh the builder for device re-use.
    builder = std::make_unique<PLTape::PLTapeBuilder>();
}

auto PLPythonDevice::GetNumQubits() const -> size_t { return builder->getNumQubits(); }

void PLPythonDevice::SetDeviceShots(size_t shots) { device_shots = shots; }

auto PLPythonDevice::GetDeviceShots() const -> size_t { return device_shots; }

void PLPythonDevice::NamedOperation(
    const std::string &name, const std::vector<double> &params,
    const std::vector<QubitIdType> &wires, bool inverse,
    const std::vector<QubitIdType> &controlled_wires, const std::vector<bool> &controlled_values,
    [[maybe_unused]] const std::vector<std::string> &optional_params)
{
    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);

    std::vector<size_t> dev_ctrl_wires;
    if (!controlled_wires.empty()) {
        dev_ctrl_wires = getDeviceWires(controlled_wires);
    }

    builder->Gate(name, params, dev_wires, inverse, dev_ctrl_wires, controlled_values);
}

void PLPythonDevice::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                     const std::vector<QubitIdType> &wires, bool inverse,
                                     const std::vector<QubitIdType> &controlled_wires,
                                     const std::vector<bool> &controlled_values)
{
    auto &&dev_wires = getDeviceWires(wires);

    std::vector<size_t> dev_ctrl_wires;
    if (!controlled_wires.empty()) {
        dev_ctrl_wires = getDeviceWires(controlled_wires);
    }

    builder->MatrixGate(matrix, dev_wires, inverse, dev_ctrl_wires, controlled_values);
}

auto PLPythonDevice::Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result
{
    RT_FAIL_IF(postselect.has_value(), "Post-selection is not supported yet");

    auto &&dev_wire = getDeviceWires({wire});
    builder->Measure(dev_wire[0]);
    return Result{};
}

auto PLPythonDevice::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                const std::vector<QubitIdType> &wires) -> ObsIdType
{
    RT_FAIL_IF(wires.size() > GetNumQubits(), "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);

    if (id == ObsId::Hermitian) {
        return obs_manager.createHermitianObs(matrix, dev_wires);
    }

    return obs_manager.createNamedObs(id, dev_wires);
}

auto PLPythonDevice::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return obs_manager.createTensorProdObs(obs);
}

auto PLPythonDevice::HamiltonianObservable(const std::vector<double> &coeffs,
                                           const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return obs_manager.createHamiltonianObs(coeffs, obs);
}

// Helper to get kwargs JSON string for the Python module
static auto serializeKwargs(const std::unordered_map<std::string, std::string> &kwargs)
    -> std::string
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto &[key, val] : kwargs) {
        if (!first) oss << ",";
        oss << "\"" << key << "\":\"" << val << "\"";
        first = false;
    }
    oss << "}";
    return oss.str();
}

auto PLPythonDevice::Expval(ObsIdType obsKey) -> double
{
    RT_FAIL_IF(!obs_manager.isValidObservable(obsKey), "Invalid key for cached observables");

    std::string meas_json = "{\"type\":\"expval\",\"obs_idx\":" + std::to_string(obsKey) + "}";

    auto &&res = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                 serializeKwargs(device_kwargs), device_shots);
    return res[0];
}

auto PLPythonDevice::Var(ObsIdType obsKey) -> double
{
    RT_FAIL_IF(!obs_manager.isValidObservable(obsKey), "Invalid key for cached observables");

    std::string meas_json = "{\"type\":\"var\",\"obs_idx\":" + std::to_string(obsKey) + "}";

    auto &&res = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                 serializeKwargs(device_kwargs), device_shots);
    return res[0];
}

void PLPythonDevice::Probs(DataView<double, 1> &probs)
{
    std::string meas_json = "{\"type\":\"probs\"}";
    auto &&dv_probs = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                      serializeKwargs(device_kwargs), device_shots);

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");
    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void PLPythonDevice::PartialProbs(DataView<double, 1> &probs,
                                  const std::vector<QubitIdType> &wires)
{
    auto &&dev_wires = getDeviceWires(wires);

    std::ostringstream w_oss;
    w_oss << "[";
    for(size_t i = 0; i < dev_wires.size(); ++i) {
        if(i > 0) w_oss << ",";
        w_oss << dev_wires[i];
    }
    w_oss << "]";
    std::string meas_json = "{\"type\":\"probs\",\"wires\":" + w_oss.str() + "}";

    auto &&dv_probs = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                      serializeKwargs(device_kwargs), device_shots);

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");
    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void PLPythonDevice::Sample(DataView<double, 2> &samples)
{
    std::string meas_json = "{\"type\":\"sample\"}";
    auto &&li_samples = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                        serializeKwargs(device_kwargs), device_shots);

    RT_FAIL_IF(samples.size() != li_samples.size(), "Invalid size for the pre-allocated samples");

    const size_t numQubits = GetNumQubits();
    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < device_shots; shot++) {
        for (size_t wire = 0; wire < numQubits; wire++) {
            *(samplesIter++) = li_samples[shot * numQubits + wire];
        }
    }
}

void PLPythonDevice::PartialSample(DataView<double, 2> &samples,
                                   const std::vector<QubitIdType> &wires)
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF(samples.size() != device_shots * numWires,
               "Invalid size for the pre-allocated partial-samples");

    auto &&dev_wires = getDeviceWires(wires);

    std::ostringstream w_oss;
    w_oss << "[";
    for(size_t i = 0; i < dev_wires.size(); ++i) {
        if(i > 0) w_oss << ",";
        w_oss << dev_wires[i];
    }
    w_oss << "]";
    std::string meas_json = "{\"type\":\"sample\",\"wires\":" + w_oss.str() + "}";

    // PennyLane will now only return samples for the target wires
    auto &&li_samples = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                        serializeKwargs(device_kwargs), device_shots);

    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < device_shots; shot++) {
        for (size_t w_idx = 0; w_idx < dev_wires.size(); w_idx++) {
            *(samplesIter++) = li_samples[shot * dev_wires.size() + w_idx];
        }
    }
}

void PLPythonDevice::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts)
{
    const size_t numQubits = GetNumQubits();
    const size_t numElements = 1U << numQubits;

    RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
               "Invalid size for the pre-allocated counts");

    std::string meas_json = "{\"type\":\"sample\"}";
    auto &&li_samples = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                        serializeKwargs(device_kwargs), device_shots);

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < device_shots; shot++) {
        std::bitset<52> basisState;
        size_t idx = numQubits;
        for (size_t wire = 0; wire < numQubits; wire++) {
            basisState[--idx] = static_cast<size_t>(li_samples[shot * numQubits + wire]);
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

void PLPythonDevice::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                   const std::vector<QubitIdType> &wires)
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();
    const size_t numElements = 1U << numWires;

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF((eigvals.size() != numElements || counts.size() != numElements),
               "Invalid size for the pre-allocated partial-counts");

    auto &&dev_wires = getDeviceWires(wires);

    std::ostringstream w_oss;
    w_oss << "[";
    for(size_t i = 0; i < dev_wires.size(); ++i) {
        if(i > 0) w_oss << ",";
        w_oss << dev_wires[i];
    }
    w_oss << "]";
    std::string meas_json = "{\"type\":\"sample\",\"wires\":" + w_oss.str() + "}";

    auto &&li_samples = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                        serializeKwargs(device_kwargs), device_shots);

    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t shot = 0; shot < device_shots; shot++) {
        std::bitset<52> basisState;
        size_t idx = dev_wires.size();
        for (size_t w_idx = 0; w_idx < dev_wires.size(); w_idx++) {
            basisState[--idx] = static_cast<size_t>(li_samples[shot * dev_wires.size() + w_idx]);
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

void PLPythonDevice::State(DataView<std::complex<double>, 1> &state)
{
    std::string meas_json = "{\"type\":\"state\"}";
    auto &&dv_state = runner->Execute(builder->toJSON(), obs_manager.toJSON(), meas_json,
                                      serializeKwargs(device_kwargs), device_shots);

    RT_FAIL_IF(state.size() * 2 != dv_state.size(), "Invalid size for the pre-allocated state vector");

    auto stateIter = state.begin();
    for (size_t i = 0; i < dv_state.size(); i += 2) {
        *(stateIter++) = std::complex<double>(dv_state[i], dv_state[i+1]);
    }
}

} // namespace Catalyst::Runtime::Device

GENERATE_DEVICE_FACTORY(PLPythonDevice, Catalyst::Runtime::Device::PLPythonDevice);
