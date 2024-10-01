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

#include "LightningSimulator.hpp"

#include "AdjointJacobianLQubit.hpp"
#include "JacobianData.hpp"
#include "LinearAlgebra.hpp"
#include "MeasurementsLQubit.hpp"

namespace Catalyst::Runtime::Simulator {

auto LightningSimulator::AllocateQubit() -> QubitIdType
{
    size_t sv_id = this->device_sv->allocateWire();
    return this->qubit_manager.Allocate(sv_id);
}

auto LightningSimulator::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (!num_qubits) {
        return {};
    }

    // at the first call when num_qubits == 0
    if (!this->GetNumQubits()) {
        this->device_sv = std::make_unique<StateVectorT>(num_qubits);
        return this->qubit_manager.AllocateRange(0, num_qubits);
    }

    std::vector<QubitIdType> result(num_qubits);
    std::generate_n(result.begin(), num_qubits, [this]() { return AllocateQubit(); });
    return result;
}

void LightningSimulator::ReleaseAllQubits()
{
    this->device_sv->clearData();
    this->qubit_manager.ReleaseAll();
}

void LightningSimulator::ReleaseQubit(QubitIdType q)
{
    if (this->qubit_manager.isValidQubitId(q)) {
        this->device_sv->releaseWire(this->qubit_manager.getDeviceId(q));
    }
    this->qubit_manager.Release(q);
}

auto LightningSimulator::GetNumQubits() const -> size_t { return this->device_sv->getNumQubits(); }

void LightningSimulator::StartTapeRecording()
{
    RT_FAIL_IF(this->tape_recording, "Cannot re-activate the cache manager");
    this->tape_recording = true;
    this->cache_manager.Reset();
}

void LightningSimulator::StopTapeRecording()
{
    RT_FAIL_IF(!this->tape_recording, "Cannot stop an already stopped cache manager");
    this->tape_recording = false;
}

auto LightningSimulator::CacheManagerInfo()
    -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>
{
    return {this->cache_manager.getNumOperations(), this->cache_manager.getNumObservables(),
            this->cache_manager.getNumParams(), this->cache_manager.getOperationsNames(),
            this->cache_manager.getObservablesKeys()};
}

void LightningSimulator::SetDeviceShots(size_t shots) { this->device_shots = shots; }

auto LightningSimulator::GetDeviceShots() const -> size_t { return this->device_shots; }

void LightningSimulator::SetDevicePRNG(std::mt19937 *gen) { this->gen = gen; }

void LightningSimulator::PrintState()
{
    using std::cout;
    using std::endl;

    const size_t num_qubits = this->device_sv->getNumQubits();
    const size_t size = Pennylane::Util::exp2(num_qubits);
    size_t idx = 0;
    cout << "*** State-Vector of Size " << size << " ***" << endl;
    cout << "[";
    auto &&state = this->device_sv->getDataVector();
    for (; idx < size - 1; idx++) {
        cout << state[idx] << ", ";
    }
    cout << state[idx] << "]" << endl;
}

void LightningSimulator::SetState(DataView<std::complex<double>, 1> &data,
                                  std::vector<QubitIdType> &wires)
{
    std::vector<std::complex<double>> data_vector(data.begin(), data.end());
    this->device_sv->setStateVector(data_vector, getDeviceWires(wires));
}

void LightningSimulator::SetBasisState(DataView<int8_t, 1> &data, std::vector<QubitIdType> &wires)
{
    std::vector<std::size_t> data_vector(data.begin(), data.end());
    this->device_sv->setBasisState(data_vector, getDeviceWires(wires));
}

auto LightningSimulator::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto LightningSimulator::One() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST);
}

void LightningSimulator::NamedOperation(const std::string &name, const std::vector<double> &params,
                                        const std::vector<QubitIdType> &wires, bool inverse,
                                        const std::vector<QubitIdType> &controlled_wires,
                                        const std::vector<bool> &controlled_values)
{
    // Check the validity of number of qubits and parameters
    RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
               "Controlled wires/values size mismatch");
    RT_FAIL_IF(!isValidQubits(wires), "Given wires do not refer to qubits");
    RT_FAIL_IF(!isValidQubits(controlled_wires), "Given controlled wires do not refer to qubits");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);
    auto &&dev_controlled_wires = getDeviceWires(controlled_wires);

    // Update the state-vector
    if (controlled_wires.empty()) {
        this->device_sv->applyOperation(name, dev_wires, inverse, params);
    }
    else {
        this->device_sv->applyOperation(name, dev_controlled_wires, controlled_values, dev_wires,
                                        inverse, params);
    }

    // Update tape caching if required
    if (this->tape_recording) {
        this->cache_manager.addOperation(name, params, dev_wires, inverse, {}, dev_controlled_wires,
                                         controlled_values);
    }
}

void LightningSimulator::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                         const std::vector<QubitIdType> &wires, bool inverse,
                                         const std::vector<QubitIdType> &controlled_wires,
                                         const std::vector<bool> &controlled_values)
{
    RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
               "Controlled wires/values size mismatch");
    RT_FAIL_IF(!isValidQubits(wires), "Given wires do not refer to qubits");
    RT_FAIL_IF(!isValidQubits(controlled_wires), "Given controlled wires do not refer to qubits");

    // Convert wires to device wires
    // with checking validity of wires
    auto &&dev_wires = getDeviceWires(wires);
    auto &&dev_controlled_wires = getDeviceWires(controlled_wires);

    // Update the state-vector
    if (controlled_wires.empty()) {
        this->device_sv->applyMatrix(matrix.data(), dev_wires, inverse);
    }
    else {
        this->device_sv->applyControlledMatrix(matrix.data(), dev_controlled_wires,
                                               controlled_values, dev_wires, inverse);
    }

    // Update tape caching if required
    if (this->tape_recording) {
        this->cache_manager.addOperation("QubitUnitary", {}, dev_wires, inverse, matrix,
                                         dev_controlled_wires, controlled_values);
    }
}

auto LightningSimulator::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                    const std::vector<QubitIdType> &wires) -> ObsIdType
{
    RT_FAIL_IF(wires.size() > this->GetNumQubits(), "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);

    if (id == ObsId::Hermitian) {
        return this->obs_manager.createHermitianObs(matrix, dev_wires);
    }

    return this->obs_manager.createNamedObs(id, dev_wires);
}

auto LightningSimulator::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return this->obs_manager.createTensorProdObs(obs);
}

auto LightningSimulator::HamiltonianObservable(const std::vector<double> &coeffs,
                                               const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return this->obs_manager.createHamiltonianObs(coeffs, obs);
}

auto LightningSimulator::Expval(ObsIdType obsKey) -> double
{
    RT_FAIL_IF(!this->obs_manager.isValidObservables({obsKey}),
               "Invalid key for cached observables");
    auto &&obs = this->obs_manager.getObservable(obsKey);

    // update tape caching
    if (this->tape_recording) {
        this->cache_manager.addObservable(obsKey, MeasurementsT::Expval);
    }

    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};

    return device_shots ? m.expval(*obs, device_shots, {}) : m.expval(*obs);
}

auto LightningSimulator::Var(ObsIdType obsKey) -> double
{
    RT_FAIL_IF(!this->obs_manager.isValidObservables({obsKey}),
               "Invalid key for cached observables");
    auto &&obs = this->obs_manager.getObservable(obsKey);

    // update tape caching
    if (this->tape_recording) {
        this->cache_manager.addObservable(obsKey, MeasurementsT::Var);
    }

    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};

    return device_shots ? m.var(*obs, device_shots) : m.var(*obs);
}

void LightningSimulator::State(DataView<std::complex<double>, 1> &state)
{
    auto &&dv_state = this->device_sv->getDataVector();
    RT_FAIL_IF(state.size() != dv_state.size(), "Invalid size for the pre-allocated state vector");

    std::move(dv_state.begin(), dv_state.end(), state.begin());
}

void LightningSimulator::Probs(DataView<double, 1> &probs)
{
    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};
    auto &&dv_probs = device_shots ? m.probs(device_shots) : m.probs();

    RT_FAIL_IF(probs.size() != dv_probs.size(), "Invalid size for the pre-allocated probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void LightningSimulator::PartialProbs(DataView<double, 1> &probs,
                                      const std::vector<QubitIdType> &wires)
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");

    auto dev_wires = getDeviceWires(wires);
    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};
    auto &&dv_probs = device_shots ? m.probs(dev_wires, device_shots) : m.probs(dev_wires);

    RT_FAIL_IF(probs.size() != dv_probs.size(),
               "Invalid size for the pre-allocated partial-probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

std::vector<size_t> LightningSimulator::GenerateSamplesMetropolis(size_t shots)
{
    // generate_samples_metropolis is a member function of the Measures class.
    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    //
    // Return Value Optimization (RVO)
    return m.generate_samples_metropolis(this->kernel_name, this->num_burnin, shots);
}

std::vector<size_t> LightningSimulator::GenerateSamples(size_t shots)
{
    if (this->mcmc) {
        return this->GenerateSamplesMetropolis(shots);
    }
    // generate_samples is a member function of the Measures class.
    Pennylane::LightningQubit::Measures::Measurements<StateVectorT> m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    //
    // Return Value Optimization (RVO)
    if (this->gen) {
        return m.generate_samples(shots, (*(this->gen))());
    }
    return m.generate_samples(shots);
}

void LightningSimulator::Sample(DataView<double, 2> &samples, size_t shots)
{
    auto li_samples = this->GenerateSamples(shots);

    RT_FAIL_IF(samples.size() != li_samples.size(), "Invalid size for the pre-allocated samples");

    const size_t numQubits = this->GetNumQubits();

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < shots; shot++) {
        for (size_t wire = 0; wire < numQubits; wire++) {
            *(samplesIter++) = static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}

void LightningSimulator::PartialSample(DataView<double, 2> &samples,
                                       const std::vector<QubitIdType> &wires, size_t shots)
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF(samples.size() != shots * numWires,
               "Invalid size for the pre-allocated partial-samples");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    auto li_samples = this->GenerateSamples(shots);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    auto samplesIter = samples.begin();
    for (size_t shot = 0; shot < shots; shot++) {
        for (auto wire : dev_wires) {
            *(samplesIter++) = static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}

void LightningSimulator::Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                size_t shots)
{
    const size_t numQubits = this->GetNumQubits();
    const size_t numElements = 1U << numQubits;

    RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
               "Invalid size for the pre-allocated counts");

    auto li_samples = this->GenerateSamples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the bits of all qubits
    // into a bitstring.
    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<CHAR_BIT * sizeof(double)> basisState;
        size_t idx = numQubits;
        for (size_t wire = 0; wire < numQubits; wire++) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

void LightningSimulator::PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                                       const std::vector<QubitIdType> &wires, size_t shots)
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();
    const size_t numElements = 1U << numWires;

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF((eigvals.size() != numElements || counts.size() != numElements),
               "Invalid size for the pre-allocated partial-counts");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    auto li_samples = this->GenerateSamples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<CHAR_BIT * sizeof(double)> basisState;
        size_t idx = dev_wires.size();
        for (auto wire : dev_wires) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<size_t>(basisState.to_ulong())) += 1;
    }
}

auto LightningSimulator::Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result
{
    // get a measurement
    std::vector<QubitIdType> wires = {reinterpret_cast<QubitIdType>(wire)};

    std::vector<double> probs(1U << wires.size());
    DataView<double, 1> buffer_view(probs);
    auto device_shots = GetDeviceShots();
    SetDeviceShots(0);
    PartialProbs(buffer_view, wires);
    SetDeviceShots(device_shots);

    // It represents the measured result, true for 1, false for 0
    bool mres = Lightning::simulateDraw(probs, postselect, this->gen);
    auto dev_wires = getDeviceWires(wires);
    this->device_sv->collapse(dev_wires[0], mres ? 1 : 0);
    return mres ? this->One() : this->Zero();
}

// Gradient
void LightningSimulator::Gradient(std::vector<DataView<double, 1>> &gradients,
                                  const std::vector<size_t> &trainParams)
{
    const bool tp_empty = trainParams.empty();
    const size_t num_observables = this->cache_manager.getNumObservables();
    const size_t num_params = this->cache_manager.getNumParams();
    const size_t num_train_params = tp_empty ? num_params : trainParams.size();
    const size_t jac_size = num_train_params * this->cache_manager.getNumObservables();

    if (!jac_size) {
        return;
    }

    RT_FAIL_IF(gradients.size() != num_observables, "Invalid number of pre-allocated gradients");

    auto &&obs_callees = this->cache_manager.getObservablesCallees();
    bool is_valid_measurements =
        std::all_of(obs_callees.begin(), obs_callees.end(),
                    [](const auto &m) { return m == MeasurementsT::Expval; });
    RT_FAIL_IF(!is_valid_measurements,
               "Unsupported measurements to compute gradient; "
               "Adjoint differentiation method only supports expectation return type");

    auto &&state = this->device_sv->getDataVector();

    // create OpsData
    auto &&ops_names = this->cache_manager.getOperationsNames();
    auto &&ops_params = this->cache_manager.getOperationsParameters();
    auto &&ops_wires = this->cache_manager.getOperationsWires();
    auto &&ops_inverses = this->cache_manager.getOperationsInverses();
    auto &&ops_matrices = this->cache_manager.getOperationsMatrices();
    auto &&ops_controlled_wires = this->cache_manager.getOperationsControlledWires();
    auto &&ops_controlled_values = this->cache_manager.getOperationsControlledValues();

    const auto &&ops = Pennylane::Algorithms::OpsData<StateVectorT>(
        ops_names, ops_params, ops_wires, ops_inverses, ops_matrices, ops_controlled_wires,
        ops_controlled_values);

    // create the vector of observables
    auto &&obs_keys = this->cache_manager.getObservablesKeys();
    std::vector<std::shared_ptr<Pennylane::Observables::Observable<StateVectorT>>> obs_vec;
    obs_vec.reserve(obs_keys.size());
    for (auto idx : obs_keys) {
        obs_vec.emplace_back(this->obs_manager.getObservable(idx));
    }

    std::vector<size_t> all_params;
    if (tp_empty) {
        all_params.reserve(num_params);
        for (size_t i = 0; i < num_params; i++) {
            all_params.push_back(i);
        }
    }

    // construct the Jacobian data
    Pennylane::Algorithms::JacobianData<StateVectorT> tape{
        num_params, state.size(), state.data(), obs_vec, ops, tp_empty ? all_params : trainParams};

    Pennylane::LightningQubit::Algorithms::AdjointJacobian<StateVectorT> adj;
    std::vector<double> jacobian(jac_size, 0);
    adj.adjointJacobian(std::span{jacobian}, tape,
                        /* ref_data */ *this->device_sv,
                        /* apply_operations */ false);

    // convert jacobians to a list of lists for each observable
    std::vector<double> jacobian_t =
        Pennylane::LightningQubit::Util::Transpose(jacobian, num_train_params, num_observables);

    std::vector<double> cur_buffer(num_train_params);
    auto begin_loc_iter = jacobian_t.begin();
    for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
        RT_ASSERT(begin_loc_iter != jacobian_t.end());
        RT_ASSERT(num_train_params <= gradients[obs_idx].size());
        std::move(begin_loc_iter, begin_loc_iter + num_train_params, cur_buffer.begin());
        std::move(cur_buffer.begin(), cur_buffer.end(), gradients[obs_idx].begin());
        begin_loc_iter += num_train_params;
    }
}

} // namespace Catalyst::Runtime::Simulator

GENERATE_DEVICE_FACTORY(LightningSimulator, Catalyst::Runtime::Simulator::LightningSimulator);
