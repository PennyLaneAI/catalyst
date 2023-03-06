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
#include "BaseUtils.hpp"

namespace Catalyst::Runtime::Simulator {

auto LightningSimulator::AllocateQubit() -> QubitIdType
{
    size_t sv_id = this->device_sv->allocateWire();
    return this->qubit_manager.Allocate(sv_id);
}

auto LightningSimulator::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (num_qubits == 0U) {
        return {};
    }

    // at the first call when num_qubits == 0
    if (this->GetNumQubits() == 0U) {
        this->device_sv = std::make_unique<Pennylane::StateVectorDynamicCPU<double>>(num_qubits);
        return this->qubit_manager.AllocateRange(0, num_qubits);
    }

    std::vector<QubitIdType> result{};
    result.reserve(num_qubits);
    for (size_t i = 0; i < num_qubits; i++) {
        result.push_back(AllocateQubit());
    }
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
    QFailIf(this->cache_recording, "Cannot re-activate the cache manager");
    this->cache_recording = true;
    this->cache_manager.Reset();
}

void LightningSimulator::StopTapeRecording()
{
    if (this->cache_recording) {
        this->cache_recording = false;
    }
}

auto LightningSimulator::CacheManagerInfo()
    -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>
{
    return {this->cache_manager.getNumOperations(), this->cache_manager.getNumObservables(),
            this->cache_manager.getNumParams(), this->cache_manager.getOperationsNames(),
            this->cache_manager.getObservablesKeys()};
}

void LightningSimulator::SetDeviceShots(size_t shots) { device_shots = shots; }

auto LightningSimulator::GetDeviceShots() const -> size_t { return device_shots; }

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

auto LightningSimulator::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto LightningSimulator::One() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST);
}

void LightningSimulator::NamedOperation(const std::string &name, const std::vector<double> &params,
                                        const std::vector<QubitIdType> &wires, bool inverse)
{
    // First, check if operation `name` is supported by the simulator
    auto &&[op_num_wires, op_num_params] =
        Lightning::lookup_gates(Lightning::simulator_gate_info, name);

    // Check the validity of number of qubits and parameters
    QFailIf((!wires.size() && wires.size() != op_num_wires), "Invalid number of qubits");
    QFailIf(params.size() != op_num_params, "Invalid number of parameters");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);

    // Update the state-vector
    this->device_sv->applyOperation(name, dev_wires, inverse, params);

    // Update tape caching if required
    if (this->cache_recording) {
        this->cache_manager.addOperation(name, params, dev_wires, inverse);
    }
}

void LightningSimulator::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                         const std::vector<QubitIdType> &wires, bool inverse)
{
    // Convert wires to device wires
    // with checking validity of wires
    auto &&dev_wires = getDeviceWires(wires);

    // Update the state-vector
    this->device_sv->applyMatrix(matrix.data(), dev_wires, inverse);
}

auto LightningSimulator::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                    const std::vector<QubitIdType> &wires) -> ObsIdType
{
    QFailIf(wires.size() > GetNumQubits(), "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires");

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
    QFailIf(!this->obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");
    auto obs = this->obs_manager.getObservable(obsKey);

    // update tape caching
    if (this->cache_recording) {
        this->cache_manager.addObservable(obsKey, Lightning::Measurements::Expval);
    }

    Pennylane::Simulators::Measures m{*(this->device_sv)};

    return m.expval(*obs);
}

auto LightningSimulator::Var(ObsIdType obsKey) -> double
{
    QFailIf(!this->obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");

    auto obs = this->obs_manager.getObservable(obsKey);

    // update tape caching
    if (this->cache_recording) {
        this->cache_manager.addObservable(obsKey, Lightning::Measurements::Var);
    }

    auto obs_str = obs->getObsName();
    size_t found = obs_str.find_first_of("[");
    if (found != std::string::npos) {
        obs_str = obs_str.substr(0, found);
    }

    auto obs_wires = obs->getWires();

    Pennylane::Simulators::Measures m{*(this->device_sv)};

    const double result = m.var(obs_str, obs_wires);

    return result;
}

auto LightningSimulator::State() -> std::vector<std::complex<double>>
{
    auto &&state = this->device_sv->getDataVector();
    return std::vector<std::complex<double>>(state.begin(), state.end());
}

auto LightningSimulator::Probs() -> std::vector<double>
{
    // QFailIf((1U << numQubits) != numAlloc,
    //         "Cannot copy the probabilities to an array with different size; "
    //         "allocation size must be '2 ** numQubits'");

    Pennylane::Simulators::Measures m{*(this->device_sv)};

    return m.probs();
}

auto LightningSimulator::PartialProbs(const std::vector<QubitIdType> &wires) -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    auto dev_wires = getDeviceWires(wires);
    Pennylane::Simulators::Measures m{*(this->device_sv)};

    return m.probs(dev_wires);
}

auto LightningSimulator::Sample(size_t shots) -> std::vector<double>
{
    // generate_samples is a member function of the Measures class.
    Pennylane::Simulators::Measures m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    auto &&li_samples = m.generate_samples(shots);

    const size_t numQubits = GetNumQubits();

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    // TODO: matrix transpose
    std::vector<double> samples(li_samples.size());
    for (size_t shot = 0; shot < shots; shot++) {
        for (size_t wire = 0; wire < numQubits; wire++) {
            samples[shot * numQubits + wire] =
                static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }

    return samples;
}

auto LightningSimulator::PartialSample(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    // generate_samples is a member function of the Measures class.
    Pennylane::Simulators::Measures m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    auto &&li_samples = m.generate_samples(shots);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    // TODO: matrix transpose
    std::vector<double> samples(shots * numWires);
    for (size_t shot = 0; shot < shots; shot++) {
        size_t idx = 0;
        for (auto wire : dev_wires) {
            samples[shot * numWires + idx++] =
                static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }

    return samples;
}

auto LightningSimulator::Counts(size_t shots)
    -> std::tuple<std::vector<double>, std::vector<int64_t>>
{
    // generate_samples is a member function of the Measures class.
    Pennylane::Simulators::Measures m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    auto &&li_samples = m.generate_samples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    const size_t numQubits = GetNumQubits();
    const size_t numElements = 1U << numQubits;
    std::vector<double> eigvals(numElements);
    std::iota(eigvals.begin(), eigvals.end(), 0);
    eigvals.reserve(numElements);
    std::vector<int64_t> counts(numElements);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the bits of all qubits
    // into a bitstring.
    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = 0;
        for (size_t wire = 0; wire < numQubits; wire++) {
            basisState[idx++] = li_samples[shot * numQubits + wire];
        }
        counts[basisState.to_ulong()] += 1;
    }

    return {eigvals, counts};
}

auto LightningSimulator::PartialCounts(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::tuple<std::vector<double>, std::vector<int64_t>>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    // generate_samples is a member function of the Measures class.
    Pennylane::Simulators::Measures m{*(this->device_sv)};

    // PL-Lightning generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Alias_method
    // Given the number of samples, returns 1-D vector of samples
    // in binary, each sample is separated by a stride equal to
    // the number of qubits.
    auto &&li_samples = m.generate_samples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    const size_t numElements = 1U << numWires;
    std::vector<double> eigvals(numElements);
    std::iota(eigvals.begin(), eigvals.end(), 0);
    eigvals.reserve(numElements);
    std::vector<int64_t> counts(numElements);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    for (size_t shot = 0; shot < shots; shot++) {
        std::bitset<52> basisState; // only 52 bits of precision in a double, TODO: improve
        size_t idx = 0;
        for (auto wire : dev_wires) {
            basisState[idx++] = li_samples[shot * numQubits + wire];
        }
        counts[basisState.to_ulong()] += 1;
    }

    return {eigvals, counts};
}

auto LightningSimulator::Measure(QubitIdType wire) -> Result
{
    // get a measurement
    std::vector<QubitIdType> wires = {reinterpret_cast<QubitIdType>(wire)};
    auto &&probs = this->PartialProbs(wires);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);
    float draw = dis(gen);
    bool mres = draw > probs[0];

    const size_t numQubits = GetNumQubits();

    auto &&state = this->device_sv->getDataVector();

    const auto stride = pow(2, numQubits - (1 + wire));
    const auto vec_size = pow(2, numQubits);
    const auto section_size = vec_size / stride;
    const auto half_section_size = section_size / 2;

    // zero half the entries
    // the "half" entries depend on the stride
    // *_*_*_*_ for stride 1
    // **__**__ for stride 2
    // ****____ for stride 4
    const size_t k = mres ? 0 : 1;
    for (size_t idx = 0; idx < half_section_size; idx++) {
        for (size_t ids = 0; ids < stride; ids++) {
            auto v = stride * (k + 2 * idx) + ids;
            state[v] = {0., 0.};
        }
    }

    // get the total of the new vector (since we need to normalize)
    double total = 0.;
    for (size_t idx = 0; idx < vec_size; idx++) {
        total = total + std::real(state[idx] * std::conj(state[idx]));
    }

    // normalize the vector
    double norm = std::sqrt(total);
    std::for_each(state.begin(), state.end(), [norm](auto &elem) { elem /= norm; });

    return mres ? this->One() : this->Zero();
}

// Gradient
auto LightningSimulator::Gradient(const std::vector<size_t> &trainParams)
    -> std::vector<std::vector<double>>
{

    const bool tp_empty = trainParams.empty();
    const size_t num_observables = this->cache_manager.getNumObservables();
    const size_t num_params = this->cache_manager.getNumParams();
    const size_t num_train_params = tp_empty ? num_params : trainParams.size();
    const size_t jac_size = num_train_params * this->cache_manager.getNumObservables();

    if (!jac_size) {
        return {};
    }

    auto &&obs_callees = this->cache_manager.getObservablesCallees();
    bool is_valid_measurements =
        std::all_of(obs_callees.begin(), obs_callees.end(),
                    [](const auto &m) { return m == Lightning::Measurements::Expval; });
    QFailIf(!is_valid_measurements,
            "Unsupported measurements to compute gradient; "
            "Adjoint differentiation method only supports expectation return type");

    auto &&state = this->device_sv->getDataVector();

    // create OpsData
    auto &&ops_names = this->cache_manager.getOperationsNames();
    auto &&ops_params = this->cache_manager.getOperationsParameters();
    auto &&ops_wires = this->cache_manager.getOperationsWires();

    auto &&ops_inverses = this->cache_manager.getOperationsInverses();
    const auto &&ops =
        Pennylane::Algorithms::OpsData<double>(ops_names, ops_params, ops_wires, ops_inverses);

    // create the vector of observables
    auto &&obs_keys = this->cache_manager.getObservablesKeys();
    std::vector<std::shared_ptr<Pennylane::Simulators::Observable<double>>> obs_vec;
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
    Pennylane::Algorithms::JacobianData<double> tape{
        num_params, state.size(), state.data(), obs_vec, ops, tp_empty ? all_params : trainParams};

    std::vector<double> jacobian(jac_size, 0);
    Pennylane::Algorithms::adjointJacobian(std::span{jacobian}, tape,
                                           /* apply_operations */ false);

    // convert jacobians to a list of lists for each observable
    std::vector<double> jacobian_t =
        Pennylane::Util::Transpose(jacobian, num_train_params, num_observables);

    std::vector<std::vector<double>> results(num_observables);
    auto begin_loc_iter = jacobian_t.begin();
    for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
        assert(begin_loc_iter != jacobian_t.end());
        results[obs_idx].insert(results[obs_idx].begin(), begin_loc_iter,
                                begin_loc_iter + num_train_params);
        begin_loc_iter += num_train_params;
    }

    return results;
}

} // namespace Catalyst::Runtime::Simulator

namespace Catalyst::Runtime {
auto CreateQuantumDevice() -> std::unique_ptr<QuantumDevice>
{
    return std::make_unique<Simulator::LightningSimulator>();
}
} // namespace Catalyst::Runtime
