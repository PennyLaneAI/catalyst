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

#include "LightningKokkosSimulator.hpp"
#include "BaseUtils.hpp"

namespace Catalyst::Runtime::Simulator {

auto LightningKokkosSimulator::AllocateQubit() -> QubitIdType
{
    const size_t num_qubits = this->device_sv->getNumQubits();
    this->device_sv = std::make_unique<Pennylane::StateVectorKokkos<double>>(num_qubits + 1);
    return this->qubit_manager.Allocate(num_qubits);
}

auto LightningKokkosSimulator::AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType>
{
    if (!num_qubits) {
        return {};
    }

    const size_t cur_num_qubits = this->device_sv->getNumQubits();
    const size_t new_num_qubits = cur_num_qubits + num_qubits;
    this->device_sv = std::make_unique<Pennylane::StateVectorKokkos<double>>(new_num_qubits);
    return this->qubit_manager.AllocateRange(cur_num_qubits, new_num_qubits);
}

void LightningKokkosSimulator::ReleaseQubit(QubitIdType q) { this->qubit_manager.Release(q); }

void LightningKokkosSimulator::ReleaseAllQubits() { this->qubit_manager.ReleaseAll(); }

auto LightningKokkosSimulator::GetNumQubits() const -> size_t
{
    return this->device_sv->getNumQubits();
}

void LightningKokkosSimulator::StartTapeRecording()
{
    QFailIf(this->cache_recording, "Cannot re-activate the cache manager");
    this->cache_recording = true;
    this->cache_manager.Reset();
}

void LightningKokkosSimulator::StopTapeRecording()
{
    if (this->cache_recording) {
        this->cache_recording = false;
    }
}

auto LightningKokkosSimulator::CacheManagerInfo()
    -> std::tuple<size_t, size_t, size_t, std::vector<std::string>, std::vector<ObsIdType>>
{
    return {this->cache_manager.getNumOperations(), this->cache_manager.getNumObservables(),
            this->cache_manager.getNumParams(), this->cache_manager.getOperationsNames(),
            this->cache_manager.getObservablesKeys()};
}

void LightningKokkosSimulator::SetDeviceShots(size_t shots) { this->device_shots = shots; }

auto LightningKokkosSimulator::GetDeviceShots() const -> size_t { return this->device_shots; }

void LightningKokkosSimulator::PrintState()
{
    using std::cout;
    using std::endl;

    const size_t num_qubits = this->device_sv->getNumQubits();
    const size_t size = Pennylane::Util::exp2(num_qubits);
    size_t idx = 0;
    cout << "*** State-Vector of Size " << size << " ***" << endl;
    cout << "[";
    for (; idx < size - 1; idx++) {
        auto elem_subview = Kokkos::subview(this->device_sv->getData(), idx);
        Kokkos::complex<double> elem_cp;
        Kokkos::deep_copy(elem_cp, elem_subview);

        cout << "(" << real(elem_cp) << "," << imag(elem_cp) << "), ";
    }
    auto elem_last_subview = Kokkos::subview(this->device_sv->getData(), idx);
    Kokkos::complex<double> elem_last_cp;
    Kokkos::deep_copy(elem_last_cp, elem_last_subview);
    cout << "(" << real(elem_last_cp) << "," << imag(elem_last_cp) << ")]" << endl;
}

auto LightningKokkosSimulator::Zero() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto LightningKokkosSimulator::One() const -> Result
{
    return const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST);
}

void LightningKokkosSimulator::NamedOperation(const std::string &name,
                                              const std::vector<double> &params,
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

void LightningKokkosSimulator::MatrixOperation(const std::vector<std::complex<double>> &matrix,
                                               const std::vector<QubitIdType> &wires, bool inverse)
{
    using UnmanagedComplexHostView = Kokkos::View<Kokkos::complex<double> *, Kokkos::HostSpace,
                                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Check the validity of number of qubits and parameters
    QFailIf(!wires.size(), "Invalid number of qubits");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);

    std::vector<Kokkos::complex<double>> matrix_kok;
    matrix_kok.resize(matrix.size());
    std::transform(matrix.begin(), matrix.end(), matrix_kok.begin(),
                   [](auto c) { return static_cast<Kokkos::complex<double>>(c); });

    Kokkos::View<Kokkos::complex<double> *> gate_matrix("gate_matrix", matrix_kok.size());
    Kokkos::deep_copy(gate_matrix, UnmanagedComplexHostView(matrix_kok.data(), matrix_kok.size()));

    // Update the state-vector
    this->device_sv->applyMultiQubitOp(gate_matrix, dev_wires, inverse);

    // Update tape caching if required
    if (this->cache_recording) {
        this->cache_manager.addOperation("MatrixOp", {}, dev_wires, inverse);
    }
}

auto LightningKokkosSimulator::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                          const std::vector<QubitIdType> &wires) -> ObsIdType
{
    QFailIf(wires.size() > this->GetNumQubits(), "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);

    if (id == ObsId::Hermitian) {
        return this->obs_manager.createHermitianObs(matrix, dev_wires);
    }

    return this->obs_manager.createNamedObs(id, dev_wires);
}

auto LightningKokkosSimulator::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return this->obs_manager.createTensorProdObs(obs);
}

auto LightningKokkosSimulator::HamiltonianObservable(const std::vector<double> &coeffs,
                                                     const std::vector<ObsIdType> &obs) -> ObsIdType
{
    return this->obs_manager.createHamiltonianObs(coeffs, obs);
}

// TODO: remove this kernel after merging expval(const ObservableKokkos<T> &ob)
// in PennyLane-Lightning-Kokkos
template <class Precision> struct getRealOfComplexInnerProductFunctor {
    Kokkos::View<Kokkos::complex<Precision> *> sv1;
    Kokkos::View<Kokkos::complex<Precision> *> sv2;

    getRealOfComplexInnerProductFunctor(Kokkos::View<Kokkos::complex<Precision> *> sv1_,
                                        Kokkos::View<Kokkos::complex<Precision> *> sv2_)
    {
        sv1 = sv1_;
        sv2 = sv2_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, Precision &inner) const
    {
        inner += real(conj(sv1[k]) * sv2[k]);
    }
};

template <class Precision>
inline auto getRealOfComplexInnerProduct(Kokkos::View<Kokkos::complex<Precision> *> sv1_vec,
                                         Kokkos::View<Kokkos::complex<Precision> *> sv2_vec)
    -> Precision
{
    assert(sv1_vec.size() == sv2_vec.size());
    Precision inner = 0;
    Kokkos::parallel_reduce(
        sv1_vec.size(), getRealOfComplexInnerProductFunctor<Precision>(sv1_vec, sv2_vec), inner);
    return inner;
}

auto LightningKokkosSimulator::Expval(ObsIdType obsKey) -> double
{
    using UnmanagedComplexHostView = Kokkos::View<Kokkos::complex<double> *, Kokkos::HostSpace,
                                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    QFailIf(!this->obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");

    // update tape caching
    if (this->cache_recording) {
        cache_manager.addObservable(obsKey, Lightning::Measurements::Expval);
    }

    auto &&obs = this->obs_manager.getObservable(obsKey);

    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};

    return m.expval(*obs);
}

auto LightningKokkosSimulator::Var(ObsIdType obsKey) -> double
{
    QFailIf(!this->obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");

    // update tape caching
    if (this->cache_recording) {
        this->cache_manager.addObservable(obsKey, Lightning::Measurements::Var);
    }

    auto &&obs = this->obs_manager.getObservable(obsKey);

    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};

    return m.var(*obs);
}

auto LightningKokkosSimulator::State() -> std::vector<std::complex<double>>
{
    const size_t num_qubits = this->device_sv->getNumQubits();
    const size_t size = Pennylane::Util::exp2(num_qubits);
    std::vector<std::complex<double>> state;
    state.reserve(size);

    for (size_t idx = 0; idx < size; idx++) {
        auto elem_subview = Kokkos::subview(this->device_sv->getData(), idx);
        Kokkos::complex<double> elem_cp;
        Kokkos::deep_copy(elem_cp, elem_subview);
        double elem_cp_real = real(elem_cp);
        double elem_cp_imag = imag(elem_cp);
        state.emplace_back(static_cast<double>(real(elem_cp)), static_cast<double>(imag(elem_cp)));
    }
    return state;
}

auto LightningKokkosSimulator::Probs() -> std::vector<double>
{
    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    return m.probs();
}

auto LightningKokkosSimulator::PartialProbs(const std::vector<QubitIdType> &wires)
    -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    auto dev_wires = getDeviceWires(wires);

    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    return m.probs(dev_wires);
}

auto LightningKokkosSimulator::Sample(size_t shots) -> std::vector<double>
{
    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = m.generate_samples(shots);

    const size_t numQubits = this->GetNumQubits();

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
auto LightningKokkosSimulator::PartialSample(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = m.generate_samples(shots);

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

auto LightningKokkosSimulator::Counts(size_t shots)
    -> std::tuple<std::vector<double>, std::vector<int64_t>>
{
    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = m.generate_samples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    const size_t numQubits = this->GetNumQubits();
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

auto LightningKokkosSimulator::PartialCounts(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::tuple<std::vector<double>, std::vector<int64_t>>
{
    const size_t numWires = wires.size();
    const size_t numQubits = this->GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    Pennylane::Simulators::MeasuresKokkos m{*(this->device_sv)};
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = m.generate_samples(shots);

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

auto LightningKokkosSimulator::Measure(QubitIdType wire) -> Result
{
    // get a measurement
    std::vector<QubitIdType> wires = {reinterpret_cast<QubitIdType>(wire)};
    auto &&probs = this->PartialProbs(wires);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);
    float draw = dis(gen);
    bool mres = draw > probs[0];

    const size_t num_qubits = this->GetNumQubits();

    auto &&dev_wires = this->getDeviceWires(wires);
    const auto stride = pow(2, num_qubits - (1 + dev_wires[0]));
    const auto vec_size = pow(2, num_qubits);
    const auto section_size = vec_size / stride;
    const auto half_section_size = section_size / 2;

    std::vector<Kokkos::complex<double>> state;
    state.reserve(vec_size);

    Kokkos::complex<double> elem_cp;
    for (size_t idx = 0; idx < vec_size; idx++) {
        auto elem_subview = Kokkos::subview(this->device_sv->getData(), idx);
        Kokkos::deep_copy(elem_cp, elem_subview);
        state.emplace_back(elem_cp);
    }

    // zero half the entries
    // the "half" entries depend on the stride
    // *_*_*_*_ for stride 1
    // **__**__ for stride 2
    // ****____ for stride 4
    const size_t k = mres ? 0 : 1;
    for (size_t idx = 0; idx < half_section_size; idx++) {
        for (size_t ids = 0; ids < stride; ids++) {
            auto v = stride * (k + 2 * idx) + ids;
            state[v] = Kokkos::complex<double>(0.0, 0.0);
        }
    }

    // get the total of the new vector (since we need to normalize)
    double total =
        std::accumulate(state.begin(), state.end(), 0.0, [](double sum, Kokkos::complex<double> c) {
            return sum + real(c * conj(c));
        });

    // normalize the vector
    double norm = std::sqrt(total);
    std::for_each(state.begin(), state.end(), [norm](auto &elem) { elem /= norm; });

    // TODO: rewrite this method using setStateVector(vectorKokkos<T>)
    // and LinAlg Functors in the next version of PennyLane-Lightning-Kokkos
    this->device_sv =
        std::make_unique<Pennylane::StateVectorKokkos<double>>(state.data(), vec_size);

    return mres ? this->One() : this->Zero();
}

auto LightningKokkosSimulator::Gradient(const std::vector<size_t> &trainParams)
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

    // Create OpsData
    auto &&ops_names = this->cache_manager.getOperationsNames();
    auto &&ops_params = this->cache_manager.getOperationsParameters();
    auto &&ops_wires = this->cache_manager.getOperationsWires();
    auto &&ops_inverses = this->cache_manager.getOperationsInverses();
    Pennylane::Algorithms::AdjointJacobianKokkos<double> adj;
    const auto ops = adj.createOpsData(ops_names, ops_params, ops_wires, ops_inverses);

    // Create the vector of observables
    auto &&obs_keys = this->cache_manager.getObservablesKeys();
    std::vector<std::shared_ptr<Pennylane::Simulators::ObservableKokkos<double>>> obs_vec;
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

    std::vector<std::vector<double>> jacobian(num_observables,
                                              std::vector<double>(num_train_params, 0.0));
    adj.adjointJacobian(*this->device_sv, jacobian, obs_vec, ops,
                        tp_empty ? all_params : trainParams, /* apply_operations */ false);

    return jacobian;
}

} // namespace Catalyst::Runtime::Simulator

namespace Catalyst::Runtime {
auto CreateQuantumDevice() -> std::unique_ptr<QuantumDevice>
{
    return std::make_unique<Simulator::LightningKokkosSimulator>();
}
} // namespace Catalyst::Runtime
