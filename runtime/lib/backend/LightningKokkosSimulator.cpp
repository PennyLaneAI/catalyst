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

namespace Catalyst::Runtime::Simulator {
auto LightningKokkosSimulator::AllocateQubit() -> QubitIdType
{
    const size_t num_qubits = this->device_sv->getNumQubits();
    this->device_sv = std::make_unique<Pennylane::StateVectorKokkos<double>>(num_qubits + 1);
    return qubit_manager.Allocate(num_qubits);
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

void LightningKokkosSimulator::ReleaseQubit(QubitIdType q) { qubit_manager.Release(q); }

void LightningKokkosSimulator::ReleaseAllQubits() { this->qubit_manager.ReleaseAll(); }

auto LightningKokkosSimulator::GetNumQubits() -> size_t const
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

void LightningKokkosSimulator::SetDeviceShots(size_t shots) { device_shots = shots; }

auto LightningKokkosSimulator::GetDeviceShots() -> size_t const { return device_shots; }

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

auto LightningKokkosSimulator::Zero() -> Result const
{
    return const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

auto LightningKokkosSimulator::One() -> Result const
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
    throw std::logic_error("MatrixOperation not implemented in PennyLane-Lightning-Kokkos");
}

auto LightningKokkosSimulator::Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                                          const std::vector<QubitIdType> &wires) -> ObsIdType
{
    if (id == ObsId::Hermitian) {
        throw std::logic_error(
            "Hermitian observable not implemented in PennyLane-Lightning-Kokkos");
    }

    QFailIf(wires.size() > GetNumQubits(), "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);
    return this->obs_manager.createNamedObs(id, dev_wires);
}

auto LightningKokkosSimulator::TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType
{
    throw std::logic_error("Tensor observable not implemented in PennyLane-Lightning-Kokkos");
}

auto LightningKokkosSimulator::HamiltonianObservable(const std::vector<double> &coeffs,
                                                     const std::vector<ObsIdType> &obs) -> ObsIdType
{
    throw std::logic_error("Hamiltonian observable not implemented in PennyLane-Lightning-Kokkos");
}

auto LightningKokkosSimulator::Expval(ObsIdType obsKey) -> double
{
    QFailIf(!this->obs_manager.isValidObservables({obsKey}), "Invalid key for cached observables");
    auto &&[obs, wires] = this->obs_manager.getObservable(obsKey);

    // update tape caching
    if (this->cache_recording) {
        cache_manager.addObservable(obsKey, Lightning::Measurements::Expval);
    }

    if (obs == ObsId::PauliX) {
        return this->device_sv->getExpectationValuePauliX(wires);
    }
    else if (obs == ObsId::PauliY) {
        return this->device_sv->getExpectationValuePauliY(wires);
    }
    else if (obs == ObsId::PauliZ) {
        return this->device_sv->getExpectationValuePauliZ(wires);
    }
    else if (obs == ObsId::Hadamard) {
        return this->device_sv->getExpectationValueHadamard(wires);
    }
    return this->device_sv->getExpectationValueIdentity(wires);
}

auto LightningKokkosSimulator::Var(ObsIdType obsKey) -> double
{
    throw std::logic_error("Variance not implemented in PennyLane-Lightning-Kokkos");
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

auto LightningKokkosSimulator::Probs() -> std::vector<double> { return this->device_sv->probs(); }

auto LightningKokkosSimulator::PartialProbs(const std::vector<QubitIdType> &wires)
    -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    auto dev_wires = getDeviceWires(wires);

    return this->device_sv->probs(dev_wires);
}

auto LightningKokkosSimulator::Sample(size_t shots) -> std::vector<double>
{
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = this->device_sv->generate_samples(shots);

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
auto LightningKokkosSimulator::PartialSample(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::vector<double>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = this->device_sv->generate_samples(shots);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    // TODO: matrix transpose
    std::vector<double> samples(li_samples.size());
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
    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = this->device_sv->generate_samples(shots);

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

auto LightningKokkosSimulator::PartialCounts(const std::vector<QubitIdType> &wires, size_t shots)
    -> std::tuple<std::vector<double>, std::vector<int64_t>>
{
    const size_t numWires = wires.size();
    const size_t numQubits = GetNumQubits();

    QFailIf(numWires > numQubits, "Invalid number of wires");
    QFailIf(!isValidQubits(wires), "Invalid given wires to measure");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    // PL-Lightning-Kokkos generates samples using the alias method.
    // Reference: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    auto li_samples = this->device_sv->generate_samples(shots);

    // Fill the eigenvalues with the integer representation of the corresponding
    // computational basis bitstring. In the future, eigenvalues can also be
    // obtained from an observable, hence the bitstring integer is stored as a
    // double.
    const size_t numElements = 1U << numQubits;
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

    const size_t numQubits = GetNumQubits();

    auto &&state = this->State();

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

auto LightningKokkosSimulator::Gradient(const std::vector<size_t> &trainParams)
    -> std::vector<std::vector<double>>
{
    throw std::logic_error("Gradient not fully supported for the "
                           "PennyLane-Lightning-Kokkos simulator");
}

} // namespace Catalyst::Runtime::Simulator

namespace Catalyst::Runtime {
auto CreateQuantumDevice() -> std::unique_ptr<QuantumDevice>
{
    return std::make_unique<Simulator::LightningKokkosSimulator>();
}
} // namespace Catalyst::Runtime
