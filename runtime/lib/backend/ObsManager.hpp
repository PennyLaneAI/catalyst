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

#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "BaseUtils.hpp"
#include "LightningUtils.hpp"
#include "Types.h"

#if __has_include("Observables.hpp")
#include "Observables.hpp"
#define LIGHTNING_OBSERVABLE_CLASS
#endif

namespace Catalyst::Runtime::Simulator {

#if defined(LIGHTNING_OBSERVABLE_CLASS)

/**
 * @brief The LightningObsManager caches observables of a program at runtime
 * and maps each one to a const unique index (`int64_t`) in the scope
 * of the global context manager.
 */
template <typename PrecisionT> class LightningObsManager {
  private:
    std::vector<std::pair<std::shared_ptr<Pennylane::Simulators::Observable<PrecisionT>>, ObsType>>
        observables_;

    static constexpr std::array<ObsType, 2> hamiltonian_valid_obs_types = {
        ObsType::Basic,
        ObsType::TensorProd,
    };

  public:
    LightningObsManager() = default;
    ~LightningObsManager() = default;

    LightningObsManager(const LightningObsManager &) = delete;
    LightningObsManager &operator=(const LightningObsManager &) = delete;

    /**
     * @brief A helper function to clear constructed observables in the program.
     */
    void clear() { observables_.clear(); }

    [[nodiscard]] auto getObservable(ObsIdType key)
        -> std::shared_ptr<Pennylane::Simulators::Observable<PrecisionT>>
    {
        int64_t key_t = reinterpret_cast<int64_t>(key);
        QFailIf(static_cast<size_t>(key_t) >= observables_.size() || key_t < 0,
                "Invalid observable key");

        return std::get<0>(observables_[key_t]);
    }

    [[nodiscard]] auto numObservables() -> size_t { return observables_.size(); }

    [[nodiscard]] auto isValidObservables(const std::vector<ObsIdType> &obsKeys) -> bool
    {
        return std::all_of(obsKeys.begin(), obsKeys.end(),
                           [this](auto i) { return (this->getObservable(i) != nullptr); });
    }

    auto createNamedObs(ObsId obsId, const std::vector<size_t> &wires) -> ObsIdType
    {
        auto obs_str =
            std::string(Lightning::lookup_obs<Lightning::simulator_observable_support_size>(
                Lightning::simulator_observable_support, obsId));

        observables_.push_back(std::make_pair(
            std::make_shared<Pennylane::Simulators::NamedObs<PrecisionT>>(obs_str, wires),
            ObsType::Basic));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto createHermitianObs(const std::vector<std::complex<PrecisionT>> &matrix,
                            const std::vector<size_t> &wires) -> ObsIdType
    {
        observables_.push_back(
            std::make_pair(std::make_shared<Pennylane::Simulators::HermitianObs<PrecisionT>>(
                               Pennylane::Simulators::HermitianObs<PrecisionT>{matrix, wires}),
                           ObsType::Basic));

        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto createTensorProdObs(const std::vector<ObsIdType> &obsKeys) -> ObsIdType
    {
        const auto key_size = obsKeys.size();
        const auto obs_size = observables_.size();

        std::vector<std::shared_ptr<Pennylane::Simulators::Observable<PrecisionT>>> obs_vec;
        obs_vec.reserve(key_size);

        for (const auto &key : obsKeys) {
            int64_t key_t = reinterpret_cast<int64_t>(key);
            QFailIf(static_cast<size_t>(key_t) >= obs_size || key_t < 0, "Invalid observable key");

            auto &&[obs, type] = observables_[key_t];

            QFailIf(type != ObsType::Basic, "Invalid basic observable to construct TensorProd; "
                                            "NamedObs and HermitianObs are only supported");

            obs_vec.push_back(obs);
        }

        observables_.push_back(
            std::make_pair(std::make_shared<Pennylane::Simulators::TensorProdObs<PrecisionT>>(
                               Pennylane::Simulators::TensorProdObs<PrecisionT>::create(obs_vec)),
                           ObsType::TensorProd));

        return static_cast<ObsIdType>(obs_size);
    }

    auto createHamiltonianObs(const std::vector<PrecisionT> &coeffs,
                              const std::vector<ObsIdType> &obsKeys) -> ObsIdType
    {
        const auto key_size = obsKeys.size();
        const auto obs_size = observables_.size();

        QFailIf(key_size != coeffs.size(),
                "Incompatible list of observables and coefficients; "
                "Number of observables and number of coefficients must be equal");

        std::vector<std::shared_ptr<Pennylane::Simulators::Observable<PrecisionT>>> obs_vec;
        obs_vec.reserve(key_size);

        for (auto key : obsKeys) {
            int64_t key_t = reinterpret_cast<int64_t>(key);
            QFailIf(static_cast<size_t>(key_t) >= obs_size || key_t < 0, "Invalid observable key");

            auto &&[obs, type] = observables_[key_t];
            auto contain_obs = std::find(hamiltonian_valid_obs_types.begin(),
                                         hamiltonian_valid_obs_types.end(), type);

            QFailIf(contain_obs == hamiltonian_valid_obs_types.end(),
                    "Invalid observable to construct Hamiltonian; "
                    "NamedObs, HermitianObs and TensorProdObs are only supported");

            obs_vec.push_back(obs);
        }

        observables_.push_back(std::make_pair(
            std::make_shared<Pennylane::Simulators::Hamiltonian<PrecisionT>>(
                Pennylane::Simulators::Hamiltonian<PrecisionT>(coeffs, std::move(obs_vec))),
            ObsType::Hamiltonian));

        return static_cast<ObsIdType>(obs_size);
    }
};

#endif

/**
 * @brief The LightningKokkosObsManager caches observables of a program at
 * runtime and maps each one to a const unique index (`int64_t`) in the scope of
 * the global context manager.
 */
template <typename PrecisionT> class LightningKokkosObsManager {
  private:
    std::vector<std::tuple<ObsId, std::vector<size_t>>> observables_;

  public:
    LightningKokkosObsManager() = default;
    ~LightningKokkosObsManager() = default;

    LightningKokkosObsManager(const LightningKokkosObsManager &) = delete;
    LightningKokkosObsManager &operator=(const LightningKokkosObsManager &) = delete;

    void clear() { observables_.clear(); }

    [[nodiscard]] auto getObservable(ObsIdType key) -> std::pair<ObsId, std::vector<size_t>>
    {
        int64_t key_t = reinterpret_cast<int64_t>(key);
        QFailIf(static_cast<size_t>(key_t) >= observables_.size() || key_t < 0,
                "Invalid observable key");

        auto &&[obs, wires] = observables_[key_t];
        return {obs, wires};
    }

    [[nodiscard]] auto numObservables() -> size_t { return observables_.size(); }

    [[nodiscard]] auto isValidObservables(const std::vector<ObsIdType> &obsKeys) -> bool
    {
        return std::all_of(obsKeys.begin(), obsKeys.end(),
                           [this](auto i) { return (this->getObservable(i) != nullptr); });
    }

    auto createNamedObs(ObsId obsId, const std::vector<size_t> &wires) -> ObsIdType
    {
        auto obs_str =
            std::string(Lightning::lookup_obs<Lightning::simulator_observable_support_size>(
                Lightning::simulator_observable_support, obsId));

        observables_.emplace_back(obsId, wires);

        return static_cast<ObsIdType>(observables_.size() - 1);
    }
};
} // namespace Catalyst::Runtime::Simulator
