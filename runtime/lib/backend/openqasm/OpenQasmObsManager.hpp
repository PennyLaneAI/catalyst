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

#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "Exception.hpp"
#include "OpenQasmBuilder.hpp"
#include "Utils.hpp"

#include "Types.h"

namespace Catalyst::Runtime::Device::OpenQasm {

/**
 * @brief The OpenQasmObsManager caches observables of a program at runtime
 * and maps each one to a const unique index (`int64_t`) in the scope
 * of the global context manager.
 */
class OpenQasmObsManager {
  private:
    using ObservablePairType = std::pair<std::shared_ptr<QasmObs>, ObsType>;
    std::vector<ObservablePairType> observables_{};

    static constexpr std::array<ObsType, 2> hamiltonian_valid_obs_types = {
        ObsType::Basic,
        ObsType::TensorProd,
    };

  public:
    OpenQasmObsManager() = default;
    ~OpenQasmObsManager() = default;

    OpenQasmObsManager(const OpenQasmObsManager &) = delete;
    OpenQasmObsManager &operator=(const OpenQasmObsManager &) = delete;
    OpenQasmObsManager(OpenQasmObsManager &&) = delete;
    OpenQasmObsManager &operator=(OpenQasmObsManager &&) = delete;

    /**
     * @brief A helper function to clear constructed observables in the program.
     */
    void clear() { this->observables_.clear(); }

    /**
     * @brief Check the validity of observable keys.
     *
     * @param obsKeys The vector of observable keys
     * @return bool
     */
    [[nodiscard]] auto isValidObservables(const std::vector<ObsIdType> &obsKeys) const -> bool {
        return std::all_of(obsKeys.begin(), obsKeys.end(), [this](auto i) {
            return (i >= 0 && static_cast<size_t>(i) < this->observables_.size());
        });
    }

    /**
     * @brief Get the constructed observable instance.
     *
     * @param key The observable key
     * @return std::shared_ptr<ObservableClassName>
     */
    [[nodiscard]] auto getObservable(ObsIdType key) -> std::shared_ptr<QasmObs> {
        RT_FAIL_IF(!this->isValidObservables({key}), "Invalid observable key");
        return std::get<0>(this->observables_[reinterpret_cast<int64_t>(key)]);
    }

    /**
     * @brief Get the number of observables.
     *
     * @return size_t
     */
    [[nodiscard]] auto numObservables() const -> size_t { return this->observables_.size(); }

    /**
     * @brief Create and cache a new NamedObs instance.
     *
     * @param obsId The named observable id of type ObsId
     * @param wires The vector of wires the observable acts on
     * @return ObsIdType
     */
    [[nodiscard]] auto createNamedObs(ObsId obsId, const std::vector<size_t> &wires) -> ObsIdType {
        using namespace Catalyst::Runtime::Simulator::Lightning;

        auto &&obs_str = std::string(
            lookup_obs<simulator_observable_support_size>(simulator_observable_support, obsId));

        this->observables_.push_back(
            std::make_pair(std::make_shared<QasmNamedObs>(obs_str, wires), ObsType::Basic));
        return static_cast<ObsIdType>(this->observables_.size() - 1);
    }

    /**
     * @brief Create and cache a new HermitianObs instance.
     *
     * @param matrix The row-wise Hermitian matrix
     * @param wires The vector of wires the observable acts on
     * @return ObsIdType
     */
    [[nodiscard]] auto
    createHermitianObs([[maybe_unused]] const std::vector<std::complex<double>> &matrix,
                       [[maybe_unused]] const std::vector<size_t> &wires) -> ObsIdType {
        RT_FAIL("Unsupported functionality");
        return static_cast<ObsIdType>(this->observables_.size() - 1);
    }

    /**
     * @brief Create and cache a new TensorProd instance.
     *
     * @param obsKeys The vector of observable keys
     * @return ObsIdType
     */
    [[nodiscard]] auto createTensorProdObs(const std::vector<ObsIdType> &obsKeys) -> ObsIdType {
        const auto key_size = obsKeys.size();
        const auto obs_size = this->observables_.size();

        std::vector<std::shared_ptr<QasmObs>> obs_vec;
        obs_vec.reserve(key_size);

        for (const auto &key : obsKeys) {
            auto key_t = reinterpret_cast<int64_t>(key);
            RT_FAIL_IF(static_cast<size_t>(key_t) >= obs_size || key_t < 0,
                       "Invalid observable key");

            auto &&[obs, type] = this->observables_[key_t];

            RT_FAIL_IF(type != ObsType::Basic, "Invalid basic observable to construct TensorProd; "
                                               "NamedObs and HermitianObs are only supported");

            obs_vec.push_back(obs);
        }

        this->observables_.push_back(
            std::make_pair(std::make_shared<QasmTensorObs>(QasmTensorObs(std::move(obs_vec))),
                           ObsType::TensorProd));

        return static_cast<ObsIdType>(obs_size);
    }

    /**
     * @brief Create and cache a new HamiltonianObs instance.
     *
     * @param coeffs The vector of coefficients
     * @param obsKeys The vector of observable keys
     * @return ObsIdType
     */
    [[nodiscard]] auto createHamiltonianObs(const std::vector<double> &coeffs,
                                            const std::vector<ObsIdType> &obsKeys) -> ObsIdType {
        const auto key_size = obsKeys.size();
        const auto obs_size = this->observables_.size();

        RT_FAIL_IF(key_size != coeffs.size(),
                   "Incompatible list of observables and coefficients; "
                   "Number of observables and number of coefficients must be equal");

        std::vector<std::shared_ptr<QasmObs>> obs_vec;
        obs_vec.reserve(key_size);

        for (auto key : obsKeys) {
            auto key_t = reinterpret_cast<int64_t>(key);
            RT_FAIL_IF(static_cast<size_t>(key_t) >= obs_size || key_t < 0,
                       "Invalid observable key");

            auto &&[obs, type] = this->observables_[key_t];
            auto contain_obs = std::find(hamiltonian_valid_obs_types.begin(),
                                         hamiltonian_valid_obs_types.end(), type);

            RT_FAIL_IF(contain_obs == hamiltonian_valid_obs_types.end(),
                       "Invalid observable to construct Hamiltonian; "
                       "NamedObs, HermitianObs and TensorProdObs are only supported");

            obs_vec.push_back(obs);
        }

        this->observables_.push_back(std::make_pair(
            std::make_shared<QasmHamiltonianObs>(QasmHamiltonianObs(coeffs, std::move(obs_vec))),
            ObsType::Hamiltonian));

        return static_cast<ObsIdType>(obs_size);
    }
};
} // namespace Catalyst::Runtime::Device::OpenQasm
