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

#include <string>
#include <vector>

#include "Types.h"
#include "Utils.hpp"

namespace Catalyst::Runtime {
/**
 * @brief The CacheManager caches the entire operations and observables of
 * a program at runtime.
 *
 * One direct use case of this functionality is explored to compute gradient
 * of a circuit with taking advantage of gradient methods provided by
 * simulators.
 */
class CacheManager {
  protected:
    // Operations Data
    std::vector<std::string> ops_names_{};
    std::vector<std::vector<double>> ops_params_{};
    std::vector<std::vector<size_t>> ops_wires_{};
    std::vector<bool> ops_inverses_{};
    std::vector<std::vector<size_t>> ops_controlled_wires_{};
    std::vector<std::vector<bool>> ops_controlled_values_{};

    // Observables Data
    std::vector<ObsIdType> obs_keys_{};
    std::vector<MeasurementsT> obs_callees_{};

    // Number of parameters
    size_t num_params_{0};

  public:
    CacheManager() = default;
    ~CacheManager() = default;

    CacheManager(const CacheManager &) = delete;
    CacheManager &operator=(const CacheManager &) = delete;
    CacheManager(CacheManager &&) = delete;
    CacheManager &operator=(CacheManager &&) = delete;

    /**
     * Reset cached gates
     */
    void Reset()
    {
        this->ops_names_.clear();
        this->ops_params_.clear();
        this->ops_wires_.clear();
        this->ops_inverses_.clear();
        this->ops_controlled_wires_.clear();
        this->ops_controlled_values_.clear();

        this->obs_keys_.clear();
        this->obs_callees_.clear();

        this->num_params_ = 0;
    }

    /**
     * @brief Add a new operation to the list of cached gates.
     *
     * @param name Name of the given gate
     * @param params Parameters of the gate
     * @param wires Wires the gate acts on
     * @param inverse If true, inverse of the gate is applied
     */
    void addOperation(const std::string &name, const std::vector<double> &params,
                      const std::vector<size_t> &dev_wires, bool inverse)
    {
        this->ops_names_.push_back(name);
        this->ops_params_.push_back(params);

        std::vector<size_t> wires_ul;
        wires_ul.reserve(dev_wires.size());
        std::transform(dev_wires.begin(), dev_wires.end(), std::back_inserter(wires_ul),
                       [](auto w) { return static_cast<size_t>(w); });

        this->ops_wires_.push_back(wires_ul);
        this->ops_inverses_.push_back(inverse);

        this->num_params_ += params.size();

        this->ops_controlled_wires_.push_back({});
        this->ops_controlled_values_.push_back({});
    }

    /**
     * @brief Add a new operation to the list of cached gates.
     *
     * @param name Name of the given gate
     * @param params Parameters of the gate
     * @param wires Wires the gate acts on
     * @param inverse If true, inverse of the gate is applied
     */
    void addOperation2(const std::string &name, const std::vector<double> &params,
                       const std::vector<size_t> &dev_wires, bool inverse,
                       const std::vector<size_t> &dev_controlled_wires,
                       const std::vector<bool> &controlled_values)
    {
        this->ops_names_.push_back(name);
        this->ops_params_.push_back(params);

        std::vector<size_t> wires_ul;
        wires_ul.reserve(dev_wires.size());
        std::transform(dev_wires.begin(), dev_wires.end(), std::back_inserter(wires_ul),
                       [](auto w) { return static_cast<size_t>(w); });
        this->ops_wires_.push_back(wires_ul);

        std::vector<size_t> controlled_wires_ul;
        controlled_wires_ul.reserve(dev_controlled_wires.size());
        std::transform(dev_controlled_wires.begin(), dev_controlled_wires.end(), std::back_inserter(controlled_wires_ul),
                       [](auto w) { return static_cast<size_t>(w); });
        this->ops_controlled_wires_.push_back(controlled_wires_ul);
        this->ops_controlled_values_.push_back(controlled_values);

        this->ops_inverses_.push_back(inverse);

        this->num_params_ += params.size();
    }

    /**
     * @brief Add a new observable to the list of cached gates.
     *
     * @param id The observable key created by LObsManager()
     * @param callee The measurement operation
     */
    void addObservable(const ObsIdType id, const MeasurementsT &callee = MeasurementsT::None)
    {
        this->obs_keys_.push_back(id);
        this->obs_callees_.push_back(callee);
    }

    /**
     * @brief Get a reference to observables keys.
     */
    auto getObservablesKeys() -> const std::vector<ObsIdType> & { return this->obs_keys_; }

    /**
     * @brief Get a reference to observables callees.
     */
    auto getObservablesCallees() -> const std::vector<MeasurementsT> &
    {
        return this->obs_callees_;
    }

    /**
     * @brief Get a reference to operations names.
     */
    auto getOperationsNames() -> const std::vector<std::string> & { return this->ops_names_; }

    /**
     * @brief Get a a reference to operations parameters.
     */
    auto getOperationsParameters() -> const std::vector<std::vector<double>> &
    {
        return this->ops_params_;
    }

    /**
     * @brief Get a a reference to operations wires.
     */
    auto getOperationsWires() -> const std::vector<std::vector<size_t>> &
    {
        return this->ops_wires_;
    }

    /**
     * @brief Get a a reference to operation controlled wires.
     */
    auto getOperationsControlledWires() -> const std::vector<std::vector<size_t>> &
    {
        return this->ops_controlled_wires_;
    }

    /**
     * @brief Get a a reference to operation controlled values.
     */
    auto getOperationsControlledValues() -> const std::vector<std::vector<bool>> &
    {
        return this->ops_controlled_values_;
    }

    /**
     * @brief Get a reference to operations inverses.
     */
    auto getOperationsInverses() -> const std::vector<bool> & { return this->ops_inverses_; }

    /**
     * @brief Get total number of cached gates.
     */
    [[nodiscard]] auto getNumGates() const -> size_t
    {
        return this->ops_names_.size() + this->obs_keys_.size();
    }

    /**
     * @brief Get number of operations.
     */
    [[nodiscard]] auto getNumOperations() const -> size_t { return this->ops_names_.size(); }

    /**
     * @brief Get number of observables.
     */
    [[nodiscard]] auto getNumObservables() const -> size_t { return this->obs_keys_.size(); }

    /**
     * @brief Get total number of cached gates.
     */
    [[nodiscard]] auto getNumParams() const -> size_t { return this->num_params_; }
};
} // namespace Catalyst::Runtime
