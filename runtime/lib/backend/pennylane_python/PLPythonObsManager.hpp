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

#pragma once

#include <complex>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Exception.hpp"
#include "Types.h"

namespace Catalyst::Runtime::Device::PLTape {

/**
 * @brief Observable types for PennyLane tape serialization.
 */
enum class PLObsKind : uint8_t {
    Named,
    Hermitian,
    TensorProd,
    Hamiltonian,
};

/**
 * @brief Base class for PennyLane observable representations.
 */
struct PLObs {
    PLObsKind kind;
    virtual ~PLObs() = default;
    [[nodiscard]] virtual auto toJSON(size_t precision = 17) const -> std::string = 0;
};

struct PLNamedObs final : public PLObs {
    std::string name;
    std::vector<size_t> wires;

    PLNamedObs(ObsId id, std::vector<size_t> w) : wires(std::move(w))
    {
        kind = PLObsKind::Named;
        switch (id) {
        case ObsId::PauliX:
            name = "PauliX";
            break;
        case ObsId::PauliY:
            name = "PauliY";
            break;
        case ObsId::PauliZ:
            name = "PauliZ";
            break;
        case ObsId::Hadamard:
            name = "Hadamard";
            break;
        case ObsId::Identity:
            name = "Identity";
            break;
        default:
            RT_FAIL("Unsupported named observable");
        }
    }

    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string override
    {
        std::ostringstream oss;
        oss << "{\"kind\":\"named\",\"name\":\"" << name << "\",\"wires\":[";
        for (size_t i = 0; i < wires.size(); ++i) {
            if (i > 0) oss << ",";
            oss << wires[i];
        }
        oss << "]}";
        return oss.str();
    }
};

struct PLHermitianObs final : public PLObs {
    std::vector<std::complex<double>> matrix;
    std::vector<size_t> wires;

    PLHermitianObs(std::vector<std::complex<double>> m, std::vector<size_t> w)
        : matrix(std::move(m)), wires(std::move(w))
    {
        kind = PLObsKind::Hermitian;
    }

    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string override
    {
        std::ostringstream oss;
        oss << std::setprecision(precision);
        oss << "{\"kind\":\"hermitian\",\"matrix_re\":[";
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (i > 0) oss << ",";
            oss << matrix[i].real();
        }
        oss << "],\"matrix_im\":[";
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (i > 0) oss << ",";
            oss << matrix[i].imag();
        }
        oss << "],\"wires\":[";
        for (size_t i = 0; i < wires.size(); ++i) {
            if (i > 0) oss << ",";
            oss << wires[i];
        }
        oss << "]}";
        return oss.str();
    }
};

struct PLTensorProdObs final : public PLObs {
    std::vector<ObsIdType> sub_obs;

    explicit PLTensorProdObs(std::vector<ObsIdType> obs) : sub_obs(std::move(obs))
    {
        kind = PLObsKind::TensorProd;
    }

    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string override
    {
        std::ostringstream oss;
        oss << "{\"kind\":\"tensor\",\"obs_ids\":[";
        for (size_t i = 0; i < sub_obs.size(); ++i) {
            if (i > 0) oss << ",";
            oss << sub_obs[i];
        }
        oss << "]}";
        return oss.str();
    }
};

struct PLHamiltonianObs final : public PLObs {
    std::vector<double> coeffs;
    std::vector<ObsIdType> sub_obs;

    PLHamiltonianObs(std::vector<double> c, std::vector<ObsIdType> obs)
        : coeffs(std::move(c)), sub_obs(std::move(obs))
    {
        kind = PLObsKind::Hamiltonian;
    }

    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string override
    {
        std::ostringstream oss;
        oss << std::setprecision(precision);
        oss << "{\"kind\":\"hamiltonian\",\"coeffs\":[";
        for (size_t i = 0; i < coeffs.size(); ++i) {
            if (i > 0) oss << ",";
            oss << coeffs[i];
        }
        oss << "],\"obs_ids\":[";
        for (size_t i = 0; i < sub_obs.size(); ++i) {
            if (i > 0) oss << ",";
            oss << sub_obs[i];
        }
        oss << "]}";
        return oss.str();
    }
};

/**
 * @brief Manages observable construction and serialization for the PennyLane
 *        Python device backend.
 */
class PLPythonObsManager {
  private:
    std::vector<std::shared_ptr<PLObs>> observables_;

  public:
    PLPythonObsManager() = default;

    auto createNamedObs(ObsId id, const std::vector<size_t> &wires) -> ObsIdType
    {
        observables_.push_back(std::make_shared<PLNamedObs>(id, wires));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto createHermitianObs(const std::vector<std::complex<double>> &matrix,
                            const std::vector<size_t> &wires) -> ObsIdType
    {
        observables_.push_back(std::make_shared<PLHermitianObs>(matrix, wires));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto createTensorProdObs(const std::vector<ObsIdType> &obs) -> ObsIdType
    {
        observables_.push_back(std::make_shared<PLTensorProdObs>(obs));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto createHamiltonianObs(const std::vector<double> &coeffs,
                              const std::vector<ObsIdType> &obs) -> ObsIdType
    {
        observables_.push_back(std::make_shared<PLHamiltonianObs>(coeffs, obs));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    [[nodiscard]] auto isValidObservable(ObsIdType key) const -> bool
    {
        return key < static_cast<ObsIdType>(observables_.size());
    }

    [[nodiscard]] auto getObservable(ObsIdType key) const -> const std::shared_ptr<PLObs> &
    {
        return observables_.at(key);
    }

    /**
     * @brief Serialize all observables to JSON array string.
     */
    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < observables_.size(); ++i) {
            if (i > 0) oss << ",";
            oss << observables_[i]->toJSON(precision);
        }
        oss << "]";
        return oss.str();
    }
};

} // namespace Catalyst::Runtime::Device::PLTape
