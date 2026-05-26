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

#include <algorithm>
#include <complex>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Exception.hpp"

namespace Catalyst::Runtime::Device::PLTape {

/**
 * @brief Represents a single quantum gate operation in the tape.
 */
struct TapeGate {
    std::string name;
    std::vector<double> params;
    std::vector<size_t> wires;
    bool inverse;
    std::vector<size_t> controlled_wires;
    std::vector<bool> controlled_values;
};

/**
 * @brief Represents a unitary matrix gate in the tape.
 */
struct TapeMatrixGate {
    std::vector<std::complex<double>> matrix;
    std::vector<size_t> wires;
    bool inverse;
    std::vector<size_t> controlled_wires;
    std::vector<bool> controlled_values;
};

/**
 * @brief Builds a JSON representation of a PennyLane QuantumScript (tape).
 *
 * Unlike OpenQasmBuilder which serializes to an OpenQASM string, this builder
 * serializes the accumulated operations to a JSON format that the Python module
 * can deserialize back into a PennyLane QuantumScript with full fidelity.
 *
 * The JSON schema:
 * {
 *   "num_qubits": <int>,
 *   "ops": [
 *     {"name": "RX", "params": [0.5], "wires": [0], "inverse": false,
 *      "ctrl_wires": [], "ctrl_values": []},
 *     ...
 *   ],
 *   "matrix_ops": [
 *     {"matrix_re": [...], "matrix_im": [...], "wires": [0,1],
 *      "inverse": false, "ctrl_wires": [], "ctrl_values": []},
 *     ...
 *   ]
 * }
 */
class PLTapeBuilder {
  private:
    std::vector<TapeGate> gates_;
    std::vector<TapeMatrixGate> matrix_gates_;
    size_t num_qubits_{0};

  public:
    explicit PLTapeBuilder() = default;
    ~PLTapeBuilder() = default;

    [[nodiscard]] auto getNumQubits() const -> size_t { return num_qubits_; }

    void setNumQubits(size_t n) { num_qubits_ = n; }

    void Gate(const std::string &name, const std::vector<double> &params,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<size_t> &controlled_wires = {},
              const std::vector<bool> &controlled_values = {})
    {
        gates_.push_back({name, params, wires, inverse, controlled_wires, controlled_values});
    }

    void MatrixGate(const std::vector<std::complex<double>> &matrix,
                    const std::vector<size_t> &wires, bool inverse,
                    const std::vector<size_t> &controlled_wires = {},
                    const std::vector<bool> &controlled_values = {})
    {
        matrix_gates_.push_back({matrix, wires, inverse, controlled_wires, controlled_values});
    }

    void Measure(size_t wire)
    {
        // Mid-circuit measurement: encode as a special named gate
        gates_.push_back({"MidCircuitMeasure", {}, {wire}, false, {}, {}});
    }

    void reset()
    {
        gates_.clear();
        matrix_gates_.clear();
        num_qubits_ = 0;
    }

    /**
     * @brief Serialize accumulated tape to JSON for the Python module.
     */
    [[nodiscard]] auto toJSON(size_t precision = 17) const -> std::string
    {
        std::ostringstream oss;
        oss << std::setprecision(precision);
        oss << "{\"num_qubits\":" << num_qubits_;

        // Named gates
        oss << ",\"ops\":[";
        for (size_t i = 0; i < gates_.size(); ++i) {
            if (i > 0) oss << ",";
            const auto &g = gates_[i];
            oss << "{\"name\":\"" << g.name << "\",\"params\":[";
            for (size_t j = 0; j < g.params.size(); ++j) {
                if (j > 0) oss << ",";
                oss << g.params[j];
            }
            oss << "],\"wires\":[";
            for (size_t j = 0; j < g.wires.size(); ++j) {
                if (j > 0) oss << ",";
                oss << g.wires[j];
            }
            oss << "],\"inverse\":" << (g.inverse ? "true" : "false");
            oss << ",\"ctrl_wires\":[";
            for (size_t j = 0; j < g.controlled_wires.size(); ++j) {
                if (j > 0) oss << ",";
                oss << g.controlled_wires[j];
            }
            oss << "],\"ctrl_values\":[";
            for (size_t j = 0; j < g.controlled_values.size(); ++j) {
                if (j > 0) oss << ",";
                oss << (g.controlled_values[j] ? "true" : "false");
            }
            oss << "]}";
        }
        oss << "]";

        // Matrix gates
        oss << ",\"matrix_ops\":[";
        for (size_t i = 0; i < matrix_gates_.size(); ++i) {
            if (i > 0) oss << ",";
            const auto &mg = matrix_gates_[i];
            oss << "{\"matrix_re\":[";
            for (size_t j = 0; j < mg.matrix.size(); ++j) {
                if (j > 0) oss << ",";
                oss << mg.matrix[j].real();
            }
            oss << "],\"matrix_im\":[";
            for (size_t j = 0; j < mg.matrix.size(); ++j) {
                if (j > 0) oss << ",";
                oss << mg.matrix[j].imag();
            }
            oss << "],\"wires\":[";
            for (size_t j = 0; j < mg.wires.size(); ++j) {
                if (j > 0) oss << ",";
                oss << mg.wires[j];
            }
            oss << "],\"inverse\":" << (mg.inverse ? "true" : "false");
            oss << ",\"ctrl_wires\":[";
            for (size_t j = 0; j < mg.controlled_wires.size(); ++j) {
                if (j > 0) oss << ",";
                oss << mg.controlled_wires[j];
            }
            oss << "],\"ctrl_values\":[";
            for (size_t j = 0; j < mg.controlled_values.size(); ++j) {
                if (j > 0) oss << ",";
                oss << (mg.controlled_values[j] ? "true" : "false");
            }
            oss << "]}";
        }
        oss << "]";

        oss << "}";
        return oss.str();
    }
};

} // namespace Catalyst::Runtime::Device::PLTape
