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

#include <algorithm>
#include <array>
#include <complex>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "Exception.hpp"

namespace Catalyst::Runtime::OpenQasm2 {

/**
 * Supported OpenQasm register modes by the builder.
 */
enum class RegisterMode : uint8_t {
    Alloc, // = 0
    Reset,
};

/**
 * Supported OpenQasm register types by the builder.
 */
enum class RegisterType : uint8_t {
    Qubit, // = 0
    Bit,
};

using GateNameT = std::string_view;

/**
 * The map of supported quantum gate names by OpenQasm devices.
 */
constexpr std::array rt_qasm_gate_map = {
    // (RT-GateName, Qasm-GateName)
    std::tuple<GateNameT, GateNameT>{"Identity", "id"},
    std::tuple<GateNameT, GateNameT>{"PauliX", "x"},
    std::tuple<GateNameT, GateNameT>{"PauliY", "y"},
    std::tuple<GateNameT, GateNameT>{"PauliZ", "z"},
    std::tuple<GateNameT, GateNameT>{"CNOT", "cx"},
    std::tuple<GateNameT, GateNameT>{"Toffoli", "ccx"},
    std::tuple<GateNameT, GateNameT>{"CY", "cy"},
    std::tuple<GateNameT, GateNameT>{"CZ", "cz"},
    std::tuple<GateNameT, GateNameT>{"Hadamard", "h"},
    std::tuple<GateNameT, GateNameT>{"S", "s"},
    // std::tuple<GateNameT, GateNameT>{"Adjoint(S)", "sdg"},
    std::tuple<GateNameT, GateNameT>{"T", "t"},
    // std::tuple<GateNameT, GateNameT>{"Adjoint(T)", "tdg"},
    std::tuple<GateNameT, GateNameT>{"SWAP", "swap"},
    std::tuple<GateNameT, GateNameT>{"CSWAP", "cswap"},
    std::tuple<GateNameT, GateNameT>{"RX", "rx"},
    std::tuple<GateNameT, GateNameT>{"RY", "ry"},
    std::tuple<GateNameT, GateNameT>{"RZ", "rz"},
    std::tuple<GateNameT, GateNameT>{"CRX", "crx"},
    std::tuple<GateNameT, GateNameT>{"CRY", "cry"},
    std::tuple<GateNameT, GateNameT>{"CRZ", "crz"},
    std::tuple<GateNameT, GateNameT>{"PhaseShift", "u1"},
    std::tuple<GateNameT, GateNameT>{"U1", "u1"},
    std::tuple<GateNameT, GateNameT>{"U2", "u2"},
    std::tuple<GateNameT, GateNameT>{"U3", "u3"},

};

/**
 * Lookup OpenQasm gate names.
 */
constexpr auto lookup_qasm_gate_name(std::string_view gate_name) -> std::string_view
{
    for (auto &&[gate_qir, gate_qasm] : rt_qasm_gate_map) {
        if (gate_qir == gate_name) {
            return gate_qasm;
        }
    }

    RT_FAIL("The given QIR gate name is not supported by the OpenQASM builder.");
}


/**
 * The OpenQasm quantum register type.
 *
 * @param type Type of the register
 * @param name Name of the register
 * @param size Size of the register
 */
class QASMRegister {
  private:
    const RegisterType type;
    const std::string name;
    size_t size;

  public:
    explicit QASMRegister(RegisterType _type, const std::string &_name, size_t _size)
        : type(_type), name(_name), size(_size)
    {
    }
    ~QASMRegister() = default;

    [[nodiscard]] auto getType() const -> RegisterType { return type; }
    [[nodiscard]] auto getName() const -> std::string { return name; }
    [[nodiscard]] auto getSize() const -> size_t { return size; }

    void updateSize(size_t new_size) { size = new_size; }
    void resetSize() { size = 0; }

    [[nodiscard]] auto toOpenQASM2(RegisterMode mode) const
        -> std::string
    {
        std::ostringstream oss;
        switch (mode) {
        case RegisterMode::Alloc: {
            // qubit[size] name;
            if (type == RegisterType::Qubit) {
                oss << "qubit";
            }
            else if (type == RegisterType::Bit) {
                oss << "bit";
            }
            else {
                RT_FAIL("Unsupported OpenQasm register type");
            }
            oss << "[" << size << "] " << name << ";\n";
            return oss.str();
        }
        case RegisterMode::Reset: {
            // reset name;
            oss << "reset " << name << ";\n";
            return oss.str();
        }
        default:
            RT_FAIL("Unsupported OpenQasm register mode");
        }
    }
};

// /**
//  * The OpenQasm gate type.
//  *
//  * @param name The name of the gate to apply from the list of supported gates
//  * (`rt_qasm_gate_map`)
//  * @param matrix Optional matrix of complex numbers for QubitUnitary
//  * @param params_val Optional list of parameter values for parametric gates
//  * @param params_str Optional list of parameter names for parametric gates
//  * @param wires Wires to apply gate to
//  * @param inverse Indicates whether to use inverse of gate
//  *
//  * @note Parametric gates are currently supported via either their values or names
//  * but not both.
//  */
// class QasmGate {
//   private:
//     const std::string name;
//     const std::vector<double> params_val;
//     const std::vector<std::string> params_str;
//     const std::vector<size_t> wires;
//     const bool inverse;

//   public:
//     explicit QasmGate(const std::string &_name, const std::vector<double> &_params_val,
//                       const std::vector<std::string> &_params_str,
//                       const std::vector<size_t> &_wires, [[maybe_unused]] bool _inverse)
//         : name(lookup_qasm_gate_name(_name)), params_val(_params_val),
//           params_str(_params_str), wires(_wires), inverse(_inverse)
//     {
//         RT_FAIL_IF(!(params_str.empty() || params_val.empty()),
//                    "Parametric gates are currently supported via either their values or names but "
//                    "not both.");
//     }
//     ~QasmGate() = default;

//     [[nodiscard]] auto getName() const -> std::string { return name; }
//     [[nodiscard]] auto getParams() const -> std::vector<double> { return params_val; }
//     [[nodiscard]] auto getParamsStr() const -> std::vector<std::string> { return params_str; }
//     [[nodiscard]] auto getWires() const -> std::vector<size_t> { return wires; }
//     [[nodiscard]] auto getInverse() const -> bool { return inverse; }

//     [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister, size_t precision = 5,
//                                   const std::string &version = "3.0") const -> std::string
//     {
//         std::ostringstream oss;
//         // name(param_1, ..., param_n) qubit_1, ..., qubit_m
//         oss << name;
//         if (!params_val.empty()) {
//             oss << "(";
//             auto iter = params_val.begin();
//             for (; iter != params_val.end() - 1; iter++) {
//                 oss << std::setprecision(precision) << *iter << ", ";
//             }
//             oss << std::setprecision(precision) << *iter << ") ";
//         }
//         else if (!params_str.empty()) {
//             oss << "(";
//             auto iter = params_str.begin();
//             for (; iter != params_str.end() - 1; iter++) {
//                 oss << *iter << ", ";
//             }
//             oss << *iter << ") ";
//         }
//         else {
//             oss << " ";
//         }
//         oss << qregister.toOpenQasm(RegisterMode::Slice, wires) << ";\n";
//         return oss.str();
//     }
// };

// /**
//  * A base class for all Braket/OpenQasm3 observable types.
//  */
// class QasmObs {
//   protected:
//     QasmObs() = default;
//     QasmObs(const QasmObs &) = default;
//     QasmObs(QasmObs &&) = default;
//     QasmObs &operator=(const QasmObs &) = default;
//     QasmObs &operator=(QasmObs &&) noexcept = default;

//   public:
//     virtual ~QasmObs() = default;
//     [[nodiscard]] virtual auto getName() const -> std::string = 0;
//     [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;
//     [[nodiscard]] virtual auto toOpenQasm(const QasmRegister &qregister,
//                                           [[maybe_unused]] size_t precision = 5,
//                                           [[maybe_unused]] const std::string &version = "3.0") const
//         -> std::string = 0;
// };

// /**
//  * A class for Braket/OpenQasm3 named observable (PauliX, PauliY, PauliZ, Hadamard, etc.)
//  */
// class QasmNamedObs final : public QasmObs {
//   private:
//     const std::string name;
//     const std::vector<size_t> wires;

//   public:
//     explicit QasmNamedObs(const std::string &_name, std::vector<size_t> _wires)
//         : name(lookup_qasm_gate_name(_name)), wires(_wires)
//     {
//     }

//     [[nodiscard]] auto getName() const -> std::string override { return name; }
//     [[nodiscard]] auto getWires() const -> std::vector<size_t> override { return wires; }

//     [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister,
//                                   [[maybe_unused]] size_t precision = 5,
//                                   [[maybe_unused]] const std::string &version = "3.0") const
//         -> std::string override
//     {
//         std::ostringstream oss;
//         oss << name << "(" << qregister.toOpenQasm(RegisterMode::Slice, wires) << ")";
//         return oss.str();
//     }
// };

// /**
//  * The OpenQasm circuit builder interface.
//  *
//  * @note Only one user-specified quantum register is currently supported.
//  * @note User-specified measurement results registers are supported.
//  *
//  * @param qregs Quantum registers
//  * @param bregs Measurement results registers
//  * @param gates Quantum gates
//  */
// class OpenQasmBuilder {
//   protected:
//     std::vector<QasmRegister> qregs;
//     std::vector<QasmRegister> bregs;
//     std::vector<QasmGate> gates;
//     // std::vector<QasmMeasure> measures;
//     size_t num_qubits;
//     size_t num_bits;

//   public:
//     explicit OpenQasmBuilder() : num_qubits(0), num_bits(0) {}
//     virtual ~OpenQasmBuilder() = default;

//     [[nodiscard]] auto getNumQubits() const -> size_t { return num_qubits; }
//     [[nodiscard]] auto getNumBits() const -> size_t { return num_bits; }
//     [[nodiscard]] auto getQubits() const -> std::vector<QasmRegister> { return qregs; }

//     void Register(RegisterType type, const std::string &name, size_t size)
//     {
//         switch (type) {
//         case RegisterType::Qubit:
//             qregs.emplace_back(type, name, size);
//             num_qubits += size;
//             break;
//         case RegisterType::Bit:
//             bregs.emplace_back(type, name, size);
//             num_bits += size;
//             break;
//         default:
//             RT_FAIL("Unsupported OpenQasm register type");
//         }
//     }

//     void Gate(const std::string &name, const std::vector<double> &params_val,
//               const std::vector<std::string> &params_str, const std::vector<size_t> &wires,
//               [[maybe_unused]] bool inverse)
//     {
//         gates.emplace_back(name, params_val, params_str, wires, inverse);
//     }
//     void Gate(const std::vector<std::complex<double>> &matrix, const std::vector<size_t> &wires,
//               [[maybe_unused]] bool inverse)
//     {
//         gates.emplace_back(matrix, wires, inverse);
//     }
//     // void Measure(size_t bit, size_t wire) { measures.emplace_back(bit, wire); }

//     [[nodiscard]] virtual auto toOpenQasm(size_t precision = 5,
//                                           const std::string &version = "3.0") const -> std::string
//     {
//         RT_FAIL_IF(qregs.size() != 1, "Invalid number of quantum registers; Only one quantum "
//                                       "register is currently supported.");

//         RT_FAIL_IF(bregs.size() > 1,
//                    "Invalid number of measurement results registers; At most one measurement"
//                    "results register is currently supported.");

//         std::ostringstream oss;

//         // header
//         oss << "OPENQASM " << version << ";\n";

//         // quantum registers
//         for (auto &qreg : qregs) {
//             oss << qreg.toOpenQasm(RegisterMode::Alloc);
//         }

//         // measurement results registers
//         for (auto &breg : bregs) {
//             oss << breg.toOpenQasm(RegisterMode::Alloc);
//         }

//         // quantum gates assuming qregs.size() == 1
//         for (auto &gate : gates) {
//             oss << gate.toOpenQasm(qregs[0], precision);
//         }

//         // quantum measures assuming qregs.size() == 1, bregs.size() <= 1
//         // for (auto &m : measures) {
//         //     if (bregs.empty()) {
//         //         oss << m.toOpenQasm(qregs[0]);
//         //     }
//         //     else {
//         //         oss << m.toOpenQasm(bregs[0], qregs[0]);
//         //     }
//         // }

//         // reset quantum registers
//         for (auto &qreg : qregs) {
//             oss << qreg.toOpenQasm(RegisterMode::Reset);
//         }

//         return oss.str();
//     }
// };

} // namespace Catalyst::Runtime::OpenQasm2
