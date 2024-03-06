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

    [[nodiscard]] auto toOpenQASM2(RegisterMode mode) const -> std::string
    {
        std::ostringstream oss;
        switch (mode) {
        case RegisterMode::Alloc: {
            // qubit[size] name;
            if (type == RegisterType::Qubit) {
                oss << "qreg ";
            }
            else if (type == RegisterType::Bit) {
                oss << "creg ";
            }
            else {
                RT_FAIL("Unsupported OpenQasm register type");
            }
            oss << name << "[" << size << "]"
                << ";\n";
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

/**
 * The OpenQasm gate type.
 *
 * @param name The name of the gate to apply from the list of supported gates
 * (`rt_qasm_gate_map`)
 * @param params_val Optional list of parameter values for parametric gates
 * @param params_str Optional list of parameter names for parametric gates
 * @param wires Wires to apply gate to
 * @param inverse Indicates whether to use inverse of gate
 *
 * @note Parametric gates are currently supported via either their values or names
 * but not both.
 */
class QASMGate {
  private:
    const std::string name;
    const std::vector<double> params_val;
    const std::vector<size_t> wires;

  public:
    explicit QASMGate(const std::string &_name, const std::vector<double> &_params_val,
                      const std::vector<size_t> &_wires)
        : name(lookup_qasm_gate_name(_name)), params_val(_params_val), wires(_wires)
    {
    }
    ~QASMGate() = default;

    [[nodiscard]] auto getName() const -> std::string { return name; }
    [[nodiscard]] auto getParams() const -> std::vector<double> { return params_val; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> { return wires; }

    [[nodiscard]] auto toOpenQASM2(const QASMRegister &qregister, size_t precision = 5) const
        -> std::string
    {
        std::ostringstream oss;
        // name(param_1, ..., param_n) qubit_1, ..., qubit_m
        oss << name;
        if (!params_val.empty()) {
            oss << "(";
            auto iter = params_val.begin();
            for (; iter != params_val.end() - 1; iter++) {
                oss << std::setprecision(precision) << *iter << ", ";
            }
            oss << std::setprecision(precision) << *iter << ") ";
        }
        else {
            oss << " ";
        }
        auto iter = wires.begin();
        for (; iter != wires.end() - 1; iter++) {
            oss << qregister.getName() << "[" << *iter << "], ";
        }
        oss << qregister.getName() << "[" << *iter << "]"
            << ";\n";
        return oss.str();
    }
};

/**
 * The OpenQasm measure type.
 *
 * @param bit The measurement bit result
 * @param wire Wire to apply `measure` to
 */
class QASMMeasure {
  private:
    const size_t qubit;
    const size_t bit;

  public:
    explicit QASMMeasure(size_t _qubit, size_t _bit) : qubit(_qubit), bit(_bit) {}
    ~QASMMeasure() = default;

    [[nodiscard]] auto getQubit() const -> size_t { return qubit; }
    [[nodiscard]] auto getBit() const -> size_t { return qubit; }

    [[nodiscard]] auto toOpenQASM2(const QASMRegister &qregister,
                                   const QASMRegister &cregister) const -> std::string
    {
        // measure wire
        std::ostringstream oss;
        oss << "measure " << qregister.getName() << "[" << qubit << "] -> " << cregister.getName()
            << "[" << bit << "];\n";
        return oss.str();
    }
};

std::string MeasureAll(const QASMRegister &qregister, const QASMRegister &cregister)
{
    // measure wire
    std::ostringstream oss;
    oss << "measure " << qregister.getName() << " -> " << cregister.getName() << ";\n";
    return oss.str();
};

/**
 * The OpenQasm circuit builder interface.
 *
 * @note Only one user-specified quantum register is currently supported.
 * @note User-specified measurement results registers are supported.
 *
 * @param qregs Quantum registers
 * @param cregs Measurement results registers
 * @param gates Quantum gates
 */
class OpenQASM2Builder {
  protected:
    QASMRegister qreg;
    QASMRegister creg;
    std::vector<QASMGate> gates;
    std::vector<QASMMeasure> measures;

  public:
    explicit OpenQASM2Builder(QASMRegister _qreg, QASMRegister _creg) : qreg(_qreg), creg(_creg) {}
    virtual ~OpenQASM2Builder() = default;

    void Gate(const std::string &name, const std::vector<double> &params_val,
              const std::vector<size_t> &qubits)
    {
        gates.emplace_back(name, params_val, qubits);
    }
    void Measure(size_t bit, size_t qubit) { measures.emplace_back(bit, qubit); }

    [[nodiscard]] virtual auto toOpenQASM2(size_t precision = 5) const -> std::string
    {
        std::ostringstream oss;

        // header
        oss << "OPENQASM 2.0" << ";\n";
        oss << "include \"qelib1.inc\"" << ";\n";
        // quantum registers
        oss << qreg.toOpenQASM2(RegisterMode::Alloc);

        // measurement results registers
        oss << creg.toOpenQASM2(RegisterMode::Alloc);

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : gates) {
            oss << gate.toOpenQASM2(qreg, precision);
        }

        // quantum measures assuming qregs.size() == 1, cregs.size() <= 1
        for (auto &m : measures) {
            oss << m.toOpenQASM2(creg, qreg);
        }

        return oss.str();
    }
};

} // namespace Catalyst::Runtime::OpenQasm2
