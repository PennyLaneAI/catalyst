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

namespace Catalyst::Runtime::OpenQASM2 {

/**
 * Supported OpenQASM register modes by the builder.
 */
enum class RegisterMode : uint8_t {
    Alloc, // = 0
    Reset,
};

/**
 * Supported OpenQASM register types by the builder.
 */
enum class RegisterType : uint8_t {
    Qubit, // = 0
    Bit,
};

using GateNameT = std::string_view;

/**
 * The map of supported quantum gate names by the OQC device.
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
 * @param wires Wires to apply gate to
 *
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
 * @param qubit Qubit to apply `measure` to
 * @param bit The measurement bit result
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

/**
 * The OpenQASM2 circuit builder interface.
 *
 *
 * @param qregs Quantum registers
 * @param cregs Measurement results registers
 * @param gates Quantum gates
 * @param measurements Quantum measurements
 */
class OpenQASM2Builder {
  protected:
    std::vector<QASMRegister> qregs;
    std::vector<QASMRegister> cregs;
    std::vector<QASMGate> gates;
    std::vector<QASMMeasure> measurements;
    size_t num_qubits;
    bool measure_all = false;

  public:
    explicit OpenQASM2Builder() : measure_all(false), num_qubits(0) {}
    virtual ~OpenQASM2Builder() = default;

    void AddRegisters(const std::string &nameQreg, const size_t &numQubits,
                      const std::string &nameCreg, const size_t &numCbits)
    {
        qregs.emplace_back(RegisterType::Qubit, nameQreg, numQubits);
        num_qubits += numQubits;
        cregs.emplace_back(RegisterType::Bit, nameCreg, numCbits);
    }
    void AddGate(const std::string &name, const std::vector<double> &params_val,
                 const std::vector<size_t> &qubits)
    {
        gates.emplace_back(name, params_val, qubits);
    }
    void AddMeasurement(size_t bit, size_t qubit) { measurements.emplace_back(bit, qubit); }
    void AddMeasurements() { measure_all = true; }
    size_t getNumQubits() { return num_qubits; }
    [[nodiscard]] virtual auto toOpenQASM2(size_t precision = 5) const -> std::string
    {
        std::ostringstream oss;

        // header
        oss << "OPENQASM 2.0"
            << ";\n";
        oss << "include \"qelib1.inc\""
            << ";\n";
        // quantum registers
        oss << qregs[0].toOpenQASM2(RegisterMode::Alloc);

        // measurement results registers
        oss << cregs[0].toOpenQASM2(RegisterMode::Alloc);

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : gates) {
            oss << gate.toOpenQASM2(qregs[0], precision);
        }

        // quantum measures assuming qregs.size() == 1, cregs.size() <= 1
        if (!measure_all) {
            for (auto &m : measurements) {
                oss << m.toOpenQASM2(qregs[0], cregs[0]);
            }
        }
        else {
            oss << "measure " << qregs[0].getName() << " -> " << cregs[0].getName() << ";\n";
        }

        return oss.str();
    }
};

} // namespace Catalyst::Runtime::OpenQASM2
