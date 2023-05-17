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
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "Exception.hpp"

namespace Catalyst::Runtime::Device::OpenQasm {

enum class VariableType : uint8_t {
    Float, // = 0
};

enum class RegisterMode : uint8_t {
    Alloc, // = 0
    Slice,
    Reset,
};

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
    std::tuple<GateNameT, GateNameT>{"PauliX", "x"},
    std::tuple<GateNameT, GateNameT>{"PauliY", "y"},
    std::tuple<GateNameT, GateNameT>{"PauliZ", "z"},
    std::tuple<GateNameT, GateNameT>{"Hadamard", "h"},
    std::tuple<GateNameT, GateNameT>{"S", "s"},
    std::tuple<GateNameT, GateNameT>{"T", "t"},
    std::tuple<GateNameT, GateNameT>{"CNOT", "cnot"},
    std::tuple<GateNameT, GateNameT>{"CZ", "cz"},
    std::tuple<GateNameT, GateNameT>{"SWAP", "swap"},
    std::tuple<GateNameT, GateNameT>{"PhaseShift", "phaseshift"},
    std::tuple<GateNameT, GateNameT>{"RX", "rx"},
    std::tuple<GateNameT, GateNameT>{"RY", "ry"},
    std::tuple<GateNameT, GateNameT>{"RZ", "rz"},
    std::tuple<GateNameT, GateNameT>{"CSWAP", "cswap"},
    std::tuple<GateNameT, GateNameT>{"PSWAP", "pswap"},
    std::tuple<GateNameT, GateNameT>{"ISWAP", "iswap"},
    std::tuple<GateNameT, GateNameT>{"Toffoli", "ccnot"},
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
 * The OpenQasm variable type.
 *
 * @param type Type of the variable
 * @param Name Name of the register
 */
class QasmVariable {
  private:
    const VariableType type;
    const std::string name;

  public:
    explicit QasmVariable(VariableType _type, const std::string &_name) : type(_type), name(_name)
    {
    }
    ~QasmVariable() = default;

    [[nodiscard]] auto getType() const -> VariableType { return this->type; }
    [[nodiscard]] auto getName() const -> std::string { return this->name; }

    auto toOpenQasm([[maybe_unused]] const std::string &version = "3.0") const -> std::string
    {
        std::ostringstream oss;
        switch (this->type) {
        case VariableType::Float: {
            oss << "input float " << name << ";\n";
            return oss.str();
        }
        default:
            RT_FAIL("Unsupported OpenQasm variable type");
        }
    }
};

/**
 * The OpenQasm quantum register type.
 *
 * @param type Type of the register
 * @param name Name of the register
 * @param size Size of the register
 */
class QasmRegister {
  private:
    const RegisterType type;
    const std::string name;
    size_t size;

  public:
    explicit QasmRegister(RegisterType _type, const std::string &_name, size_t _size)
        : type(_type), name(_name), size(_size)
    {
    }
    ~QasmRegister() = default;

    [[nodiscard]] auto getType() const -> RegisterType { return this->type; }
    [[nodiscard]] auto getName() const -> std::string { return this->name; }
    [[nodiscard]] auto getSize() const -> size_t { return this->size; }

    void updateSize(size_t new_size) { this->size = new_size; }
    void resetSize() { this->size = 0; }
    auto isValidSlice(const std::vector<size_t> &slice) const -> bool
    {
        if (slice.empty()) {
            return false;
        }

        return std::all_of(slice.begin(), slice.end(),
                           [this](auto qubit) { return this->size > qubit; });
    }

    auto toOpenQasm(RegisterMode mode, [[maybe_unused]] const std::vector<size_t> &slice = {},
                    [[maybe_unused]] const std::string &version = "3.0") const -> std::string
    {
        std::ostringstream oss;
        switch (mode) {
        case RegisterMode::Alloc: {
            // qubit[size] name;
            if (this->type == RegisterType::Qubit) {
                oss << "qubit";
            }
            else if (this->type == RegisterType::Bit) {
                oss << "bit";
            }
            else {
                RT_FAIL("Unsupported OpenQasm register type");
            }
            oss << "[" << this->size << "] " << this->name << ";\n";
            return oss.str();
        }
        case RegisterMode::Reset: {
            // reset name;
            oss << "reset " << this->name << ";\n";
            return oss.str();
        }
        case RegisterMode::Slice: {
            // name[slice_0], ..., name[slice_n]
            RT_ASSERT(isValidSlice(slice));
            auto iter = slice.begin();
            for (; iter != slice.end() - 1; iter++) {
                oss << this->name << "[" << *iter << "], ";
            }
            oss << this->name << "[" << *iter << "]";
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
class QasmGate {
  private:
    const std::string name;
    const std::vector<double> params_val;
    const std::vector<std::string> params_str;
    const std::vector<size_t> wires;
    const bool inverse;

  public:
    explicit QasmGate(const std::string &_name, const std::vector<double> &_params_val,
                      const std::vector<std::string> &_params_str,
                      const std::vector<size_t> &_wires, [[maybe_unused]] bool _inverse)
        : name(lookup_qasm_gate_name(_name)), params_val(_params_val), params_str(_params_str),
          wires(_wires), inverse(_inverse)
    {
        RT_FAIL_IF(!(params_str.empty() || params_val.empty()),
                   "Parametric gates are currently supported via either their values or names but "
                   "not both.");
    }
    ~QasmGate() = default;

    [[nodiscard]] auto getName() const -> std::string { return this->name; }
    [[nodiscard]] auto getParams() const -> std::vector<double> { return this->params_val; }
    [[nodiscard]] auto getParamsStr() const -> std::vector<std::string> { return this->params_str; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> { return this->wires; }
    [[nodiscard]] auto getInverse() const -> bool { return this->inverse; }

    auto toOpenQasm(const QasmRegister &qregister, [[maybe_unused]] size_t precision = 5,
                    [[maybe_unused]] const std::string &version = "3.0") const -> std::string
    {
        // name(param_1, ..., param_n) qubit_1, ..., qubit_m
        std::ostringstream oss;
        oss << this->name;
        if (!this->params_val.empty()) {
            oss << "(";
            auto iter = this->params_val.begin();
            for (; iter != this->params_val.end() - 1; iter++) {
                oss << std::setprecision(precision) << *iter << ", ";
            }
            oss << std::setprecision(precision) << *iter << ") ";
        }
        else if (!this->params_str.empty()) {
            oss << "(";
            auto iter = this->params_str.begin();
            for (; iter != this->params_str.end() - 1; iter++) {
                oss << *iter << ", ";
            }
            oss << *iter << ") ";
        }
        else {
            oss << " ";
        }
        oss << qregister.toOpenQasm(RegisterMode::Slice, this->wires) << ";\n";
        return oss.str();
    }
};

/**
 * The OpenQasm measure type.
 *
 * @param bit The measurement bit result
 * @param wire Wire to apply `measure` to
 */
class QasmMeasure {
  private:
    const size_t bit;
    const size_t wire;

  public:
    explicit QasmMeasure(size_t _bit, size_t _wire) : bit(_bit), wire(_wire) {}
    ~QasmMeasure() = default;

    [[nodiscard]] auto getBit() const -> size_t { return this->bit; }
    [[nodiscard]] auto getWire() const -> size_t { return this->wire; }

    auto toOpenQasm(const QasmRegister &qregister,
                    [[maybe_unused]] const std::string &version = "3.0") const -> std::string
    {
        // measure wire
        std::ostringstream oss;
        oss << "measure " << qregister.toOpenQasm(RegisterMode::Slice, {this->wire}) << ";\n";
        return oss.str();
    }
    auto toOpenQasm(const QasmRegister &bregister, const QasmRegister &qregister,
                    [[maybe_unused]] const std::string &version = "3.0") const -> std::string
    {
        // bit = measure wire
        std::ostringstream oss;
        oss << bregister.toOpenQasm(RegisterMode::Slice, {this->bit}) << " = measure "
            << qregister.toOpenQasm(RegisterMode::Slice, {this->wire}) << ";\n";
        return oss.str();
    }
};

/**
 * The OpenQasm circuit builder.
 *
 * @param qregs Quantum registers
 * @param bregs Measurement results registers
 * @param gates Quantum gates
 * @param measures Quantum measures
 */
class OpenQasmBuilder {
  private:
    std::vector<QasmVariable> vars;
    std::vector<QasmRegister> qregs;
    std::vector<QasmRegister> bregs;
    std::vector<QasmGate> gates;
    std::vector<QasmMeasure> measures;
    size_t num_qubits;
    size_t num_bits;

  public:
    explicit OpenQasmBuilder() : num_qubits(0), num_bits(0) {}
    ~OpenQasmBuilder() = default;

    [[nodiscard]] auto getNumQubits() -> size_t { return this->num_qubits; }
    [[nodiscard]] auto getNumBits() -> size_t { return this->num_bits; }

    void Register(RegisterType type, const std::string &name, size_t size)
    {
        switch (type) {
        case RegisterType::Qubit:
            this->qregs.emplace_back(type, name, size);
            this->num_qubits += size;
            break;
        case RegisterType::Bit:
            this->bregs.emplace_back(type, name, size);
            this->num_bits += size;
            break;
        default:
            RT_FAIL("Unsupported OpenQasm register type");
        }
    }

    void Gate(const std::string &name, const std::vector<double> &params_val,
              const std::vector<std::string> &params_str, const std::vector<size_t> &wires,
              [[maybe_unused]] bool inverse)
    {
        this->gates.emplace_back(name, params_val, params_str, wires, inverse);

        for (auto &param : params_str) {
            this->vars.emplace_back(VariableType::Float, param);
        }
    }
    void Measure(size_t bit, size_t wire) { measures.emplace_back(bit, wire); }

    auto toOpenQasm(size_t precision = 5, const std::string &version = "3.0") const -> std::string
    {
        RT_FAIL_IF(this->qregs.size() != 1, "Invalid number of quantum registers; Only one quantum "
                                            "register is currently supported.");

        RT_FAIL_IF(this->bregs.size() > 1,
                   "Invalid number of measurement results registers; At most one measurement"
                   "results register is currently supported.");

        std::ostringstream oss;

        // header
        oss << "OPENQASM " << version << ";\n";

        // variables
        for (auto &var : this->vars) {
            oss << var.toOpenQasm();
        }

        // quantum registers
        for (auto &qreg : this->qregs) {
            oss << qreg.toOpenQasm(RegisterMode::Alloc);
        }

        // measurement results registers
        for (auto &breg : this->bregs) {
            oss << breg.toOpenQasm(RegisterMode::Alloc);
        }

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : this->gates) {
            oss << gate.toOpenQasm(this->qregs[0], precision);
        }

        // quantum measures assuming qregs.size() == 1, bregs.size() <= 1
        for (auto &m : this->measures) {
            if (bregs.empty()) {
                oss << m.toOpenQasm(this->qregs[0]);
            }
            else {
                oss << m.toOpenQasm(this->bregs[0], this->qregs[0]);
            }
        }

        // reset quantum registers
        for (auto &qreg : this->qregs) {
            oss << qreg.toOpenQasm(RegisterMode::Reset);
        }

        return oss.str();
    }
};
} // namespace Catalyst::Runtime::Device::OpenQasm
