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

namespace Catalyst::Runtime::Device::OpenQasm {

/**
 * Types of the OpenQasm builder and runner.
 */
enum class BuilderType : uint8_t {
    Common, // = 0
    BraketRemote,
    BraketLocal,
};

/**
 * Supported OpenQasm variables by the builder.
 */
enum class VariableType : uint8_t {
    Float, // = 0
};

/**
 * Supported OpenQasm register modes by the builder.
 */
enum class RegisterMode : uint8_t {
    Alloc, // = 0
    Slice,
    Name,
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
    std::tuple<GateNameT, GateNameT>{"Identity", "i"},
    std::tuple<GateNameT, GateNameT>{"PauliX", "x"},
    std::tuple<GateNameT, GateNameT>{"PauliY", "y"},
    std::tuple<GateNameT, GateNameT>{"PauliZ", "z"},
    std::tuple<GateNameT, GateNameT>{"Hadamard", "h"},
    std::tuple<GateNameT, GateNameT>{"S", "s"},
    std::tuple<GateNameT, GateNameT>{"T", "t"},
    std::tuple<GateNameT, GateNameT>{"CNOT", "cnot"},
    std::tuple<GateNameT, GateNameT>{"CY", "cy"},
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

    [[nodiscard]] auto getType() const -> VariableType { return type; }
    [[nodiscard]] auto getName() const -> std::string { return name; }

    [[nodiscard]] auto toOpenQasm([[maybe_unused]] const std::string &version = "3.0") const
        -> std::string
    {
        std::ostringstream oss;
        switch (type) {
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

    [[nodiscard]] auto getType() const -> RegisterType { return type; }
    [[nodiscard]] auto getName() const -> std::string { return name; }
    [[nodiscard]] auto getSize() const -> size_t { return size; }

    void updateSize(size_t new_size) { size = new_size; }
    void resetSize() { size = 0; }
    [[nodiscard]] auto isValidSlice(const std::vector<size_t> &slice) const -> bool
    {
        if (slice.empty()) {
            return false;
        }

        return std::all_of(slice.begin(), slice.end(), [this](auto qubit) { return size > qubit; });
    }

    [[nodiscard]] auto toOpenQasm(RegisterMode mode,
                                  [[maybe_unused]] const std::vector<size_t> &slice = {},
                                  [[maybe_unused]] const std::string &version = "3.0") const
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
        case RegisterMode::Slice: {
            // name[slice_0], ..., name[slice_n]
            RT_ASSERT(isValidSlice(slice));
            auto iter = slice.begin();
            for (; iter != slice.end() - 1; iter++) {
                oss << name << "[" << *iter << "], ";
            }
            oss << name << "[" << *iter << "]";
            return oss.str();
        }
        case RegisterMode::Name: {
            // name
            oss << name;
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
 * The OpenQasm Matrix Builder for the following matrix data-types:
 * - `std::vector<double>`
 * - `std::vector<std::complex<double>>`
 *
 * @note It doesn't store the given matrix.
 */
struct MatrixBuilder {
    [[nodiscard]] static auto toOpenQasm(const std::vector<std::complex<double>> &matrix,
                                         size_t num_cols, size_t precision = 5,
                                         [[maybe_unused]] const std::string &version = "3.0")
        -> std::string
    {
        constexpr std::complex<double> zero{0, 0};
        size_t index{0};
        std::ostringstream oss;
        oss << "[[";
        for (const auto &c : matrix) {
            if (index == num_cols) {
                oss << "], [";
                index = 0;
            }
            else if (index) {
                oss << ", ";
            }
            index++;

            if (c == zero) {
                oss << "0";
                continue;
            }
            oss << std::setprecision(precision) << c.real();
            oss << std::setprecision(precision) << (c.imag() < 0 ? "" : "+") << c.imag() << "im";
        }
        oss << "]]";
        return oss.str();
    }

    [[nodiscard]] static auto toOpenQasm(const std::vector<double> &matrix, size_t num_cols,
                                         size_t precision = 5,
                                         [[maybe_unused]] const std::string &version = "3.0")
        -> std::string
    {
        size_t index{0};

        std::ostringstream oss;
        oss << "[[";
        for (const auto &c : matrix) {
            if (index == num_cols) {
                oss << "], [";
                index = 0;
            }
            else if (index) {
                oss << ", ";
            }
            index++;

            oss << std::setprecision(precision) << c;
        }
        oss << "]]";
        return oss.str();
    }
};

/**
 * The OpenQasm gate type.
 *
 * @param name The name of the gate to apply from the list of supported gates
 * (`rt_qasm_gate_map`)
 * @param matrix Optional matrix of complex numbers for QubitUnitary
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
    const std::vector<std::complex<double>> matrix;
    const std::vector<double> params_val;
    const std::vector<std::string> params_str;
    const std::vector<size_t> wires;
    const bool inverse;

  public:
    explicit QasmGate(const std::string &_name, const std::vector<double> &_params_val,
                      const std::vector<std::string> &_params_str,
                      const std::vector<size_t> &_wires, [[maybe_unused]] bool _inverse)
        : name(lookup_qasm_gate_name(_name)), matrix({}), params_val(_params_val),
          params_str(_params_str), wires(_wires), inverse(_inverse)
    {
        RT_FAIL_IF(!(params_str.empty() || params_val.empty()),
                   "Parametric gates are currently supported via either their values or names but "
                   "not both.");
    }
    explicit QasmGate(const std::vector<std::complex<double>> _matrix,
                      const std::vector<size_t> &_wires, [[maybe_unused]] bool _inverse)
        : name("QubitUnitary"), matrix(_matrix), params_val({}), params_str({}), wires(_wires),
          inverse(_inverse)
    {
    }
    ~QasmGate() = default;

    [[nodiscard]] auto getName() const -> std::string { return name; }
    [[nodiscard]] auto getMatrix() const -> std::vector<std::complex<double>> { return matrix; }
    [[nodiscard]] auto getParams() const -> std::vector<double> { return params_val; }
    [[nodiscard]] auto getParamsStr() const -> std::vector<std::string> { return params_str; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> { return wires; }
    [[nodiscard]] auto getInverse() const -> bool { return inverse; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister, size_t precision = 5,
                                  const std::string &version = "3.0") const -> std::string
    {
        std::ostringstream oss;
        // @note This is a Braket specific functionality
        // #pragma braket unitary(matrix) qubit_1, ..., qubit_m
        if (name == "QubitUnitary") {
            oss << "#pragma braket unitary(";
            oss << MatrixBuilder::toOpenQasm(matrix, (1UL << wires.size()), precision, version);
            oss << ") ";
            oss << qregister.toOpenQasm(RegisterMode::Slice, wires) << "\n";
            return oss.str();
        }

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
        else if (!params_str.empty()) {
            oss << "(";
            auto iter = params_str.begin();
            for (; iter != params_str.end() - 1; iter++) {
                oss << *iter << ", ";
            }
            oss << *iter << ") ";
        }
        else {
            oss << " ";
        }
        oss << qregister.toOpenQasm(RegisterMode::Slice, wires) << ";\n";
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

    [[nodiscard]] auto getBit() const -> size_t { return bit; }
    [[nodiscard]] auto getWire() const -> size_t { return wire; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister,
                                  [[maybe_unused]] const std::string &version = "3.0") const
        -> std::string
    {
        // measure wire
        std::ostringstream oss;
        oss << "measure " << qregister.toOpenQasm(RegisterMode::Slice, {wire}) << ";\n";
        return oss.str();
    }
    [[nodiscard]] auto toOpenQasm(const QasmRegister &bregister, const QasmRegister &qregister,
                                  RegisterMode mode = RegisterMode::Slice,
                                  [[maybe_unused]] const std::string &version = "3.0") const
        -> std::string
    {
        // bit = measure wire
        std::ostringstream oss;
        oss << bregister.toOpenQasm(mode, {bit}) << " = measure "
            << qregister.toOpenQasm(mode, {wire}) << ";\n";
        return oss.str();
    }
};

/**
 * A base class for all Braket/OpenQasm3 observable types.
 */
class QasmObs {
  protected:
    QasmObs() = default;
    QasmObs(const QasmObs &) = default;
    QasmObs(QasmObs &&) = default;
    QasmObs &operator=(const QasmObs &) = default;
    QasmObs &operator=(QasmObs &&) noexcept = default;

  public:
    virtual ~QasmObs() = default;
    [[nodiscard]] virtual auto getName() const -> std::string = 0;
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;
    [[nodiscard]] virtual auto toOpenQasm(const QasmRegister &qregister,
                                          [[maybe_unused]] size_t precision = 5,
                                          [[maybe_unused]] const std::string &version = "3.0") const
        -> std::string = 0;
};

/**
 * A class for Braket/OpenQasm3 named observable (PauliX, PauliY, PauliZ, Hadamard, etc.)
 */
class QasmNamedObs final : public QasmObs {
  private:
    const std::string name;
    const std::vector<size_t> wires;

  public:
    explicit QasmNamedObs(const std::string &_name, std::vector<size_t> _wires)
        : name(lookup_qasm_gate_name(_name)), wires(_wires)
    {
    }

    [[nodiscard]] auto getName() const -> std::string override { return name; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override { return wires; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister,
                                  [[maybe_unused]] size_t precision = 5,
                                  [[maybe_unused]] const std::string &version = "3.0") const
        -> std::string override
    {
        std::ostringstream oss;
        oss << name << "(" << qregister.toOpenQasm(RegisterMode::Slice, wires) << ")";
        return oss.str();
    }
};

/**
 * A class for Braket/OpenQasm3 the hermitian observable.
 *
 * Note that the support is generic so that it works for any
 * implementation of the `QasmObs` base class.
 */
class QasmHermitianObs final : public QasmObs {
  private:
    std::vector<std::complex<double>> matrix;
    std::vector<size_t> wires;
    const size_t num_cols;

  public:
    template <typename T1>
    QasmHermitianObs(T1 &&_matrix, std::vector<size_t> _wires)
        : matrix{std::forward<T1>(_matrix)}, wires{std::move(_wires)}, num_cols(1UL << wires.size())
    {
        RT_ASSERT(matrix.size() == num_cols * num_cols);
    }

    [[nodiscard]] auto getMatrix() const -> const std::vector<std::complex<double>> &
    {
        return matrix;
    }
    [[nodiscard]] auto getName() const -> std::string override { return "QasmHermitianObs"; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override { return wires; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister, size_t precision = 5,
                                  const std::string &version = "3.0") const -> std::string override
    {
        std::ostringstream oss;
        oss << "hermitian(";
        oss << MatrixBuilder::toOpenQasm(matrix, num_cols, precision, version);
        oss << ") ";
        oss << qregister.toOpenQasm(RegisterMode::Slice, wires);

        return oss.str();
    }
};

/**
 * A class for Braket/OpenQasm3 tensor product of observables type.
 *
 * Note that the support is generic so that it works for any
 * implementation of the `QasmObs` base class.
 */
class QasmTensorObs final : public QasmObs {
  private:
    std::vector<std::shared_ptr<QasmObs>> obs;
    std::vector<size_t> wires;

  public:
    template <typename... Ts> explicit QasmTensorObs(Ts &&...args) : obs{std::forward<Ts>(args)...}
    {
        std::unordered_set<size_t> all_wires;

        for (const auto &ob : obs) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                if (all_wires.contains(wire)) {
                    RT_FAIL(
                        "Invalid list of total wires; All wires in observables must be disjoint.");
                }
                all_wires.insert(wire);
            }
        }
        wires = std::vector<size_t>(all_wires.begin(), all_wires.end());
        std::sort(wires.begin(), wires.end());
    }

    [[nodiscard]] auto getName() const -> std::string override { return "QasmTensorObs"; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override { return wires; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister, size_t precision = 5,
                                  const std::string &version = "3.0") const -> std::string override
    {
        std::ostringstream oss;
        const size_t obs_size = obs.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            oss << obs[idx]->toOpenQasm(qregister, precision, version);
            if (idx != obs_size - 1) {
                oss << " @ ";
            }
        }
        return oss.str();
    }
};

/**
 * A class for Braket/OpenQasm3 Hamiltonian as a sum of observables type.
 *
 * Note that the support is generic so that it works for any
 * implementation of the `QasmObs` base class.
 */
class QasmHamiltonianObs final : public QasmObs {
  private:
    std::vector<double> coeffs;
    std::vector<std::shared_ptr<QasmObs>> obs;

  public:
    template <typename ObsVecT, typename CoeffsT>
    explicit QasmHamiltonianObs(CoeffsT &&_coeffs, ObsVecT &&_obs)
        : coeffs{std::forward<CoeffsT>(_coeffs)}, obs{std::forward<ObsVecT>(_obs)}
    {
        RT_ASSERT(obs.size() == coeffs.size());
    }

    static auto create(std::initializer_list<double> _coeffs,
                       std::initializer_list<std::shared_ptr<QasmObs>> _obs)
        -> std::shared_ptr<QasmHamiltonianObs>
    {
        return std::shared_ptr<QasmHamiltonianObs>(
            new QasmHamiltonianObs{std::move(_coeffs), std::move(_obs)});
    }

    [[nodiscard]] auto getName() const -> std::string override { return "QasmHamiltonianObs"; }
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override
    {
        std::unordered_set<size_t> all_wires;

        for (const auto &ob : obs) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                if (all_wires.contains(wire)) {
                    continue;
                }
                all_wires.insert(wire);
            }
        }
        auto wires = std::vector<size_t>(all_wires.begin(), all_wires.end());
        std::sort(wires.begin(), wires.end());
        return wires;
    }
    [[nodiscard]] auto getCoeffs() const -> std::vector<double> { return coeffs; }

    [[nodiscard]] auto toOpenQasm(const QasmRegister &qregister, size_t precision = 5,
                                  const std::string &version = "3.0") const -> std::string override
    {
        std::ostringstream oss;
        const size_t obs_size = obs.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            oss << coeffs[idx] << " * " << obs[idx]->toOpenQasm(qregister, precision, version);
            if (idx != obs_size - 1) {
                oss << " + ";
            }
        }
        return oss.str();
    }
};

/**
 * The OpenQasm circuit builder interface.
 *
 * @note Only one user-specified quantum register is currently supported.
 * @note User-specified measurement results registers are supported.
 *
 * @param qregs Quantum registers
 * @param bregs Measurement results registers
 * @param gates Quantum gates
 * @param measures Quantum measures
 */
class OpenQasmBuilder {
  protected:
    std::vector<QasmVariable> vars;
    std::vector<QasmRegister> qregs;
    std::vector<QasmRegister> bregs;
    std::vector<QasmGate> gates;
    std::vector<QasmMeasure> measures;
    size_t num_qubits;
    size_t num_bits;

  public:
    explicit OpenQasmBuilder() : num_qubits(0), num_bits(0) {}
    virtual ~OpenQasmBuilder() = default;

    [[nodiscard]] auto getNumQubits() const -> size_t { return num_qubits; }
    [[nodiscard]] auto getNumBits() const -> size_t { return num_bits; }
    [[nodiscard]] auto getQubits() const -> std::vector<QasmRegister> { return qregs; }

    void Register(RegisterType type, const std::string &name, size_t size)
    {
        switch (type) {
        case RegisterType::Qubit:
            qregs.emplace_back(type, name, size);
            num_qubits += size;
            break;
        case RegisterType::Bit:
            bregs.emplace_back(type, name, size);
            num_bits += size;
            break;
        default:
            RT_FAIL("Unsupported OpenQasm register type");
        }
    }

    void Gate(const std::string &name, const std::vector<double> &params_val,
              const std::vector<std::string> &params_str, const std::vector<size_t> &wires,
              [[maybe_unused]] bool inverse)
    {
        gates.emplace_back(name, params_val, params_str, wires, inverse);

        for (auto &param : params_str) {
            vars.emplace_back(VariableType::Float, param);
        }
    }
    void Gate(const std::vector<std::complex<double>> &matrix, const std::vector<size_t> &wires,
              [[maybe_unused]] bool inverse)
    {
        gates.emplace_back(matrix, wires, inverse);
    }
    void Measure(size_t bit, size_t wire) { measures.emplace_back(bit, wire); }

    [[nodiscard]] virtual auto toOpenQasm(size_t precision = 5,
                                          const std::string &version = "3.0") const -> std::string
    {
        RT_FAIL_IF(qregs.size() != 1, "Invalid number of quantum registers; Only one quantum "
                                      "register is currently supported.");

        RT_FAIL_IF(bregs.size() > 1,
                   "Invalid number of measurement results registers; At most one measurement"
                   "results register is currently supported.");

        std::ostringstream oss;

        // header
        oss << "OPENQASM " << version << ";\n";

        // variables
        for (auto &var : vars) {
            oss << var.toOpenQasm();
        }

        // quantum registers
        for (auto &qreg : qregs) {
            oss << qreg.toOpenQasm(RegisterMode::Alloc);
        }

        // measurement results registers
        for (auto &breg : bregs) {
            oss << breg.toOpenQasm(RegisterMode::Alloc);
        }

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : gates) {
            oss << gate.toOpenQasm(qregs[0], precision);
        }

        // quantum measures assuming qregs.size() == 1, bregs.size() <= 1
        for (auto &m : measures) {
            if (bregs.empty()) {
                oss << m.toOpenQasm(qregs[0]);
            }
            else {
                oss << m.toOpenQasm(bregs[0], qregs[0]);
            }
        }

        // reset quantum registers
        for (auto &qreg : qregs) {
            oss << qreg.toOpenQasm(RegisterMode::Reset);
        }

        return oss.str();
    }

    [[nodiscard]] virtual auto
    toOpenQasmWithCustomInstructions([[maybe_unused]] const std::string &serialized_instructions,
                                     [[maybe_unused]] size_t precision = 5,
                                     [[maybe_unused]] const std::string &version = "3.0") const
        -> std::string
    {
        RT_FAIL("Unsupported functionality");
        return std::string{};
    }
};

/**
 * The Braket OpenQasm3 circuit builder derived from OpenQasmBuilder.
 *
 * @note Braket devices currently don't support mid-circuit measurement and partial measurement
 * results.
 * @note Only one user-specified quantum register is currently supported.
 * @note User-specified measurement results registers are not currently supported.
 */
class BraketBuilder : public OpenQasmBuilder {
  public:
    using OpenQasmBuilder::OpenQasmBuilder;

    [[nodiscard]] auto toOpenQasm(size_t precision = 5, const std::string &version = "3.0") const
        -> std::string override
    {
        RT_FAIL_IF(qregs.size() != 1, "Invalid number of quantum registers; Only one quantum "
                                      "register is currently supported.");

        RT_FAIL_IF(
            !bregs.empty(),
            "Invalid number of measurement results registers; User-specified measurement results "
            "register is not currently supported.");

        std::ostringstream oss;

        // header
        oss << "OPENQASM " << version << ";\n";

        // variables
        for (auto &var : vars) {
            oss << var.toOpenQasm();
        }

        // quantum registers
        oss << qregs[0].toOpenQasm(RegisterMode::Alloc, {}, version);

        // measurement results registers
        QasmRegister braket_mresults{RegisterType::Bit, "bits", qregs[0].getSize()};
        oss << braket_mresults.toOpenQasm(RegisterMode::Alloc, {}, version);

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : gates) {
            oss << gate.toOpenQasm(qregs[0], precision, version);
        }

        // quantum measures assuming bregs[0].size() == qregs[0].size()
        // and "mresults" isn't a user-specified register.
        QasmMeasure braket_measure{0, 0};
        oss << braket_measure.toOpenQasm(braket_mresults, qregs[0], RegisterMode::Name, version);

        return oss.str();
    }

    [[nodiscard]] auto toOpenQasmWithCustomInstructions(const std::string &serialized_instructions,
                                                        size_t precision = 5,
                                                        const std::string &version = "3.0") const
        -> std::string override
    {
        RT_FAIL_IF(qregs.size() != 1, "Invalid number of quantum registers; Only one quantum "
                                      "register is currently supported.");

        RT_FAIL_IF(
            !bregs.empty(),
            "Invalid number of measurement results registers; User-specified measurement results "
            "register is not currently supported.");

        std::ostringstream oss;

        // header
        oss << "OPENQASM " << version << ";\n";

        // quantum registers
        oss << qregs[0].toOpenQasm(RegisterMode::Alloc, {}, version);

        // quantum gates assuming qregs.size() == 1
        for (auto &gate : gates) {
            oss << gate.toOpenQasm(qregs[0], precision, version);
        }

        oss << serialized_instructions;

        return oss.str();
    }
};

} // namespace Catalyst::Runtime::Device::OpenQasm
