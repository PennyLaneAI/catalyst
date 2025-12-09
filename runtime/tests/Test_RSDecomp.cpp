// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <map>

#include "CliffordData.hpp"
#include "RSDecomp.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::RossSelinger;
using namespace RSDecomp::CliffordData;

// Helper function and matrices to verify decomposition result
std::map<GateType, std::vector<std::complex<double>>> gate_type_to_matrix = {
    {GateType::I, {1.0, 0.0, 0.0, 1.0}},
    {GateType::T, {1.0, 0.0, 0.0, {M_SQRT1_2, M_SQRT1_2}}},
    {GateType::H, {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2}},
    {GateType::S, {1.0, 0.0, 0.0, {0.0, 1.0}}},
    {GateType::X, {0.0, 1.0, 1.0, 0.0}},
    {GateType::Y, {0.0, {0.0, -1.0}, {0.0, 1.0}, 0.0}},
    {GateType::Z, {1.0, 0.0, 0.0, -1.0}},
    {GateType::Sd, {1.0, 0.0, 0.0, {0.0, -1.0}}},
    {GateType::HT, {M_SQRT1_2, M_SQRT1_2, {0.5, 0.5}, {-0.5, -0.5}}},        // T@H
    {GateType::SHT, {M_SQRT1_2, {0.0, M_SQRT1_2}, {0.5, 0.5}, {0.5, -0.5}}}, // T@H@S
};

std::vector<std::complex<double>> multiply_matrices(const std::vector<std::complex<double>> &A,
                                                    const std::vector<std::complex<double>> &B)
{
    std::vector<std::complex<double>> result(4, 0.0);
    result[0] = A[0] * B[0] + A[1] * B[2];
    result[1] = A[0] * B[1] + A[1] * B[3];
    result[2] = A[2] * B[0] + A[3] * B[2];
    result[3] = A[2] * B[1] + A[3] * B[3];
    return result;
}

std::vector<std::complex<double>>
matrix_from_decomp_result(const std::vector<GateType> &decomposition)
{
    std::vector<std::complex<double>> result = gate_type_to_matrix.at(GateType::I);
    for (const auto &gate : decomposition) {
        result = multiply_matrices(gate_type_to_matrix.at(gate), result);
    }
    return result;
}

TEST_CASE("Test Matrix Multiplication", "[RSDecomp][Ross Selinger]")
{
    auto res_HT =
        multiply_matrices(gate_type_to_matrix[GateType::T], gate_type_to_matrix[GateType::H]);
    auto expected_HT = gate_type_to_matrix[GateType::HT];

    for (size_t i = 0; i < res_HT.size(); ++i) {
        CHECK(res_HT[i].real() == Catch::Approx(expected_HT[i].real()));
        CHECK(res_HT[i].imag() == Catch::Approx(expected_HT[i].imag()));
    }

    auto res_SHT = multiply_matrices(res_HT, gate_type_to_matrix[GateType::S]);
    auto expected_SHT = gate_type_to_matrix[GateType::SHT];
    for (size_t i = 0; i < res_SHT.size(); ++i) {
        CHECK(res_SHT[i].real() == Catch::Approx(expected_SHT[i].real()));
        CHECK(res_SHT[i].imag() == Catch::Approx(expected_SHT[i].imag()));
    }
}

TEST_CASE("Test matrix_from_decomp_result", "[RSDecomp][Ross Selinger]")
{
    std::vector<GateType> decomp = {GateType::H, GateType::T};

    auto result_matrix = matrix_from_decomp_result(decomp);
    auto expected_matrix = gate_type_to_matrix[GateType::HT];

    for (size_t i = 0; i < result_matrix.size(); ++i) {
        CHECK(result_matrix[i].real() == Catch::Approx(expected_matrix[i].real()));
        CHECK(result_matrix[i].imag() == Catch::Approx(expected_matrix[i].imag()));
    }

    decomp = {GateType::S, GateType::H, GateType::T};

    result_matrix = matrix_from_decomp_result(decomp);
    expected_matrix = gate_type_to_matrix[GateType::SHT];

    for (size_t i = 0; i < result_matrix.size(); ++i) {
        CHECK(result_matrix[i].real() == Catch::Approx(expected_matrix[i].real()));
        CHECK(result_matrix[i].imag() == Catch::Approx(expected_matrix[i].imag()));
    }
}

TEST_CASE("Test ross_selinger generic angles", "[RSDecomp][Ross Selinger]")
{
    double tolerance = GENERATE(1e-2, 1e-3, 1e-4, 1e-5, 1e-6);
    int angle_int = GENERATE(range(-70, 71));
    double angle = angle_int / 10.0;
    CAPTURE(angle);
    auto decomp_result = eval_ross_algorithm(angle, tolerance);
    const auto &gates_vector = decomp_result.first;
    std::vector<std::complex<double>> result_matrix;
    result_matrix = matrix_from_decomp_result(gates_vector);

    double phase = decomp_result.second;
    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    std::vector<std::complex<double>> global_phase_matrix = {phase_factor, 0.0, 0.0, phase_factor};
    result_matrix = multiply_matrices(global_phase_matrix, result_matrix);

    std::complex<double> z = {std::cos(angle / 2.0), -std::sin(angle / 2.0)};
    double residue_norm = std::norm(result_matrix[0] - z) + std::norm(result_matrix[2]);
    residue_norm = std::sqrt(residue_norm);
    CHECK(residue_norm <= tolerance);
}

TEST_CASE("Test ross_selinger pi/16 multiples", "[RSDecomp][Ross Selinger]")
{
    double tolerance = GENERATE(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7);
    int angle_int = GENERATE(range(-70, 71));
    double angle = angle_int * M_PI / 16.0;
    CAPTURE(angle);
    auto decomp_result = eval_ross_algorithm(angle, tolerance);
    const auto &gates_vector = decomp_result.first;
    std::vector<std::complex<double>> result_matrix;
    result_matrix = matrix_from_decomp_result(gates_vector);

    double phase = decomp_result.second;
    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    std::vector<std::complex<double>> global_phase_matrix = {phase_factor, 0.0, 0.0, phase_factor};
    result_matrix = multiply_matrices(global_phase_matrix, result_matrix);

    std::complex<double> z = {std::cos(angle / 2.0), -std::sin(angle / 2.0)};
    double residue_norm = std::norm(result_matrix[0] - z) + std::norm(result_matrix[2]);
    residue_norm = std::sqrt(residue_norm);
    CHECK(residue_norm <= tolerance);
}
