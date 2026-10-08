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

#include <complex>
#include <cstdio>
#include <map>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/generators/catch_generators_range.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

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
                                                    const std::vector<std::complex<double>> &B) {
    std::vector<std::complex<double>> result(4, 0.0);
    result[0] = A[0] * B[0] + A[1] * B[2];
    result[1] = A[0] * B[1] + A[1] * B[3];
    result[2] = A[2] * B[0] + A[3] * B[2];
    result[3] = A[2] * B[1] + A[3] * B[3];
    return result;
}

std::vector<std::complex<double>>
matrix_from_decomp_result(const std::vector<GateType> &decomposition) {
    std::vector<std::complex<double>> result = gate_type_to_matrix.at(GateType::I);
    for (const auto &gate : decomposition) {
        result = multiply_matrices(gate_type_to_matrix.at(gate), result);
    }
    return result;
}

TEST_CASE("Test Matrix Multiplication", "[RSDecomp][Ross Selinger]") {
    auto res_HT =
        multiply_matrices(gate_type_to_matrix[GateType::T], gate_type_to_matrix[GateType::H]);
    auto expected_HT = gate_type_to_matrix[GateType::HT];

    for (size_t i = 0; i < res_HT.size(); i++) {
        CHECK_THAT(res_HT[i].real(), WithinRel(expected_HT[i].real()));
        CHECK_THAT(res_HT[i].imag(), WithinRel(expected_HT[i].imag()));
    }

    auto res_SHT = multiply_matrices(res_HT, gate_type_to_matrix[GateType::S]);
    auto expected_SHT = gate_type_to_matrix[GateType::SHT];
    for (size_t i = 0; i < res_SHT.size(); i++) {
        CHECK_THAT(res_SHT[i].real(), WithinRel(expected_SHT[i].real()));
        CHECK_THAT(res_SHT[i].imag(), WithinRel(expected_SHT[i].imag()));
    }
}

TEST_CASE("Test matrix_from_decomp_result", "[RSDecomp][Ross Selinger]") {
    std::vector<GateType> decomp = {GateType::H, GateType::T};

    auto result_matrix = matrix_from_decomp_result(decomp);
    auto expected_matrix = gate_type_to_matrix[GateType::HT];

    for (size_t i = 0; i < result_matrix.size(); i++) {
        CHECK_THAT(result_matrix[i].real(), WithinRel(expected_matrix[i].real()));
        CHECK_THAT(result_matrix[i].imag(), WithinRel(expected_matrix[i].imag()));
    }

    decomp = {GateType::S, GateType::H, GateType::T};

    result_matrix = matrix_from_decomp_result(decomp);
    expected_matrix = gate_type_to_matrix[GateType::SHT];

    for (size_t i = 0; i < result_matrix.size(); i++) {
        CHECK_THAT(result_matrix[i].real(), WithinRel(expected_matrix[i].real()));
        CHECK_THAT(result_matrix[i].imag(), WithinRel(expected_matrix[i].imag()));
    }
}

TEST_CASE("Test ross_selinger generic angles", "[RSDecomp][Ross Selinger]") {
    double tolerance = GENERATE(1e-2, 1e-3, 1e-4, 1e-5, 1e-6);
    int angle_int = GENERATE(range(-70, 71));
    double angle = angle_int / 10.0;
    CAPTURE(angle);

    const auto [gates_vector, phase] = eval_ross_algorithm(angle, tolerance);

    std::vector<std::complex<double>> result_matrix;
    result_matrix = matrix_from_decomp_result(gates_vector);

    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    std::vector<std::complex<double>> global_phase_matrix = {phase_factor, 0.0, 0.0, phase_factor};
    result_matrix = multiply_matrices(global_phase_matrix, result_matrix);

    std::complex<double> z = {std::cos(angle / 2.0), -std::sin(angle / 2.0)};
    double residue_norm = std::norm(result_matrix[0] - z) + std::norm(result_matrix[2]);
    residue_norm = std::sqrt(residue_norm);
    CHECK(residue_norm <= tolerance);
}

TEST_CASE("Test ross_selinger pi/16 multiples", "[RSDecomp][Ross Selinger]") {
    double tolerance = GENERATE(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7);
    int angle_int = GENERATE(range(-70, 71));
    double angle = angle_int * M_PI / 16.0;
    CAPTURE(angle);

    const auto [gates_vector, phase] = eval_ross_algorithm(angle, tolerance);

    std::vector<std::complex<double>> result_matrix;
    result_matrix = matrix_from_decomp_result(gates_vector);

    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    std::vector<std::complex<double>> global_phase_matrix = {phase_factor, 0.0, 0.0, phase_factor};
    result_matrix = multiply_matrices(global_phase_matrix, result_matrix);

    std::complex<double> z = {std::cos(angle / 2.0), -std::sin(angle / 2.0)};
    double residue_norm = std::norm(result_matrix[0] - z) + std::norm(result_matrix[2]);
    residue_norm = std::sqrt(residue_norm);
    CHECK(residue_norm <= tolerance);
}

TEST_CASE("Test Zero Angle (Identity)", "[RSDecomp][Ross Selinger]") {
    // An angle of 0.0 should result in Identity
    const auto [gates, phase] = eval_ross_algorithm(0.0, 1e-10);

    // Should effectively be Identity
    std::vector<std::complex<double>> mat = matrix_from_decomp_result(gates);

    // Apply global phase
    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    std::vector<std::complex<double>> global_phase_matrix = {phase_factor, 0.0, 0.0, phase_factor};
    mat = multiply_matrices(global_phase_matrix, mat);

    // Target is Identity
    std::complex<double> one(1.0, 0.0);
    std::complex<double> zero(0.0, 0.0);

    CHECK(std::abs(mat[0] - one) < 1e-9);
    CHECK(std::abs(mat[3] - one) < 1e-9);
    CHECK(std::abs(mat[1] - zero) < 1e-9);
    CHECK(std::abs(mat[2] - zero) < 1e-9);
}

TEST_CASE("Test HST_to_PPR Conversion Rules", "[RSDecomp][Ross Selinger]") {
    // Rule: HT, HT -> X8, Z8
    CHECK(HST_to_PPR({GateType::HT, GateType::HT}).first ==
          std::vector<PPRGateType>{PPRGateType::X8, PPRGateType::Z8});
    CHECK(HST_to_PPR({GateType::HT, GateType::HT}).second == -M_PI / 4.0);

    // Rule: HT, SHT -> X4, X8, Z8
    CHECK(HST_to_PPR({GateType::HT, GateType::SHT}).first ==
          std::vector<PPRGateType>{PPRGateType::X4, PPRGateType::X8, PPRGateType::Z8});
    CHECK(HST_to_PPR({GateType::HT, GateType::SHT}).second == -M_PI / 2.0);

    // Rule: SHT, HT -> Z4, X8, Z8
    CHECK(HST_to_PPR({GateType::SHT, GateType::HT}).first ==
          std::vector<PPRGateType>{PPRGateType::Z4, PPRGateType::X8, PPRGateType::Z8});
    CHECK(HST_to_PPR({GateType::SHT, GateType::HT}).second == -M_PI / 2.0);

    // Rule: SHT, SHT -> Z4, X4, X8, Z8
    CHECK(HST_to_PPR({GateType::SHT, GateType::SHT}).first ==
          std::vector<PPRGateType>{PPRGateType::Z4, PPRGateType::X4, PPRGateType::X8,
                                   PPRGateType::Z8});
    CHECK(HST_to_PPR({GateType::SHT, GateType::SHT}).second == -3 * M_PI / 4.0);

    // Test Single Gate Mappings (Standard)
    CHECK(HST_to_PPR({GateType::T}).first == std::vector<PPRGateType>{PPRGateType::Z8});
    CHECK(HST_to_PPR({GateType::S}).first == std::vector<PPRGateType>{PPRGateType::Z4});
    CHECK(HST_to_PPR({GateType::Z}).first == std::vector<PPRGateType>{PPRGateType::Z2});

    CHECK(HST_to_PPR({GateType::T}).second == -M_PI / 8.0);
    CHECK(HST_to_PPR({GateType::S}).second == -M_PI / 4.0);
    CHECK(HST_to_PPR({GateType::Z}).second == -M_PI / 2.0);

    // H -> Z4, X4, Z4
    CHECK(HST_to_PPR({GateType::H}).first ==
          std::vector<PPRGateType>{PPRGateType::Z4, PPRGateType::X4, PPRGateType::Z4});
    CHECK(HST_to_PPR({GateType::H}).second == -M_PI / 2.0);

    // Test Edge Cases for Pair Lookahead
    // Case: HT at the very end of the vector (no next gate to pair with)
    // Should fallback to single HT expansion: X8, Z4, X4, Z4
    CHECK(HST_to_PPR({GateType::HT}).first ==
          std::vector<PPRGateType>{PPRGateType::X8, PPRGateType::Z4, PPRGateType::X4,
                                   PPRGateType::Z4});
    CHECK(HST_to_PPR({GateType::HT}).second == -5 * M_PI / 8.0);

    // Case: HT followed by a gate that doesn't form a pair (e.g. T)
    std::vector<GateType> input_mixed = {GateType::HT, GateType::T};
    std::vector<PPRGateType> expected_mixed = {
        PPRGateType::X8, PPRGateType::Z4, PPRGateType::X4, PPRGateType::Z4, // HT
        PPRGateType::Z8                                                     // T
    };
    CHECK(HST_to_PPR(input_mixed).first == expected_mixed);
    CHECK(HST_to_PPR(input_mixed).second == -3 * M_PI / 4.0);
}

TEST_CASE("Test C-API Wrapper (Memref Interface)", "[RSDecomp][Ross Selinger]") {
    double angle = M_PI / 4.0; // Decomposes to exactly T
    double epsilon = 1e-5;

    // Test Clifford+T Basis API
    size_t size_std = rs_decomposition_get_size(angle, epsilon, false);
    REQUIRE(size_std > 0);

    std::vector<size_t> buffer_std(size_std);

    // Simulate MemRef call
    rs_decomposition_get_gates(nullptr, buffer_std.data(), 0, size_std, 1, angle, epsilon, false);

    CHECK(buffer_std.size() == 1);
    CHECK(static_cast<GateType>(buffer_std[0]) == GateType::T);

    // Test PPR Basis API
    size_t size_ppr = rs_decomposition_get_size(angle, epsilon, true);
    REQUIRE(size_ppr > 0);

    std::vector<size_t> buffer_ppr(size_ppr);
    rs_decomposition_get_gates(nullptr, buffer_ppr.data(), 0, size_ppr, 1, angle, epsilon, true);

    CHECK(buffer_ppr.size() == 1);
    CHECK(static_cast<PPRGateType>(buffer_ppr[0]) == PPRGateType::Z8);
}

// Helpers for the mixed decomposition tests. Matrices are row-major 2x2.
using Matrix2 = std::vector<std::complex<double>>;

Matrix2 dagger(const Matrix2 &A) {
    return {std::conj(A[0]), std::conj(A[2]), std::conj(A[1]), std::conj(A[3])};
}

Matrix2 sequence_unitary(const std::vector<GateType> &gates, double phase) {
    Matrix2 U = matrix_from_decomp_result(gates);
    std::complex<double> phase_factor = {std::cos(phase), -std::sin(phase)};
    for (auto &entry : U) {
        entry *= phase_factor;
    }
    return U;
}

size_t t_count(const std::vector<GateType> &gates) {
    size_t count = 0;
    for (GateType gate : gates) {
        count += gate == GateType::T || gate == GateType::HT || gate == GateType::SHT;
    }
    return count;
}

/**
 * Pauli transfer matrix R_ij = Tr(P_i E(P_j)) / 2 of the error channel
 * E(rho) = V^dagger M(rho) V, where M is the {Z, S}-twirled mixture and V = RZ(angle).
 */
std::vector<std::vector<double>> mixed_error_ptm(const MixedDecomposition &mixed, double angle) {
    const std::vector<Matrix2> paulis = {{1.0, 0.0, 0.0, 1.0},
                                         {0.0, 1.0, 1.0, 0.0},
                                         {0.0, {0.0, -1.0}, {0.0, 1.0}, 0.0},
                                         {1.0, 0.0, 0.0, -1.0}};
    Matrix2 V = {std::polar(1.0, -angle / 2.0), 0.0, 0.0, std::polar(1.0, angle / 2.0)};

    std::vector<std::pair<double, Matrix2>> kraus;
    for (size_t branch = 0; branch < 2; branch++) {
        double weight = branch == 0 ? mixed.probability : 1.0 - mixed.probability;
        for (size_t twirl = 0; twirl < 4; twirl++) {
            Matrix2 U =
                sequence_unitary(twirl_sequence(mixed.gates[branch], twirl), mixed.phases[branch]);
            kraus.emplace_back(weight / 4.0, multiply_matrices(dagger(V), U));
        }
    }

    std::vector<std::vector<double>> ptm(4, std::vector<double>(4, 0.0));
    for (size_t j = 0; j < 4; j++) {
        Matrix2 image(4, 0.0);
        for (const auto &[weight, K] : kraus) {
            Matrix2 term = multiply_matrices(multiply_matrices(K, paulis[j]), dagger(K));
            for (size_t e = 0; e < 4; e++) {
                image[e] += weight * term[e];
            }
        }
        for (size_t i = 0; i < 4; i++) {
            Matrix2 product = multiply_matrices(paulis[i], image);
            ptm[i][j] = 0.5 * (product[0] + product[3]).real();
        }
    }
    return ptm;
}

TEST_CASE("Test mixed diagonal decomposition accuracy", "[RSDecomp][Mixed]") {
    double epsilon = GENERATE(1e-2, 1e-4, 1e-6, 1e-8, 1e-10);
    int angle_int = GENERATE(range(-70, 71, 7));
    double angle = angle_int / 10.0 + 0.01;
    CAPTURE(angle, epsilon);

    MixedDecomposition mixed = compute_mixed_diagonal_decomposition(angle, epsilon);
    REQUIRE(mixed.probability >= 0.0);
    REQUIRE(mixed.probability <= 1.0);

    // The first branch under-rotates and the second over-rotates the target.
    std::complex<double> inverse_target = std::polar(1.0, angle / 2.0);
    std::complex<double> w_under =
        sequence_unitary(mixed.gates[0], mixed.phases[0])[0] * inverse_target;
    std::complex<double> w_over =
        sequence_unitary(mixed.gates[1], mixed.phases[1])[0] * inverse_target;
    if (mixed.probability < 1.0) {
        CHECK(w_under.imag() <= 1e-12);
        CHECK(w_over.imag() >= -1e-12);
    }

    // The error channel is a Pauli channel, so its diamond distance to the identity
    // channel is 2 (1 - p_I), with p_I = (1 + R_XX + R_YY + R_ZZ) / 4.
    auto ptm = mixed_error_ptm(mixed, angle);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            if (i != j) {
                CHECK(std::abs(ptm[i][j]) <= 1e-9);
            }
        }
    }
    double identity_weight = (ptm[0][0] + ptm[1][1] + ptm[2][2] + ptm[3][3]) / 4.0;
    double diamond_distance = 2.0 * (1.0 - identity_weight);
    CHECK(diamond_distance <= epsilon * (1.0 + 1e-6) + 1e-12);
}

TEST_CASE("Test mixed diagonal decomposition halves the T-count", "[RSDecomp][Mixed]") {
    // Compare at matched diamond-norm accuracy: a unitary with operator-norm error epsilon/2 has
    // diamond-norm error at most epsilon.
    const double epsilon = 1e-6;
    double mixed_t = 0.0;
    double deterministic_t = 0.0;
    for (int angle_int = 1; angle_int < 60; angle_int++) {
        double angle = angle_int / 10.0;
        MixedDecomposition mixed = compute_mixed_diagonal_decomposition(angle, epsilon);
        mixed_t += mixed.probability * t_count(mixed.gates[0]) +
                   (1.0 - mixed.probability) * t_count(mixed.gates[1]);
        deterministic_t += t_count(eval_ross_algorithm(angle, epsilon / 2.0).first);
    }
    CAPTURE(mixed_t, deterministic_t);
    CHECK(mixed_t < 0.6 * deterministic_t);
}

TEST_CASE("Test mixed diagonal decomposition of exact angles", "[RSDecomp][Mixed]") {
    MixedDecomposition mixed = compute_mixed_diagonal_decomposition(M_PI / 4.0, 1e-4);
    CHECK(mixed.probability == 1.0);
    CHECK(mixed.gates[0] == std::vector<GateType>{GateType::T});
}

TEST_CASE("Test mixed decomposition sampling", "[RSDecomp][Mixed]") {
    CHECK(sample_mixed_decomposition(0.25, 0.0).branch == 0);
    CHECK(sample_mixed_decomposition(0.25, 0.0).twirl == 0);
    CHECK(sample_mixed_decomposition(0.25, 0.24).twirl == 3);
    CHECK(sample_mixed_decomposition(0.25, 0.25).branch == 1);
    CHECK(sample_mixed_decomposition(0.25, 0.25).twirl == 0);
    CHECK(sample_mixed_decomposition(0.25, 0.99).twirl == 3);
    CHECK(sample_mixed_decomposition(1.0, 0.99).branch == 0);
    CHECK(sample_mixed_decomposition(0.25, 1.0).branch == 1);

    std::vector<GateType> gates = {GateType::HT};
    CHECK(twirl_sequence(gates, 0) == gates);
    CHECK(twirl_sequence(gates, 1) ==
          std::vector<GateType>{GateType::Sd, GateType::HT, GateType::S});
    CHECK(twirl_sequence(gates, 2) ==
          std::vector<GateType>{GateType::Z, GateType::HT, GateType::Z});
    CHECK(twirl_sequence(gates, 3) ==
          std::vector<GateType>{GateType::S, GateType::HT, GateType::Sd});
}

TEST_CASE("Test mixed C-API Wrapper (Memref Interface)", "[RSDecomp][Mixed]") {
    const double angle = 0.7;
    const double epsilon = 1e-6;
    bool ppr_basis = GENERATE(false, true);
    double uniform_sample = GENERATE(0.1, 0.6, 0.95);
    CAPTURE(ppr_basis, uniform_sample);

    size_t size = rs_mixed_decomposition_get_size(angle, epsilon, ppr_basis, uniform_sample);
    std::vector<size_t> buffer(size);
    rs_mixed_decomposition_get_gates(nullptr, buffer.data(), 0, size, 1, angle, epsilon, ppr_basis,
                                     uniform_sample);
    double phase = rs_mixed_decomposition_get_phase(angle, epsilon, ppr_basis, uniform_sample);

    if (ppr_basis) {
        auto [gates, expected_phase] =
            eval_mixed_ross_algorithm_ppr(angle, epsilon, uniform_sample);
        REQUIRE(buffer.size() == gates.size());
        for (size_t i = 0; i < gates.size(); i++) {
            CHECK(static_cast<PPRGateType>(buffer[i]) == gates[i]);
        }
        CHECK(phase == expected_phase);
    } else {
        MixedDecomposition mixed = compute_mixed_diagonal_decomposition(angle, epsilon);
        auto [branch, twirl] = sample_mixed_decomposition(mixed.probability, uniform_sample);
        std::vector<GateType> expected = twirl_sequence(mixed.gates[branch], twirl);
        REQUIRE(buffer.size() == expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            CHECK(static_cast<GateType>(buffer[i]) == expected[i]);
        }
        CHECK(phase == mixed.phases[branch]);
    }
}

TEST_CASE("rs_decomposition_get_size emits warning for epsilon < 1e-6", "[RSDecomp][Warning]") {
    const double theta = 0.5;

    SECTION("warning is emitted when epsilon < 1e-6") {
        std::ostringstream buf;
        std::streambuf *old = std::cerr.rdbuf(buf.rdbuf());
        (void)rs_decomposition_get_size(theta, 1e-8, false);
        std::cerr.rdbuf(old);

        CHECK_THAT(buf.str(), ContainsSubstring("Gridsynth received epsilon="));
        CHECK_THAT(buf.str(), ContainsSubstring("For epsilon smaller than 1e-6"));
    }

    SECTION("no warning when epsilon >= 1e-6") {
        std::ostringstream buf;
        std::streambuf *old = std::cerr.rdbuf(buf.rdbuf());
        (void)rs_decomposition_get_size(theta, 1e-4, false);
        std::cerr.rdbuf(old);

        CHECK(buf.str().empty());
    }

    SECTION("no warning at boundary epsilon == 1e-6") {
        std::ostringstream buf;
        std::streambuf *old = std::cerr.rdbuf(buf.rdbuf());
        (void)rs_decomposition_get_size(theta, 1e-6, false);
        std::cerr.rdbuf(old);

        CHECK(buf.str().empty());
    }
}
