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

#pragma once

#include <array>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <complex>
#include <optional>

namespace RSDecomp::Rings {

// Note on multiprecision:
// Multiprecision INTs are required to achieve down to epsilon ~ 1e-7
// Beyond this, we need to make full use of multiprecision FLOATs as well
// to avoid precision issues in intermediate calculations.
// In the current implementation, we make minimal use of multiprecision FLOATs,
// Only in the modulo calculation here. In the core Ellipse and GridSolver computation,
// we are just using doubles for now.
//
// TODO: Extend multiprecision FLOAT usage in Ellipse and GridSolver computation
// to allow even smaller epsilon values.

using INT_TYPE = boost::multiprecision::cpp_int;
using FLOAT_TYPE = boost::multiprecision::cpp_dec_float_50;

const double LAMBDA_D = 1.0 + M_SQRT2;
const std::complex<double> OMEGA = std::complex<double>(M_SQRT1_2, M_SQRT1_2);

struct ZOmega;
struct ZSqrtTwo;
struct DyadicMatrix;
struct SO3Matrix;

DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2);
SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2);
ZOmega zomega_from_sqrt_pair(const ZSqrtTwo &alpha, const ZSqrtTwo &beta, const ZOmega &shift);

/**
 * @struct ZSqrtTwo
 * @brief Represents an element of the ring Z[√2], i.e., numbers of the form a + b√2
 *        where a and b are integers.
 */
struct ZSqrtTwo {
    INT_TYPE a, b;

    ZSqrtTwo(INT_TYPE a = 0, INT_TYPE b = 0);

    ZSqrtTwo operator+(const ZSqrtTwo &other) const;
    ZSqrtTwo operator-(const ZSqrtTwo &other) const;
    ZSqrtTwo operator*(const ZSqrtTwo &other) const;
    ZSqrtTwo operator*(INT_TYPE scalar) const;
    ZSqrtTwo operator/(ZSqrtTwo other) const;
    ZSqrtTwo operator/(INT_TYPE scalar) const;
    bool operator==(const ZSqrtTwo &other) const;
    ZSqrtTwo operator%(const ZSqrtTwo &other) const;

    INT_TYPE norm() const;
    ZSqrtTwo adj2() const;
    double to_double() const;
    ZSqrtTwo pow(INT_TYPE exponent) const;
    std::optional<ZSqrtTwo> sqrt() const;

    ZOmega to_omega() const;
};

/**
 * @struct ZOmega
 * @brief Represents an element of the ring Z[ω], i.e., numbers of the form
 *        aω^3 + bω^2 + cω + d where a, b, c, d are integers and ω = e^(iπ/4).
 */
struct ZOmega {
    INT_TYPE a, b, c, d;

    ZOmega(INT_TYPE a, INT_TYPE b, INT_TYPE c, INT_TYPE d);
    ZOmega(INT_TYPE a = 0);

    ZOmega operator+(const ZOmega &other) const;
    ZOmega operator-() const;
    ZOmega operator*(const ZOmega &other) const;
    ZOmega operator*(INT_TYPE scalar) const;
    ZOmega operator/(INT_TYPE scalar) const;
    ZOmega operator-(const ZOmega &other) const;
    bool operator==(const ZOmega &other) const;
    ZOmega operator%(const ZOmega &other) const;

    std::complex<double> to_complex() const;
    bool parity() const;
    ZOmega adj2() const;
    INT_TYPE norm4() const;
    ZOmega conj() const;
    ZOmega norm2() const;

    ZSqrtTwo to_sqrt_two() const;
    std::pair<ZOmega, int> normalize();
};

/**
 * @struct DyadicMatrix
 * @brief Represents the matrices over the ring D[ω], the ring of dyadic fractions adjointed with ω.
 *
 * The dyadic fractions D = Z[1/2] are defined as D={a / 2^k | a ∈ Z, k ∈ {N U 0}}.
 *
 * ZOmega represents a subset of D[ω], and therefore can be used to construct the elements of a
 * DyadicMatrix, which is reprsented as:
 * 1/2^k * [[a, b], [c, d]]
 * where a, b, c, d ∈ D[ω] and k ∈ Z.
 *
 */
struct DyadicMatrix {
    ZOmega a, b, c, d;
    INT_TYPE k;

    void normalize();

    DyadicMatrix(const ZOmega &a, const ZOmega &b, const ZOmega &c, const ZOmega &d,
                 INT_TYPE k = 0);

    DyadicMatrix operator-() const;
    bool operator==(const DyadicMatrix &other) const;
    std::array<ZOmega, 4> flatten() const;
    DyadicMatrix operator*(const ZOmega &scalar) const;
};

/**
 * @struct SO3Matrix
 * @brief Represents the SO(3) matrices over the ring D[√2], the ring of dyadic integers adjointed
 * with √2.
 *
 * ZSqrtTwo represents a subset of this ring, and can be used to construct its elements.
 */
struct SO3Matrix {
    DyadicMatrix dyadic_mat{ZOmega(), ZOmega(), ZOmega(), ZOmega(), 0};
    std::array<std::array<ZSqrtTwo, 3>, 3> so3_mat;
    INT_TYPE k;

    void normalize();
    void from_dyadic_matrix(const DyadicMatrix &dy_mat);

    SO3Matrix(const DyadicMatrix &dy_mat);
    SO3Matrix(const std::array<std::array<ZSqrtTwo, 3>, 3> &mat, INT_TYPE k = 0);

    bool operator==(const SO3Matrix &other) const;
    std::array<std::array<int, 3>, 3> parity_mat() const;
    std::array<int, 3> parity_vec() const;
    std::array<ZSqrtTwo, 9> flatten() const;
};

} // namespace RSDecomp::Rings
