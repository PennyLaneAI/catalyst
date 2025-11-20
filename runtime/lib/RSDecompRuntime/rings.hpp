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
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <complex>
#include <iostream>
#include <optional>

namespace RSDecomp::Rings {

const double LAMBDA = 1.0 + M_SQRT2;
const std::complex<double> OMEGA = std::complex<double>(M_SQRT1_2, M_SQRT1_2);

using INT_TYPE = __int128;
using MULTI_PREC_INT = boost::multiprecision::cpp_int;

struct ZOmega;
struct ZOmega_multiprec;
struct ZSqrtTwo;
struct DyadicMatrix;
struct SO3Matrix;

DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2);
SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2);
ZOmega zomega_from_sqrt_pair(const ZSqrtTwo &alpha, const ZSqrtTwo &beta, const ZOmega &shift);

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

    INT_TYPE abs() const;
    ZSqrtTwo adj2() const;
    double to_double() const;
    ZSqrtTwo pow(INT_TYPE exponent) const;
    std::optional<ZSqrtTwo> sqrt() const;

    ZOmega to_omega() const;
};

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
    MULTI_PREC_INT abs() const;
    ZOmega conj() const;
    ZOmega norm() const;

    ZSqrtTwo to_sqrt_two() const;
    std::pair<ZOmega, int> normalize();
};

// This is a helper struct for high-precision ZOmega calculations
// When we move to full arbitrary precision, this can be combined with
// ZOmega
struct ZOmega_multiprec {
    MULTI_PREC_INT a, b, c, d;

    ZOmega_multiprec(MULTI_PREC_INT a, MULTI_PREC_INT b, MULTI_PREC_INT c, MULTI_PREC_INT d);
    ZOmega_multiprec(ZOmega zomega);

    ZOmega_multiprec operator*(const ZOmega_multiprec &other) const;
    ZOmega_multiprec adj2() const;
    ZOmega_multiprec operator-(ZOmega_multiprec other) const;
};

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

struct SO3Matrix {
    DyadicMatrix dyadic_mat{ZOmega(), ZOmega(), ZOmega(), ZOmega(), 0};
    std::array<std::array<ZSqrtTwo, 3>, 3> so3_mat;
    INT_TYPE k;

    void normalize();
    void from_dyadic_matrix(const DyadicMatrix &dy_mat);

    SO3Matrix(const DyadicMatrix &dy_mat);
    SO3Matrix(const std::array<std::array<ZSqrtTwo, 3>, 3> &mat, int k = 0);

    bool operator==(const SO3Matrix &other) const;
    std::array<std::array<int, 3>, 3> parity_mat() const;
    std::array<int, 3> parity_vec() const;
    std::array<ZSqrtTwo, 9> flatten() const;
};

} // namespace RSDecomp::Rings
// Helper print functions that can be deleted
std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::ZOmega &zomega);
std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::ZSqrtTwo &zsqtwo);
std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::SO3Matrix &matrix);
// end helper functions
