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

#include <iostream>
#include <numeric>

#include "Rings.hpp"
#include "Utils.hpp"

namespace RSDecomp::Rings {
using namespace RSDecomp::Utils;

// --- ZSqrtTwo Method Implementations ---

ZSqrtTwo::ZSqrtTwo(INT_TYPE a, INT_TYPE b) : a(a), b(b) {}

ZSqrtTwo ZSqrtTwo::operator+(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a + other.a, b + other.b);
}

ZSqrtTwo ZSqrtTwo::operator-(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a - other.a, b - other.b);
}

ZSqrtTwo ZSqrtTwo::operator*(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a * other.a + 2 * b * other.b, a * other.b + b * other.a);
}

ZSqrtTwo ZSqrtTwo::operator*(INT_TYPE scalar) const { return ZSqrtTwo(a * scalar, b * scalar); }

ZSqrtTwo ZSqrtTwo::operator/(ZSqrtTwo other) const { return (*this * other.adj2()) / other.abs(); }

ZSqrtTwo ZSqrtTwo::operator/(INT_TYPE scalar) const
{
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero");
    }
    if (a % scalar != 0 || b % scalar != 0) {
        throw std::invalid_argument("Non-integer division result");
    }
    return ZSqrtTwo(a / scalar, b / scalar);
}

bool ZSqrtTwo::operator==(const ZSqrtTwo &other) const { return a == other.a && b == other.b; }

INT_TYPE ZSqrtTwo::abs() const { return a * a - 2 * b * b; }

ZSqrtTwo ZSqrtTwo::adj2() const { return ZSqrtTwo(a, -b); }

double ZSqrtTwo::to_double() const
{
    return static_cast<double>(a) + static_cast<double>(b) * M_SQRT2;
}

ZSqrtTwo ZSqrtTwo::pow(INT_TYPE exponent) const
{
    if (exponent < 0) {
        throw std::invalid_argument("Negative exponent not supported for ZSqrtTwo");
    }
    ZSqrtTwo result(1, 0);
    ZSqrtTwo base = *this;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = result * base;
        }
        base = base * base;
        exponent /= 2;
    }
    return result;
}

ZSqrtTwo ZSqrtTwo::operator%(const ZSqrtTwo &other) const
{
    INT_TYPE d = other.abs();
    ZSqrtTwo num = *this * other.adj2();
    INT_TYPE q1 = std::nearbyint(static_cast<double>(num.a) / d);
    INT_TYPE q2 = std::nearbyint(static_cast<double>(num.b) / d);
    ZSqrtTwo quotient(q1, q2);
    return *this - quotient * other;
}

std::optional<ZSqrtTwo> ZSqrtTwo::sqrt() const
{
    const INT_TYPE d = this->abs();
    const INT_TYPE r = std::sqrt(d);
    if (r * r != d) {
        return std::nullopt;
    }
    for (INT_TYPE s : {1, -1}) {
        INT_TYPE x_numerator = a + s * r;
        INT_TYPE y_numerator = a - s * r;
        INT_TYPE x = std::sqrt(x_numerator / 2);
        INT_TYPE y = std::sqrt(y_numerator / 4);
        ZSqrtTwo zrt{x, y};
        if (zrt * zrt == *this) {
            return zrt;
        }
        ZSqrtTwo art = zrt.adj2();
        if (art * art == *this) {
            return art;
        }
    }
    return std::nullopt;
}

ZOmega ZSqrtTwo::to_omega() const { return ZOmega(-b, 0, b, a); }

// --- ZOmega Method Implementations ---

ZOmega::ZOmega(INT_TYPE a, INT_TYPE b, INT_TYPE c, INT_TYPE d) : a(a), b(b), c(c), d(d) {}

ZOmega::ZOmega(INT_TYPE d) : a(0), b(0), c(0), d(d) {}

ZOmega ZOmega::operator-() const { return ZOmega(-a, -b, -c, -d); }

ZOmega ZOmega::operator*(const ZOmega &other) const
{
    return ZOmega(a * other.d + b * other.c + c * other.b + d * other.a,
                  b * other.d + c * other.c + d * other.b - a * other.a,
                  c * other.d + d * other.c - a * other.b - b * other.a,
                  d * other.d - a * other.c - b * other.b - c * other.a);
}

ZOmega ZOmega::operator*(INT_TYPE scalar) const
{
    return ZOmega(a * scalar, b * scalar, c * scalar, d * scalar);
}

ZOmega ZOmega::operator/(INT_TYPE scalar) const
{
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero");
    }
    if (a % scalar != 0 || b % scalar != 0 || c % scalar != 0 || d % scalar != 0) {
        throw std::invalid_argument("Non-integer division result");
    }
    return ZOmega(a / scalar, b / scalar, c / scalar, d / scalar);
}

ZOmega ZOmega::operator+(const ZOmega &other) const
{
    return ZOmega(a + other.a, b + other.b, c + other.c, d + other.d);
}

ZOmega ZOmega::operator-(const ZOmega &other) const
{
    return ZOmega(a - other.a, b - other.b, c - other.c, d - other.d);
}

bool ZOmega::operator==(const ZOmega &other) const
{
    return a == other.a && b == other.b && c == other.c && d == other.d;
}

std::complex<double> ZOmega::to_complex() const
{
    return std::complex<double>(static_cast<double>(a) * std::pow(OMEGA, 3) +
                                static_cast<double>(b) * std::pow(OMEGA, 2) +
                                static_cast<double>(c) * OMEGA + static_cast<double>(d));
}

bool ZOmega::parity() const { return (a + c) % 2 != 0; }

ZOmega ZOmega::adj2() const { return ZOmega(-a, b, -c, d); }

MULTI_PREC_INT ZOmega::abs() const
{
    MULTI_PREC_INT first = a * a + b * b + c * c + d * d;
    MULTI_PREC_INT second = a * b + b * c + c * d - d * a;
    return first * first - 2 * second * second;
}

ZOmega ZOmega::conj() const { return ZOmega(-c, -b, -a, d); }

ZOmega ZOmega::norm() const { return (*this) * (*this).conj(); }

ZOmega ZOmega::operator%(const ZOmega &other) const
{
    MULTI_PREC_INT d = other.abs();
    ZOmega_multiprec other_multiprec{other};
    ZOmega_multiprec other_conj_multiprec{other.conj()};
    ZOmega_multiprec n = ZOmega_multiprec(*this) * other_conj_multiprec *
                         ((other_multiprec * other_conj_multiprec).adj2());

    MULTI_PREC_INT na = floor_div((n.a + floor_div(d, (MULTI_PREC_INT)2)), d);
    MULTI_PREC_INT nb = floor_div((n.b + floor_div(d, (MULTI_PREC_INT)2)), d);
    MULTI_PREC_INT nc = floor_div((n.c + floor_div(d, (MULTI_PREC_INT)2)), d);
    MULTI_PREC_INT nd = floor_div((n.d + floor_div(d, (MULTI_PREC_INT)2)), d);

    ZOmega_multiprec result = ZOmega_multiprec{na, nb, nc, nd} * other - ZOmega_multiprec{*this};

    return ZOmega(result.a.convert_to<INT_TYPE>(), result.b.convert_to<INT_TYPE>(),
                  result.c.convert_to<INT_TYPE>(), result.d.convert_to<INT_TYPE>());
}

ZSqrtTwo ZOmega::to_sqrt_two() const
{
    if ((c + a == 0) && (b == 0)) {
        return ZSqrtTwo(d, (c - a) / 2);
    }
    throw std::invalid_argument("Invalid ZOmega for conversion to ZSqrtTwo");
}

std::pair<ZOmega, int> ZOmega::normalize()
{
    int ix = 0;
    ZOmega res = *this;
    while (((res.a + res.c) % 2) == 0 && ((res.b + res.d) % 2) == 0) {
        INT_TYPE a = floor_div(res.b - res.d, (INT_TYPE)2);
        INT_TYPE b = floor_div(res.a + res.c, (INT_TYPE)2);
        INT_TYPE c = floor_div(res.b + res.d, (INT_TYPE)2);
        INT_TYPE d = floor_div(res.c - res.a, (INT_TYPE)2);
        res = ZOmega(a, b, c, d);
        ix += 1;
    }
    return {res, ix};
}

// --- ZOmega_multiprec Method Implementations ---

ZOmega_multiprec::ZOmega_multiprec(MULTI_PREC_INT a, MULTI_PREC_INT b, MULTI_PREC_INT c,
                                   MULTI_PREC_INT d)
    : a(a), b(b), c(c), d(d)
{
}

ZOmega_multiprec::ZOmega_multiprec(ZOmega zomega)
    : a(zomega.a), b(zomega.b), c(zomega.c), d(zomega.d)
{
}

ZOmega_multiprec ZOmega_multiprec::operator*(const ZOmega_multiprec &other) const
{
    return ZOmega_multiprec(a * other.d + b * other.c + c * other.b + d * other.a,
                            b * other.d + c * other.c + d * other.b - a * other.a,
                            c * other.d + d * other.c - a * other.b - b * other.a,
                            d * other.d - a * other.c - b * other.b - c * other.a);
}

ZOmega_multiprec ZOmega_multiprec::adj2() const { return ZOmega_multiprec(-a, b, -c, d); }

ZOmega_multiprec ZOmega_multiprec::operator-(ZOmega_multiprec other) const
{
    return ZOmega_multiprec(a - other.a, b - other.b, c - other.c, d - other.d);
}

// --- DyadicMatrix Method Implementations ---

DyadicMatrix::DyadicMatrix(const ZOmega &a, const ZOmega &b, const ZOmega &c, const ZOmega &d,
                           INT_TYPE k)
    : a(a), b(b), c(c), d(d), k(k)
{
    normalize();
}

void DyadicMatrix::normalize()
{
    if (a == ZOmega(0) && b == ZOmega(0) && c == ZOmega(0) && d == ZOmega(0)) {
        k = 0;
        return;
    }

    auto is_all_even = [](const ZOmega &s) {
        return (s.a % 2 == 0) && (s.b % 2 == 0) && (s.c % 2 == 0) && (s.d % 2 == 0);
    };

    while (is_all_even(a) && is_all_even(b) && is_all_even(c) && is_all_even(d)) {
        a = a / 2;
        b = b / 2;
        c = c / 2;
        d = d / 2;
        k -= 2;
    }
    auto is_divisible = [](const ZOmega &s) {
        return ((s.a + s.c) % 2 == 0) && ((s.b + s.d) % 2 == 0);
    };
    auto div_by_2 = [](const ZOmega &s) {
        INT_TYPE new_a = (s.b - s.d) / 2;
        INT_TYPE new_b = (s.a + s.c) / 2;
        INT_TYPE new_c = (s.b + s.d) / 2;
        INT_TYPE new_d = (s.c - s.a) / 2;
        return ZOmega(new_a, new_b, new_c, new_d);
    };

    while ((k > 0) && is_divisible(a) && is_divisible(b) && is_divisible(c) && is_divisible(d)) {
        a = div_by_2(a);
        b = div_by_2(b);
        c = div_by_2(c);
        d = div_by_2(d);
        k -= 1;
    }
}

DyadicMatrix DyadicMatrix::operator-() const { return DyadicMatrix(-a, -b, -c, -d, k); }

bool DyadicMatrix::operator==(const DyadicMatrix &other) const
{
    return a == other.a && b == other.b && c == other.c && d == other.d && k == other.k;
}

std::array<ZOmega, 4> DyadicMatrix::flatten() const { return {a, b, c, d}; }

DyadicMatrix DyadicMatrix::operator*(const ZOmega &scalar) const
{
    return DyadicMatrix(a * scalar, b * scalar, c * scalar, d * scalar, k);
}

// --- SO3Matrix Method Implementations ---

SO3Matrix::SO3Matrix(const DyadicMatrix &dy_mat) : dyadic_mat(dy_mat)
{
    from_dyadic_matrix(dy_mat);
    normalize();
}

SO3Matrix::SO3Matrix(const std::array<std::array<ZSqrtTwo, 3>, 3> &mat, int k) : so3_mat(mat), k(k)
{
    normalize();
}

void SO3Matrix::from_dyadic_matrix(const DyadicMatrix &dy_mat)
{
    const auto &su2_elems = dy_mat.flatten();
    INT_TYPE current_k = 2 * dy_mat.k;

    bool has_parity = false;
    for (const auto &s : su2_elems) {
        if (s.parity()) {
            has_parity = true;
            break;
        }
    }

    std::array<std::pair<ZSqrtTwo, ZSqrtTwo>, 4> z_sqrt2;
    if (has_parity) {
        current_k += 2;
        for (std::size_t i = 0; i < su2_elems.size(); ++i) {
            const auto &s = su2_elems[i];
            z_sqrt2[i] = {ZSqrtTwo((s.c - s.a), s.d), ZSqrtTwo((s.c + s.a), s.b)};
        }
    }
    else {
        for (std::size_t i = 0; i < su2_elems.size(); ++i) {
            const auto &s = su2_elems[i];
            z_sqrt2[i] = {ZSqrtTwo(s.d, (s.c - s.a) / 2), ZSqrtTwo(s.b, (s.c + s.a) / 2)};
        }
    }

    const auto &a_ = z_sqrt2[0];
    const auto &b_ = z_sqrt2[1];
    const auto &c_ = z_sqrt2[2];
    const auto &d_ = z_sqrt2[3];

    so3_mat[0][0] =
        a_.first * d_.first + a_.second * d_.second + b_.first * c_.first + b_.second * c_.second;
    so3_mat[0][1] =
        a_.second * d_.first + b_.first * c_.second - b_.second * c_.first - a_.first * d_.second;
    so3_mat[0][2] =
        a_.first * c_.first + a_.second * c_.second - b_.first * d_.first - b_.second * d_.second;
    so3_mat[1][0] =
        a_.first * d_.second - a_.second * d_.first + b_.first * c_.second - b_.second * c_.first;
    so3_mat[1][1] =
        a_.first * d_.first + a_.second * d_.second - b_.first * c_.first - b_.second * c_.second;
    so3_mat[1][2] =
        a_.first * c_.second - a_.second * c_.first - b_.first * d_.second + b_.second * d_.first;
    so3_mat[2][0] = (a_.first * b_.first + a_.second * b_.second) * 2;
    so3_mat[2][1] = (a_.second * b_.first - a_.first * b_.second) * 2;
    so3_mat[2][2] =
        a_.first * a_.first + a_.second * a_.second - b_.first * b_.first - b_.second * b_.second;

    this->k = current_k;
}

bool SO3Matrix::operator==(const SO3Matrix &other) const
{
    return so3_mat == other.so3_mat && k == other.k;
}

std::array<std::array<int, 3>, 3> SO3Matrix::parity_mat() const
{
    std::array<std::array<int, 3>, 3> p_mat;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            p_mat[i][j] = (so3_mat[i][j].a % 2 + 2) % 2;
        }
    }
    return p_mat;
}

std::array<int, 3> SO3Matrix::parity_vec() const
{
    auto p_mat = this->parity_mat();
    std::array<int, 3> p_vec;
    for (std::size_t i = 0; i < 3; ++i) {
        p_vec[i] = std::accumulate(p_mat[i].begin(), p_mat[i].end(), 0);
    }
    return p_vec;
}

std::array<ZSqrtTwo, 9> SO3Matrix::flatten() const
{
    return {so3_mat[0][0], so3_mat[0][1], so3_mat[0][2], so3_mat[1][0], so3_mat[1][1],
            so3_mat[1][2], so3_mat[2][0], so3_mat[2][1], so3_mat[2][2]};
}

void SO3Matrix::normalize()
{
    auto elements = this->flatten();
    if (std::all_of(elements.begin(), elements.end(),
                    [](const ZSqrtTwo &z) { return z.a == 0 && z.b == 0; })) {
        k = 0;
        return;
    }
    while (std::all_of(elements.begin(), elements.end(),
                       [](const ZSqrtTwo &z) { return z.a % 2 == 0 && z.b % 2 == 0; })) {
        for (std::size_t i = 0; i < elements.size(); ++i) {
            elements[i] = elements[i] / 2;
        }
        k -= 2;
    }
    while (std::all_of(elements.begin(), elements.end(),
                       [](const ZSqrtTwo &z) { return z.a % 2 == 0; })) {
        for (std::size_t i = 0; i < elements.size(); ++i) {
            elements[i] = ZSqrtTwo(elements[i].b, elements[i].a / 2);
        }
        k -= 1;
    }
    so3_mat[0] = {elements[0], elements[1], elements[2]};
    so3_mat[1] = {elements[3], elements[4], elements[5]};
    so3_mat[2] = {elements[6], elements[7], elements[8]};
}

// --- Free Function Implementations ---

ZOmega zomega_from_sqrt_pair(const ZSqrtTwo &alpha, const ZSqrtTwo &beta, const ZOmega &shift)
{
    return ZOmega(beta.b - alpha.b, beta.a, beta.b + alpha.b, alpha.a) + shift;
}

DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2)
{
    return DyadicMatrix(m1.a * m2.a + m1.b * m2.c, m1.a * m2.b + m1.b * m2.d,
                        m1.c * m2.a + m1.d * m2.c, m1.c * m2.b + m1.d * m2.d, m1.k + m2.k);
}

SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2)
{
    std::array<std::array<ZSqrtTwo, 3>, 3> result_mat{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            result_mat[i][j] = ZSqrtTwo(0, 0);
            for (std::size_t k = 0; k < 3; ++k) {
                result_mat[i][j] = result_mat[i][j] + (m1.so3_mat[i][k] * m2.so3_mat[k][j]);
            }
        }
    }
    SO3Matrix result(result_mat, m1.k + m2.k);
    result.dyadic_mat = dyadic_matrix_mul(m1.dyadic_mat, m2.dyadic_mat);
    return result;
}

} // namespace RSDecomp::Rings

// Helper print functions that can be deleted
using namespace RSDecomp::Utils;
std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::SO3Matrix &matrix)
{
    os << "SO3Matrix(k=" << matrix.k << ", mat=[";
    for (const auto &row : matrix.so3_mat) {
        os << "[";
        for (const auto &elem : row) {
            os << "(" << elem.a << " + " << elem.b << "√2), ";
        }
        os << "], " << std::endl;
    }
    os << "])" << std::endl << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::ZOmega &zomega)
{
    os << "ZOmega(" << zomega.a << " ω^3 + " << zomega.b << "ω^2 + " << zomega.c << "ω + "
       << zomega.d << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const RSDecomp::Rings::ZSqrtTwo &zsqtwo)
{
    os << "ZSqrtTwo(" << zsqtwo.a << " + " << zsqtwo.b << "√2)";
    return os;
}
