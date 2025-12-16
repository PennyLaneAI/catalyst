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

#include <numeric>

#include "RSUtils.hpp"
#include "Rings.hpp"

namespace RSDecomp::Rings {
using namespace RSDecomp::Utils;

// Note:
// Definitions for most of the ring operations here are available in
// https://arxiv.org/pdf/1212.6253 secion 3
// and for parity calculations in
// https://arxiv.org/pdf/1312.6584

// --- ZSqrtTwo Method Implementations ---

/**
 * @brief Constructor for ZSqrtTwo: a + b * sqrt(2)
 */
ZSqrtTwo::ZSqrtTwo(INT_TYPE a, INT_TYPE b) : a(a), b(b) {}

/**
 * @brief Equality operator for ZSqrtTwo elements.
 */
bool ZSqrtTwo::operator==(const ZSqrtTwo &other) const { return a == other.a && b == other.b; }

/**
 * @brief Addition operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator+(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a + other.a, b + other.b);
}

/**
 * @brief Subtraction operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator-(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a - other.a, b - other.b);
}

/**
 * @brief Multiplication operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator*(const ZSqrtTwo &other) const
{
    return ZSqrtTwo(a * other.a + 2 * b * other.b, a * other.b + b * other.a);
}

/**
 * @brief Scalar multiplication operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator*(INT_TYPE scalar) const { return ZSqrtTwo(a * scalar, b * scalar); }

/**
 * @brief Division operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator/(ZSqrtTwo other) const { return (*this * other.adj2()) / other.norm(); }

/**
 * @brief Division by scalar operator for ZSqrtTwo elements.
 */
ZSqrtTwo ZSqrtTwo::operator/(INT_TYPE scalar) const
{
    RT_FAIL_IF(scalar == 0, "Division by zero");
    RT_FAIL_IF(a % scalar != 0 || b % scalar != 0, "Non-integer division result");
    return ZSqrtTwo(a / scalar, b / scalar);
}

/**
 * @brief Computes the norm of the ZSqrtTwo element. (Definition 4, arXiv:1212.6253)
 */
INT_TYPE ZSqrtTwo::norm() const { return a * a - 2 * b * b; }

/**
 * @brief Computes the adjoint of the ZSqrtTwo element.
 */
ZSqrtTwo ZSqrtTwo::adj2() const { return ZSqrtTwo(a, -b); }

/**
 * @brief Converts the ZSqrtTwo element to a double.
 */
double ZSqrtTwo::to_double() const
{
    return static_cast<double>(a) + static_cast<double>(b) * M_SQRT2;
}

/**
 * @brief Computes the exponentiation of the ZSqrtTwo element.
 */
ZSqrtTwo ZSqrtTwo::pow(INT_TYPE exponent) const
{
    RT_FAIL_IF(exponent < 0, "Negative exponent not supported for ZSqrtTwo");
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

/**
 * @brief Computes the modulo of the ZSqrtTwo element. (Remark 6, arXiv:1212.6253)
 */
ZSqrtTwo ZSqrtTwo::operator%(const ZSqrtTwo &other) const
{
    INT_TYPE d = other.norm();
    ZSqrtTwo num = *this * other.adj2();
    FLOAT_TYPE q1_float = FLOAT_TYPE(num.a) / FLOAT_TYPE(d);
    FLOAT_TYPE q2_float = FLOAT_TYPE(num.b) / FLOAT_TYPE(d);
    INT_TYPE q1 = boost::multiprecision::round(q1_float).convert_to<INT_TYPE>();
    INT_TYPE q2 = boost::multiprecision::round(q2_float).convert_to<INT_TYPE>();
    ZSqrtTwo quotient(q1, q2);
    return *this - quotient * other;
}

/**
 * @brief Computes the square root of the ZSqrtTwo element, if it exists.
 */
std::optional<ZSqrtTwo> ZSqrtTwo::sqrt() const
{
    const INT_TYPE d = this->norm();
    INT_TYPE abs_d = d < 0 ? -d : d;
    const INT_TYPE r = boost::multiprecision::sqrt(abs_d);
    if (r * r != d) {
        return std::nullopt;
    }
    for (int s : {1, -1}) {
        INT_TYPE x_numerator = a + s * r;
        INT_TYPE y_numerator = a - s * r;
        if (x_numerator < 0 || y_numerator < 0)
            continue;
        INT_TYPE x_div = x_numerator / 2;
        INT_TYPE y_div = y_numerator / 4;
        INT_TYPE x = boost::multiprecision::sqrt(x_div);
        INT_TYPE y = boost::multiprecision::sqrt(y_div);
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

/**
 * @brief Converts the ZSqrtTwo element to a ZOmega element.
 */
ZOmega ZSqrtTwo::to_omega() const { return ZOmega(-b, 0, b, a); }

// --- ZOmega Method Implementations ---

/**
 * @brief Constructor for ZOmega: a*ω^3 + b*ω^2 + c*ω + d
 */
ZOmega::ZOmega(INT_TYPE a, INT_TYPE b, INT_TYPE c, INT_TYPE d) : a(a), b(b), c(c), d(d) {}

/**
 * @brief Constructor for ZOmega with only d specified.
 */
ZOmega::ZOmega(INT_TYPE d) : a(0), b(0), c(0), d(d) {}

/**
 * @brief Equality operator for ZOmega elements.
 */
bool ZOmega::operator==(const ZOmega &other) const
{
    return a == other.a && b == other.b && c == other.c && d == other.d;
}

/**
 * @brief Addition operator for ZOmega elements.
 */
ZOmega ZOmega::operator+(const ZOmega &other) const
{
    return ZOmega(a + other.a, b + other.b, c + other.c, d + other.d);
}

/**
 * @brief Subtraction operator for ZOmega elements.
 */
ZOmega ZOmega::operator-(const ZOmega &other) const
{
    return ZOmega(a - other.a, b - other.b, c - other.c, d - other.d);
}

/**
 * @brief Negation operator for ZOmega elements.
 */
ZOmega ZOmega::operator-() const { return ZOmega(-a, -b, -c, -d); }

/**
 * @brief Multiplication operator for ZOmega elements.
 */
ZOmega ZOmega::operator*(const ZOmega &other) const
{
    return ZOmega(a * other.d + b * other.c + c * other.b + d * other.a,
                  b * other.d + c * other.c + d * other.b - a * other.a,
                  c * other.d + d * other.c - a * other.b - b * other.a,
                  d * other.d - a * other.c - b * other.b - c * other.a);
}

/**
 * @brief Scalar multiplication operator for ZOmega elements.
 */
ZOmega ZOmega::operator*(INT_TYPE scalar) const
{
    return ZOmega(a * scalar, b * scalar, c * scalar, d * scalar);
}

/**
 * @brief Division by scalar operator for ZOmega elements.
 */
ZOmega ZOmega::operator/(INT_TYPE scalar) const
{
    RT_FAIL_IF(scalar == 0, "Division by zero");
    RT_FAIL_IF(a % scalar != 0 || b % scalar != 0 || c % scalar != 0 || d % scalar != 0,
               "Non-integer division result");
    return ZOmega(a / scalar, b / scalar, c / scalar, d / scalar);
}

/**
 * @brief Converts the ZOmega element to a complex number.
 */
std::complex<double> ZOmega::to_complex() const
{
    return std::complex<double>(static_cast<double>(a) * std::pow(OMEGA, 3) +
                                static_cast<double>(b) * std::pow(OMEGA, 2) +
                                static_cast<double>(c) * OMEGA + static_cast<double>(d));
}

/**
 * @brief Return the parity indicating structure of real and imaginary parts as a DyadicMatrix
 * element.
 */
bool ZOmega::parity() const { return (a + c) % 2 != 0; }

/**
 * @brief Return the adjoint, i.e., the root-2 conjugate.
 */
ZOmega ZOmega::adj2() const { return ZOmega(-a, b, -c, d); }

/**
 * @brief Computes the norm of the ZOmega element. (x^T x)^dot (x^T x) (Definition 4,
 * arXiv:1212.6253) We call this norm4 to denote the fact that this is quartic.
 */
INT_TYPE ZOmega::norm4() const
{
    INT_TYPE first = a * a + b * b + c * c + d * d;
    INT_TYPE second = a * b + b * c + c * d - d * a;
    return first * first - 2 * second * second;
}

/**
 * @brief Computes the conjugate of the ZOmega element.
 */
ZOmega ZOmega::conj() const { return ZOmega(-c, -b, -a, d); }

/**
 * @brief Computes the 'norm' of the ZOmega element.
 */
ZOmega ZOmega::norm2() const { return (*this) * (*this).conj(); }

/**
 * @brief Computes the modulo of the ZOmega element. (Remark 6, arXiv:1212.6253)
 */
ZOmega ZOmega::operator%(const ZOmega &other) const
{
    INT_TYPE d = other.norm4();
    ZOmega other_z{other};
    ZOmega other_conj_z{other.conj()};
    ZOmega n = (*this) * other_conj_z * ((other_z * other_conj_z).adj2());

    INT_TYPE na = floor_div((n.a + floor_div(d, (INT_TYPE)2)), d);
    INT_TYPE nb = floor_div((n.b + floor_div(d, (INT_TYPE)2)), d);
    INT_TYPE nc = floor_div((n.c + floor_div(d, (INT_TYPE)2)), d);
    INT_TYPE nd = floor_div((n.d + floor_div(d, (INT_TYPE)2)), d);

    ZOmega result = ZOmega{na, nb, nc, nd} * other - (*this);

    return result;
}

/**
 * @brief Converts the ZOmega element to a ZSqrtTwo element.
 */
ZSqrtTwo ZOmega::to_sqrt_two() const
{
    if ((c + a == 0) && (b == 0)) {
        return ZSqrtTwo(d, (c - a) / 2);
    }
    RT_FAIL("Invalid ZOmega for conversion to ZSqrtTwo");
}

/**
 * @brief Normalize the ZOmega element and return the number of times 2 was factored out.
 */
std::pair<ZOmega, int> ZOmega::normalize()
{
    int ix = 0;
    ZOmega res = *this;
    INT_TYPE two(2);
    while (((res.a + res.c) % 2) == 0 && ((res.b + res.d) % 2) == 0) {
        INT_TYPE bd = res.b - res.d;
        INT_TYPE ac = res.a + res.c;
        INT_TYPE bd_sum = res.b + res.d;
        INT_TYPE ca = res.c - res.a;
        INT_TYPE new_a = floor_div(bd, two);
        INT_TYPE new_b = floor_div(ac, two);
        INT_TYPE new_c = floor_div(bd_sum, two);
        INT_TYPE new_d = floor_div(ca, two);
        res = ZOmega(new_a, new_b, new_c, new_d);
        ix += 1;
    }
    return {res, ix};
}

// --- DyadicMatrix Method Implementations ---

/**
 * @brief Constructor for DyadicMatrix.
 */
DyadicMatrix::DyadicMatrix(const ZOmega &a, const ZOmega &b, const ZOmega &c, const ZOmega &d,
                           INT_TYPE k)
    : a(a), b(b), c(c), d(d), k(k)
{
    normalize();
}

/**
 * @brief Reduce the k value of the dyadic matrix.
 */
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

/**
 * @brief Negation operator for DyadicMatrix elements.
 */
DyadicMatrix DyadicMatrix::operator-() const { return DyadicMatrix(-a, -b, -c, -d, k); }

bool DyadicMatrix::operator==(const DyadicMatrix &other) const
{
    return a == other.a && b == other.b && c == other.c && d == other.d && k == other.k;
}

/**
 * @brief Flatten the DyadicMatrix into an array of ZOmega elements.
 */
std::array<ZOmega, 4> DyadicMatrix::flatten() const { return {a, b, c, d}; }

/**
 * @brief Scalar multiplication operator for DyadicMatrix elements.
 */
DyadicMatrix DyadicMatrix::operator*(const ZOmega &scalar) const
{
    return DyadicMatrix(a * scalar, b * scalar, c * scalar, d * scalar, k);
}

// --- SO3Matrix Method Implementations ---

/**
 * @brief Constructor for SO3Matrix from DyadicMatrix.
 */
SO3Matrix::SO3Matrix(const DyadicMatrix &dy_mat) : dyadic_mat(dy_mat)
{
    from_dyadic_matrix(dy_mat);
    normalize();
}

/**
 * @brief Constructor for SO3Matrix from matrix and k value.
 */
SO3Matrix::SO3Matrix(const std::array<std::array<ZSqrtTwo, 3>, 3> &mat, INT_TYPE k)
    : so3_mat(mat), k(k)
{
    normalize();
}

/**
 * @brief Helper function to construct for SO3Matrix from DyadicMatrix.
 */
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

    const auto &[a_, b_, c_, d_] = z_sqrt2;

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

/**
 * @brief Equality operator for SO3Matrix elements.
 */
bool SO3Matrix::operator==(const SO3Matrix &other) const
{
    return so3_mat == other.so3_mat && k == other.k;
}

/**
 * @brief Return the parity matrix of the SO3Matrix element.
 */
std::array<std::array<int, 3>, 3> SO3Matrix::parity_mat() const
{
    std::array<std::array<int, 3>, 3> p_mat;
    for (std::size_t i = 0; i < 3; i++) {
        for (std::size_t j = 0; j < 3; j++) {
            INT_TYPE val = (so3_mat[i][j].a % 2 + 2) % 2;
            p_mat[i][j] = static_cast<int>(val);
        }
    }
    return p_mat;
}

/**
 * @brief Return the parity vector of the SO3Matrix element.
 */
std::array<int, 3> SO3Matrix::parity_vec() const
{
    auto p_mat = this->parity_mat();
    std::array<int, 3> p_vec;
    for (std::size_t i = 0; i < 3; i++) {
        p_vec[i] = std::accumulate(p_mat[i].begin(), p_mat[i].end(), 0);
    }
    return p_vec;
}

/**
 * @brief Flatten the SO3Matrix into an array of ZSqrtTwo elements.
 */
std::array<ZSqrtTwo, 9> SO3Matrix::flatten() const
{
    return {so3_mat[0][0], so3_mat[0][1], so3_mat[0][2], so3_mat[1][0], so3_mat[1][1],
            so3_mat[1][2], so3_mat[2][0], so3_mat[2][1], so3_mat[2][2]};
}

/**
 * @brief Reduce the k value of the SO(3) matrix.
 */
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

/**
 * @brief Return ZOmega element as A + 1j * B + shift, where A and B are ZSqrtTwo elements and shift
 * is ZOmega element.
 */
ZOmega zomega_from_sqrt_pair(const ZSqrtTwo &alpha, const ZSqrtTwo &beta, const ZOmega &shift)
{
    return ZOmega(beta.b - alpha.b, beta.a, beta.b + alpha.b, alpha.a) + shift;
}

/**
 * @brief Multiply two DyadicMatrix .
 */
DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2)
{
    return DyadicMatrix(m1.a * m2.a + m1.b * m2.c, m1.a * m2.b + m1.b * m2.d,
                        m1.c * m2.a + m1.d * m2.c, m1.c * m2.b + m1.d * m2.d, m1.k + m2.k);
}

/**
 * @brief Multiply two SO3Matrix.
 */
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
