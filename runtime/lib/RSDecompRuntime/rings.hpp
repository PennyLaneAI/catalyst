#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <complex>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <optional> 
#include <boost/multiprecision/cpp_int.hpp>

const double LAMBDA = 1.0 + M_SQRT2;
const std::complex<double> OMEGA = std::complex<double>(1.0 / M_SQRT2, 1.0 / M_SQRT2);



// using INT_TYPE = long long;
using INT_TYPE = __int128;
using MULTI_PREC_INT = boost::multiprecision::cpp_int;
//using MULTI_PREC_INT = __int128;

std::ostream& operator<<(std::ostream& os, __int128 n);
std::ostream& operator<<(std::ostream& os, unsigned __int128 n);

inline INT_TYPE math_mod(INT_TYPE a, INT_TYPE n) {
    if (n == 0) {
        throw std::invalid_argument("Modulo by zero");
    }
    if (n < 0) {
        // Handle negative divisor, though typically not needed
        n = -n;
    }
    INT_TYPE r = a % n;
    return r < 0 ? r + n : r;
}

template<typename T>
inline T floor_div(T a, T b) {
    if (b == 0) {
        throw std::invalid_argument("Division by zero");
    }
    T q = a / b;
    T r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
        q -= 1;
    }
    return q;
}

class ZOmega; // Forward declaration
class ZOmega_multiprec; // <-- ADDED forward declaration
class ZSqrtTwo
{
public:
    INT_TYPE a, b;
    ZSqrtTwo(INT_TYPE a = 0, INT_TYPE b = 0) : a(a), b(b) {}

    ZSqrtTwo operator+(const ZSqrtTwo &other) const
    {
        return ZSqrtTwo(a + other.a, b + other.b);
    }

    ZSqrtTwo operator-(const ZSqrtTwo &other) const
    {
        return ZSqrtTwo(a - other.a, b - other.b);
    }

    ZSqrtTwo operator*(const ZSqrtTwo &other) const
    {
        return ZSqrtTwo(a * other.a + 2 * b * other.b, a * other.b + b * other.a);
    }

    ZSqrtTwo operator*(INT_TYPE scalar) const
    {
        return ZSqrtTwo(a * scalar, b * scalar);
    }
    
    ZSqrtTwo operator/(ZSqrtTwo other) const
    {
        return (*this * other.adj2()) / other.abs();
    }

    ZSqrtTwo operator/(INT_TYPE scalar) const
    {
        if (scalar == 0)
        {
            throw std::invalid_argument("Division by zero");
        }
        if (a % scalar != 0 || b % scalar != 0)
        {
            throw std::invalid_argument("Non-integer division result");
        }
        return ZSqrtTwo(a / scalar, b / scalar);
    }

    bool operator==(const ZSqrtTwo &other) const
    {
        return a == other.a && b == other.b;
    }


    INT_TYPE abs() const
    {
        return a * a - 2 * b * b;
    }

    ZSqrtTwo adj2() const
    {
        return ZSqrtTwo(a, -b);
    }

    double to_double() const
    {
        return static_cast<double>(a) + static_cast<double>(b) * M_SQRT2;
    }

    ZSqrtTwo pow(int exponent) const
    {
        if (exponent < 0)
        {
            throw std::invalid_argument("Negative exponent not supported for ZSqrtTwo");
        }
        ZSqrtTwo result(1, 0);
        ZSqrtTwo base = *this;
        while (exponent > 0)
        {
            if (exponent % 2 == 1)
            {
                result = result * base;
            }
            base = base * base;
            exponent /= 2;
        }
        return result;
    }

    ZSqrtTwo operator%(const ZSqrtTwo &other) const
    {
        INT_TYPE d = other.abs();
        if (d == 0)
        {
            throw std::invalid_argument("Modulo by zero in ZSqrtTwo");
        }

        // To find the quotient, we compute (self / other) in the field of fractions
        // and round its components to the nearest integer.
        // self / other = (self * other.adj2()) / d
        ZSqrtTwo num = *this * other.adj2();

        // Round the components of the fractional quotient to find the integer quotient
        auto q1 = static_cast<INT_TYPE>(std::nearbyint(static_cast<double>(num.a) / d));
        auto q2 = static_cast<INT_TYPE>(std::nearbyint(static_cast<double>(num.b) / d));
        ZSqrtTwo quotient(q1, q2);

        // Remainder r = self - quotient * other
        return *this - quotient * other;
    }

    std::optional<ZSqrtTwo> sqrt() const
    {
        // d := abs(self)
        const INT_TYPE d = this->abs();

        const INT_TYPE r = static_cast<INT_TYPE>(std::sqrt(d));

        // if r * r != d: return None
        if (r * r != d)
        {
            // The norm is not a perfect square, so no integer square root exists.
            return std::nullopt;
        }

        for (INT_TYPE s : {1, -1})
        {
            INT_TYPE x_numerator = a + s * r;
            INT_TYPE y_numerator = a - s * r;

            INT_TYPE x = static_cast<INT_TYPE>(std::sqrt(x_numerator / 2));
            INT_TYPE y = static_cast<INT_TYPE>(std::sqrt(y_numerator / 4));

            ZSqrtTwo zrt{x, y};
            if (zrt * zrt == *this)
            {
                return zrt;
            }

            ZSqrtTwo art = zrt.adj2(); // Assumes a conjugate() method exists
            if (art * art == *this)
            {
                return art;
            }
        }

        // No solution found
        return std::nullopt;
    }

    ZOmega to_omega() const;
};


class ZOmega
{
public:
    INT_TYPE a, b, c, d;

public:
    ZOmega(INT_TYPE a = 0, INT_TYPE b = 0, INT_TYPE c = 0, INT_TYPE d = 0)
        : a(a), b(b), c(c), d(d) {}

    ZOmega operator-() const
    {
        return ZOmega(-a, -b, -c, -d);
    }

    ZOmega operator*(const ZOmega &other) const
    {
        return ZOmega(
            a * other.d + b * other.c + c * other.b + d * other.a,
            b * other.d + c * other.c + d * other.b - a * other.a,
            c * other.d + d * other.c - a * other.b - b * other.a,
            d * other.d - a * other.c - b * other.b - c * other.a);
    }

    ZOmega operator*(INT_TYPE scalar) const
    {
        return ZOmega(a * scalar, b * scalar, c * scalar, d * scalar);
    }

    ZOmega operator/(INT_TYPE scalar) const
    {
        if (scalar == 0)
        {
            throw std::invalid_argument("Division by zero");
        }
        if (a % scalar != 0 || b % scalar != 0 || c % scalar != 0 || d % scalar != 0)
        {
            throw std::invalid_argument("Non-integer division result");
        }
        return ZOmega(a / scalar, b / scalar, c / scalar, d / scalar);
    }

    ZOmega operator+(const ZOmega &other) const
    {
        return ZOmega(a + other.a, b + other.b, c + other.c, d + other.d);
    }

    ZOmega operator-(const ZOmega &other) const
    {
        return ZOmega(a - other.a, b - other.b, c - other.c, d - other.d);
    }

    bool operator==(const ZOmega &other) const
    {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }

    std::complex<double> to_complex() const
    {
        return std::complex<double>(static_cast<double>(a) * std::pow(OMEGA, 3) +
                                    static_cast<double>(b) * std::pow(OMEGA, 2) +
                                    static_cast<double>(c) * OMEGA + static_cast<double>(d));
    }

    bool parity() const
    {
        return (a + c) % 2 != 0;
    }

    ZOmega adj2() const
    {
        return ZOmega(-a, b, -c, d);
    }

    INT_TYPE abs() const
    {
        INT_TYPE first = a * a + b * b + c * c + d * d;
        INT_TYPE second = a * b + b * c + c * d - d * a;
        return first * first - 2 * second * second;
    }

    MULTI_PREC_INT abs_multiprec() const
    {
        MULTI_PREC_INT first = a * a + b * b + c * c + d * d;
        MULTI_PREC_INT second = a * b + b * c + c * d - d * a;
        return first * first - 2 * second * second;
    }

    ZOmega conj() const
    {
        return ZOmega(-c, -b, -a, d);
    }

    ZOmega norm() const
    {
        return (*this) * (*this).conj();
    }

    /* ZOmega operator%(const ZOmega &other) const
    {
        INT_TYPE d = other.abs();
        ZOmega n = *this * other.conj() * ((other * other.conj()).adj2());
        return ZOmega(floor_div((n.a + floor_div(d,2)) , d),
                      floor_div((n.b + floor_div(d,2)) , d),
                      floor_div((n.c + floor_div(d,2)) , d),
                      floor_div((n.d + floor_div(d,2)) , d)) * other -
               *this;
    } */

    // MODIFIED: Declare operator% here, define it after ZOmega_multiprec
    ZOmega operator%(const ZOmega &other) const;

    ZSqrtTwo to_sqrt_two() const
    {
        if ((c + a == 0) && (b == 0))
        {
            return ZSqrtTwo(d, (c - a) / 2);
        }
        throw std::invalid_argument("Invalid ZOmega for conversion to ZSqrtTwo");
    }

    std::pair<ZOmega, int> normalize() {
        int ix = 0;
        ZOmega res = *this;
        while (((res.a + res.c) % 2) == 0 && ((res.b + res.d) % 2) == 0) {
            // FIX 3: Cast the literal '2' to INT_TYPE for correct template deduction
            INT_TYPE a = floor_div(res.b - res.d, (INT_TYPE)2);
            INT_TYPE b = floor_div(res.a + res.c, (INT_TYPE)2);
            INT_TYPE c = floor_div(res.b + res.d, (INT_TYPE)2);
            INT_TYPE d = floor_div(res.c - res.a, (INT_TYPE)2);
            res = ZOmega(a, b, c, d);
            ix += 1;
        }
        return {res, ix};
    }
};

// --- FIX 1: Moved ZOmega_multiprec definition after ZOmega ---
// This ensures ZOmega is a complete type when used in the constructor.
class ZOmega_multiprec{
public:
    MULTI_PREC_INT a, b, c, d;

    ZOmega_multiprec(MULTI_PREC_INT a = 0, MULTI_PREC_INT b = 0, MULTI_PREC_INT c = 0, MULTI_PREC_INT d = 0)
        : a(a), b(b), c(c), d(d) {}

    // MODIFIED: Declare constructor here, define it after the class
    ZOmega_multiprec(ZOmega zomega);


    ZOmega_multiprec operator*(const ZOmega_multiprec &other) const
    {
        return ZOmega_multiprec(
            a * other.d + b * other.c + c * other.b + d * other.a,
            b * other.d + c * other.c + d * other.b - a * other.a,
            c * other.d + d * other.c - a * other.b - b * other.a,
            d * other.d - a * other.c - b * other.b - c * other.a);
    }


    ZOmega_multiprec adj2() const
    {
        return ZOmega_multiprec(-a, b, -c, d);
    }

    ZOmega_multiprec operator-(ZOmega_multiprec other) const
    {
        return ZOmega_multiprec(a - other.a, b - other.b, c - other.c, d - other.d);
    }

};
// --- End of FIX 1 ---


// --- ADDED DEFINITIONS ---
// Define the constructor for ZOmega_multiprec (needs ZOmega definition, which is now complete)
inline ZOmega_multiprec::ZOmega_multiprec(ZOmega zomega) : a(zomega.a), b(zomega.b), c(zomega.c), d(zomega.d) {}

// Define ZOmega::operator% (needs ZOmega_multiprec definition, which is now complete)
inline ZOmega ZOmega::operator%(const ZOmega &other) const
{
    MULTI_PREC_INT d = other.abs_multiprec();
    ZOmega_multiprec other_multiprec{other};
    ZOmega_multiprec other_conj_multiprec{other.conj()};
    ZOmega_multiprec n = ZOmega_multiprec(*this) * other_conj_multiprec * ((other_multiprec * other_conj_multiprec).adj2());

    // FIX 2: Cast the literal '2' to MULTI_PREC_INT for correct template deduction
    MULTI_PREC_INT na = floor_div((n.a + floor_div(d,(MULTI_PREC_INT)2)) , d);
    MULTI_PREC_INT nb = floor_div((n.b + floor_div(d,(MULTI_PREC_INT)2)) , d);
    MULTI_PREC_INT nc = floor_div((n.c + floor_div(d,(MULTI_PREC_INT)2)) , d);
    MULTI_PREC_INT nd = floor_div((n.d + floor_div(d,(MULTI_PREC_INT)2)) , d);

    ZOmega_multiprec result = ZOmega_multiprec{na, nb, nc, nd} * other - ZOmega_multiprec{*this};

    return ZOmega(result.a.convert_to<INT_TYPE>(),
                  result.b.convert_to<INT_TYPE>(),
                  result.c.convert_to<INT_TYPE>(),
                  result.d.convert_to<INT_TYPE>());
    // return ZOmega(result.a, result.b, result.c, result.d);
}
// --- END ADDED DEFINITIONS ---


inline ZOmega ZSqrtTwo::to_omega() const
{
    return ZOmega(-b, 0, b, a);
}
ZOmega zomega_from_sqrt_pair(const ZSqrtTwo &alpha, const ZSqrtTwo &beta, const ZOmega &shift)
{
    return ZOmega(beta.b - alpha.b, beta.a, beta.b + alpha.b, alpha.a) + shift;
}

class DyadicMatrix
{
public:
    ZOmega a, b, c, d;
    INT_TYPE k;

    void normalize();

public:
    DyadicMatrix(const ZOmega &a, const ZOmega &b, const ZOmega &c, const ZOmega &d, INT_TYPE k = 0)
        : a(a), b(b), c(c), d(d), k(k) { normalize(); }

    DyadicMatrix operator-() const
    {
        return DyadicMatrix(-a, -b, -c, -d, k);
    }

    bool operator==(const DyadicMatrix &other) const
    {
        return a == other.a && b == other.b && c == other.c && d == other.d && k == other.k;
    }

    std::array<ZOmega, 4> flatten() const
    {
        return {a, b, c, d};
    }
    friend DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2);

    DyadicMatrix operator*(const ZOmega &scalar) const
    {
        return DyadicMatrix(a * scalar, b * scalar, c * scalar, d * scalar, k);
    }
};

class SO3Matrix
{
public:
    std::array<std::array<ZSqrtTwo, 3>, 3> so3_mat;
    // DyadicMatrix dyadic_mat;
    INT_TYPE k;
    void normalize();

    void from_dyadic_matrix(const DyadicMatrix &dy_mat)
    {
        const auto &su2_elems = dy_mat.flatten();
        INT_TYPE current_k = 2 * dy_mat.k;

        bool has_parity = false;
        for (const auto &s : su2_elems)
        {
            if (s.parity())
            {
                has_parity = true;
                break;
            }
        }

        std::array<std::pair<ZSqrtTwo, ZSqrtTwo>, 4> z_sqrt2;
        if (has_parity)
        {
            current_k += 2;
            for (size_t i = 0; i < su2_elems.size(); ++i)
            {
                const auto &s = su2_elems[i];
                z_sqrt2[i] = {ZSqrtTwo((s.c - s.a), s.d), ZSqrtTwo((s.c + s.a), s.b)};
            }
        }
        else
        {
            for (size_t i = 0; i < su2_elems.size(); ++i)
            {
                const auto &s = su2_elems[i];
                // Integer division `//` in Python is the default for integers in C++
                z_sqrt2[i] = {ZSqrtTwo(s.d, (s.c - s.a) / 2), ZSqrtTwo(s.b, (s.c + s.a) / 2)};
            }
        }

        const auto &a_ = z_sqrt2[0];
        const auto &b_ = z_sqrt2[1];
        const auto &c_ = z_sqrt2[2];
        const auto &d_ = z_sqrt2[3];

        // --- SO(3) Matrix Construction ---
        // This is a direct translation of the arithmetic formulas.
        // Python's a_[0] becomes a_.first, a_[1] becomes a_.second
        so3_mat[0][0] = a_.first * d_.first + a_.second * d_.second + b_.first * c_.first + b_.second * c_.second;
        so3_mat[0][1] = a_.second * d_.first + b_.first * c_.second - b_.second * c_.first - a_.first * d_.second;
        so3_mat[0][2] = a_.first * c_.first + a_.second * c_.second - b_.first * d_.first - b_.second * d_.second;

        so3_mat[1][0] = a_.first * d_.second - a_.second * d_.first + b_.first * c_.second - b_.second * c_.first;
        so3_mat[1][1] = a_.first * d_.first + a_.second * d_.second - b_.first * c_.first - b_.second * c_.second;
        so3_mat[1][2] = a_.first * c_.second - a_.second * c_.first - b_.first * d_.second + b_.second * d_.first;

        // Using the `square` helper for clarity, same as `x * x`
        so3_mat[2][0] = (a_.first * b_.first + a_.second * b_.second) * 2;
        so3_mat[2][1] = (a_.second * b_.first - a_.first * b_.second) * 2;
        so3_mat[2][2] = a_.first * a_.first + a_.second * a_.second - b_.first * b_.first - b_.second * b_.second;

        // --- Final State Update ---
        this->k = current_k;
        // No return statement needed as we modified the member variable directly
    }

public:
    SO3Matrix(const DyadicMatrix &dy_mat)
    {
        from_dyadic_matrix(dy_mat);
        normalize();
    }

    SO3Matrix(const std::array<std::array<ZSqrtTwo, 3>, 3> &mat, int k = 0) : k(k)
    {
        so3_mat = mat;
        normalize();
    }

    bool operator==(const SO3Matrix &other) const
    {
        return so3_mat == other.so3_mat && k == other.k;
    }
    std::array<std::array<int, 3>, 3> parity_mat() const
    {
        std::array<std::array<int, 3>, 3> p_mat;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                p_mat[i][j] = (so3_mat[i][j].a % 2 + 2) % 2;
            }
        }
        return p_mat;
    }

    std::array<int, 3> parity_vec() const
    {
        auto p_mat = this->parity_mat();
        std::array<int, 3> p_vec;

        for (size_t i = 0; i < 3; ++i)
        {
            // Sum the elements of the current row
            p_vec[i] = std::accumulate(p_mat[i].begin(), p_mat[i].end(), 0);
        }

        return p_vec;
    }

    std::array<ZSqrtTwo, 9> flatten() const
    {
        return {so3_mat[0][0], so3_mat[0][1], so3_mat[0][2],
                so3_mat[1][0], so3_mat[1][1], so3_mat[1][2],
                so3_mat[2][0], so3_mat[2][1], so3_mat[2][2]};
    }

    DyadicMatrix dyadic_matrix()
    {
        // Placeholder: Actual conversion logic needed
        return DyadicMatrix({}, {}, {}, {}, 0);
    }

    friend SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2);
};

void DyadicMatrix::normalize()
{
   // Placeholder normalization logic
}

void SO3Matrix::normalize()
// DOUBLE CHECK
{
    auto elements = this->flatten();

    if (std::all_of(elements.begin(), elements.end(), [](const ZSqrtTwo &z)
                    { return z.a == 0 && z.b == 0; }))
    {
        k = 0;
        return;
    }

    while (std::all_of(elements.begin(), elements.end(), [](const ZSqrtTwo &z)
                       { return z.a % 2 == 0 && z.b % 2 == 0; }))
    {
        for (auto &elem : elements)
        {
            elem = elem / 2;
        }
        k -= 2;
    }

    while (std::all_of(elements.begin(), elements.end(), [](const ZSqrtTwo &z)
                       { return z.a % 2 == 0; }))
    {
        for (auto &elem : elements)
        {
            elem = ZSqrtTwo(elem.b, elem.a / 2);
        }
        k -= 1;
    }
    so3_mat[0] = {elements[0], elements[1], elements[2]};
    so3_mat[1] = {elements[3], elements[4], elements[5]};
    so3_mat[2] = {elements[6], elements[7], elements[8]};
}

DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2)
{
    return DyadicMatrix(m1.a * m2.a + m1.b * m2.c, m1.a * m2.b + m1.b * m2.d, m1.c * m2.a + m1.d * m2.c, m1.c * m2.b + m1.d * m2.d, m1.k + m2.k);
}

SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2)
{
    std::array<std::array<ZSqrtTwo, 3>, 3> result_mat{};
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            result_mat[i][j] = ZSqrtTwo(0, 0);
            for (size_t k = 0; k < 3; ++k)
            {
                result_mat[i][j] = result_mat[i][j] + (m1.so3_mat[i][k] * m2.so3_mat[k][j]);
            }
        }
    }
    SO3Matrix result(result_mat, m1.k + m2.k);
    return result;
}

std::ostream &operator<<(std::ostream &os, const SO3Matrix &matrix)
{
    os << "SO3Matrix(k=" << matrix.k << ", mat=[";
    for (const auto &row : matrix.so3_mat)
    {
        os << "[";
        for (const auto &elem : row)
        {
            os << "(" << elem.a << " + " << elem.b << "√2), ";
        }
        os << "], " << std::endl;
    }
    os << "])" << std::endl
       << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const ZOmega &zomega)
{
    os << "ZOmega(" << zomega.a << " ω^3 + " << zomega.b << "ω^2 + " << zomega.c << "ω + " << zomega.d << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const ZSqrtTwo &zsqtwo)
{
    os << "ZSqrtTwo(" << zsqtwo.a << " + " << zsqtwo.b << "√2)";
    return os;
}


std::ostream& operator<<(std::ostream& os, __int128_t value) {
    // Handle the zero case explicitly
    if (value == 0) {
        os << "0";
        return os;
    }

    std::string str;
    bool is_negative = false;

    // Handle negative numbers
    if (value < 0) {
        is_negative = true;
        // Work with the absolute value.
        // Avoid simple negation (value = -value) which can overflow
        // if value is the minimum representable __int128_t.
        // Instead, we extract digits from the positive equivalent.
        // We do this by taking modulo 10, which gives a negative
        // result, and then negating that digit.
    }

    // Extract digits in reverse order
    while (value != 0) {
        int digit;
        if (is_negative) {
            // For negative numbers, (value % 10) is <= 0.
            // e.g., -123 % 10 = -3. We negate it to get '3'.
            digit = -(value % 10);
            value /= 10;
        } else {
            // For positive numbers, (value % 10) is >= 0.
            digit = value % 10;
            value /= 10;
        }
        str += (char)('0' + digit);
    }

    // Add the negative sign if necessary
    if (is_negative) {
        str += '-';
    }

    // The string was built in reverse (e.g., "321-" for -123)
    std::reverse(str.begin(), str.end());

    // Send the final string to the output stream
    os << str;
    return os;
}
