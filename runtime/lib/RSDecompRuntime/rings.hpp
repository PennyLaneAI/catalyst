#pragma once

#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

const double LAMBDA = 1.0 + M_SQRT2;
const std::complex<double> OMEGA = std::complex<double>(M_SQRT1_2, M_SQRT1_2);

using INT_TYPE = __int128;
using MULTI_PREC_INT = boost::multiprecision::cpp_int;

struct ZOmega;
struct ZOmega_multiprec;
struct ZSqrtTwo;
struct DyadicMatrix;
struct SO3Matrix;

std::ostream &operator<<(std::ostream &os, const ZOmega &zomega);
std::ostream &operator<<(std::ostream &os, const ZSqrtTwo &zsqtwo);
std::ostream &operator<<(std::ostream &os, const SO3Matrix &matrix);

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
    // INT_TYPE abs() const;
    // MULTI_PREC_INT abs_multiprec() const;
    MULTI_PREC_INT abs() const;
    ZOmega conj() const;
    ZOmega norm() const;

    ZSqrtTwo to_sqrt_two() const;
    std::pair<ZOmega, int> normalize();
};

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

    friend DyadicMatrix dyadic_matrix_mul(const DyadicMatrix &m1, const DyadicMatrix &m2);
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

    friend SO3Matrix so3_matrix_mul(const SO3Matrix &m1, const SO3Matrix &m2);
};
