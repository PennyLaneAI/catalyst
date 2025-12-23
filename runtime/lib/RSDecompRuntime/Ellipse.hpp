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

#include "Exception.hpp"
#include "RSUtils.hpp"
#include "Rings.hpp"
#include <array>
#include <cmath>
#include <string_view>
#include <utility>

namespace RSDecomp::GridProblem {

using namespace RSDecomp::Rings;
using namespace RSDecomp::Utils;
struct GridOp;
struct Ellipse;
struct EllipseState;

/**
 * @class GridOp
 * @brief GridOperator
 *
 * A Grid Operator G represented by a 2x2 matrix. The elements {a, b, c, d} are given as an array
 * corresponding to a = a[0] + a[1]/sqrt2 (Def 5.10, Lemma 5.11, arXiv:1403.2975).
 */
struct GridOp {
    std::array<INT_TYPE, 2> a, b, c, d;

    /**
     * @brief Constructor for GridOp.
     * @param a_val
     * @param b_val
     * @param c_val
     * @param d_val
     * @param check_valid If true, validates the grid operation properties.
     */
    GridOp(const std::array<INT_TYPE, 2> &a_val, const std::array<INT_TYPE, 2> &b_val,
           const std::array<INT_TYPE, 2> &c_val, const std::array<INT_TYPE, 2> &d_val,
           bool check_valid = true)
        : a(a_val), b(b_val), c(c_val), d(d_val)
    {
        if (check_valid) {
            RT_FAIL_IF((a[0] + b[0] + c[0] + d[0]) % 2 != 0,
                       "sum of a_0, b_0, c_0, d_0 must be even");

            RT_FAIL_IF(!(((a[1] & 1) == (b[1] & 1)) && ((b[1] & 1) == (c[1] & 1)) &&
                         ((c[1] & 1) == (d[1] & 1))),
                       "a_1, b_1, c_1, d_1 must have same parity");
        }
    }

    GridOp() : a({1, 0}), b({0, 0}), c({0, 0}), d({1, 0}) {}

    bool operator==(const GridOp &other) const
    {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }

    static GridOp from_string(std::string_view s);

    /**
     * @brief Multiplies this GridOp with another GridOp.
     * @param other The other GridOp to multiply with.
     * @return The resulting GridOp after multiplication.
     */
    GridOp operator*(const GridOp &other) const
    {
        return GridOp(
            {a[0] * other.a[0] + b[0] * other.c[0] + (a[1] * other.a[1] + b[1] * other.c[1]) / 2,
             a[0] * other.a[1] + a[1] * other.a[0] + b[0] * other.c[1] + b[1] * other.c[0]},
            {a[0] * other.b[0] + b[0] * other.d[0] + (a[1] * other.b[1] + b[1] * other.d[1]) / 2,
             a[0] * other.b[1] + a[1] * other.b[0] + b[0] * other.d[1] + b[1] * other.d[0]},
            {c[0] * other.a[0] + d[0] * other.c[0] + (c[1] * other.a[1] + d[1] * other.c[1]) / 2,
             c[0] * other.a[1] + c[1] * other.a[0] + d[0] * other.c[1] + d[1] * other.c[0]},
            {c[0] * other.b[0] + d[0] * other.d[0] + (c[1] * other.b[1] + d[1] * other.d[1]) / 2,
             c[0] * other.b[1] + c[1] * other.b[0] + d[0] * other.d[1] + d[1] * other.d[0]});
    }

    /**
     * @brief Multiplies this GridOp with a ZOmega element.
     * @param other The ZOmega element to multiply with.
     * @return The resulting ZOmega after multiplication.
     */
    ZOmega operator*(const ZOmega &other) const
    {
        INT_TYPE x1 = other.d, y1 = other.b;
        INT_TYPE x2 = other.c - other.a, y2 = other.c + other.a;
        INT_TYPE a_ = a[0] * x2 + a[1] * x1 + b[0] * y2 + b[1] * y1;
        INT_TYPE c_ = c[0] * x2 + c[1] * x1 + d[0] * y2 + d[1] * y1;
        INT_TYPE d_ = a[0] * x1 + b[0] * y1 + (a[1] * x2 + b[1] * y2) / 2;
        INT_TYPE b_ = c[0] * x1 + d[0] * y1 + (c[1] * x2 + d[1] * y2) / 2;
        return {(c_ - a_) / 2, b_, (c_ + a_) / 2, d_};
    }

    /**
     * @brief Computes the determinant of the GridOp.
     * @return The determinant a = A[0] + A[1]/sqrt(2)
     */
    std::array<INT_TYPE, 2> determinant() const
    {
        return {a[0] * d[0] - b[0] * c[0] + (a[1] * d[1] - b[1] * c[1]) / 2,
                a[0] * d[1] - b[0] * c[1] + a[1] * d[0] - b[1] * c[0]};
    }

    // https://arxiv.org/pdf/1403.2975 Definition 5.10
    /**
     * @brief Checks if the GridOp is a special grid operator. (Prop 5.13, arXiv:1403.2975)
     */
    bool is_special() const
    {
        auto det = determinant();
        return det[1] == 0 && (det[0] == 1 || det[0] == -1);
    }

    /**
     * @brief Flattens the GridOp into a vector of doubles. (Lemma 5.11, arXiv:1403.2975)
     */
    std::array<double, 4> flatten() const
    {
        return {static_cast<double>(a[0]) + static_cast<double>(a[1]) / M_SQRT2,
                static_cast<double>(b[0]) + static_cast<double>(b[1]) / M_SQRT2,
                static_cast<double>(c[0]) + static_cast<double>(c[1]) / M_SQRT2,
                static_cast<double>(d[0]) + static_cast<double>(d[1]) / M_SQRT2};
    }

    /**
     * @brief Computes the inverse of the GridOp.
     */
    GridOp inverse() const
    {
        auto det = determinant();
        INT_TYPE det1 = det[0];

        RT_FAIL_IF(!(det[1] == 0 && (det1 == 1 || det1 == -1)),
                   "Grid operator needs to be special to have an inverse.");

        return GridOp({d[0] / det1, d[1] / det1}, {-b[0] / det1, -b[1] / det1},
                      {-c[0] / det1, -c[1] / det1}, {a[0] / det1, a[1] / det1});
    }

    /**
     * @brief Computes the transpose of the GridOp.
     */
    GridOp transpose() const { return GridOp(d, b, c, a); }

    /**
     * @brief Compute the sqrt(2)-conjugate of the grid operation.
     */
    GridOp adj2() const
    {
        return GridOp({a[0], -a[1]}, {b[0], -b[1]}, {c[0], -c[1]}, {d[0], -d[1]});
    }

    /**
     * @brief Apply a shift operator to the grid operation based on (Lemma A.9, arXiv:1403.2975).
     */
    GridOp apply_shift_op(INT_TYPE k) const
    {
        INT_TYPE sign = (k < 0) ? -1 : 1;
        INT_TYPE k_abs = abs_val(k);
        auto s = ZSqrtTwo(1, 1).pow(k_abs);
        INT_TYPE s1 = s.a;
        INT_TYPE s2 = s.b;
        return GridOp({a[1] * s2 + sign * a[0] * s1, 2 * a[0] * s2 + sign * a[1] * s1}, b, c,
                      {d[1] * s2 - sign * d[0] * s1, 2 * d[0] * s2 - sign * d[1] * s1});
    }

    GridOp pow(INT_TYPE n) const;
    Ellipse apply_to_ellipse(const Ellipse &ellipse) const;
    EllipseState apply_to_state(const EllipseState &state) const;
};

/**
 * @brief Creates a GridOp from its string representation.
 * @param s Supported string are: "I", "R", "A", "B", "K", "X", "Z". (Fig 6, arXiv:1403.2975).
 * @return The corresponding GridOp.
 */
inline GridOp GridOp::from_string(std::string_view s)
{
    if (s.empty()) {
        return GridOp({1, 0}, {0, 0}, {0, 0}, {1, 0}); // Default I
    }

    switch (s[0]) {
    case 'I':
        return GridOp({1, 0}, {0, 0}, {0, 0}, {1, 0});
    case 'A':
        return GridOp({1, 0}, {-2, 0}, {0, 0}, {1, 0});
    case 'B':
        return GridOp({1, 0}, {0, 2}, {0, 0}, {1, 0});
    case 'K':
        return GridOp({-1, 1}, {0, -1}, {1, 1}, {0, 1});
    case 'R':
        return GridOp({0, 1}, {0, -1}, {0, 1}, {0, 1});
    case 'U':
        return GridOp({1, 2}, {0, 0}, {0, 0}, {-1, 2});
    case 'X':
        return GridOp({0, 0}, {1, 0}, {1, 0}, {0, 0});
    case 'Z':
        return GridOp({1, 0}, {0, 0}, {0, 0}, {-1, 0});
    default:
        return GridOp({1, 0}, {0, 0}, {0, 0}, {1, 0}); // Default I
    }
}

/**
 * @brief Raises the GridOp to the power of n.
 * @param n The exponent to raise the GridOp to.
 * @return The resulting GridOp after exponentiation.
 */
inline GridOp GridOp::pow(INT_TYPE n) const
{
    if (*this == from_string("I") || n == 0) {
        return from_string("I");
    }
    if (*this == from_string("A")) {
        return GridOp({1, 0}, {-2 * n, 0}, {0, 0}, {1, 0});
    }
    if (*this == from_string("B")) {
        return GridOp({1, 0}, {0, 2 * n}, {0, 0}, {1, 0});
    }
    if (*this == from_string("U")) {
        auto c = ZSqrtTwo(1, 1).pow(abs_val(n));
        INT_TYPE c1 = c.a, c2 = c.b;
        INT_TYPE c3 = (n < 0) ? -1 : 1;
        return GridOp({c3 * c1, 2 * c2}, {0, 0}, {0, 0}, {-c3 * c1, 2 * c2});
    }

    GridOp res = from_string("I");
    GridOp x = *this;
    while (n > 0) {
        if (n % 2 == 1)
            res = res * x;
        x = x * x;
        n /= 2;
    }
    return res;
}

/**
 * @class Ellipse
 * @brief A class representing an ellipse as a positive definite matrix
 *
 * Represents an ellipse as a positive definite matrix D = [[a, b], [b, d]], with centre p.
 *
 * z and e are useful for defining the EllipseState as a pair of Ellipses:
 * D = [[e * λ^(-z), b], [b, e * λ^(z)]] (Eq 31, arXiv:1403.2975)
 */
struct Ellipse {
    double a, b, d;
    std::array<double, 2> p;

    // Useful parameters for constructing EllipseState
    // a = eλ^{-z}, d = eλ^z, b^2 = e^2 - 1; (Eq 31, arXiv:1403.2975)
    // 2z * log(λ) = log(d / a) => z = 0.5 * log(d / a) / log(λ)
    double z;
    double e;

    /**
     * @brief Constructor for Ellipse.
     * @param D The ellipse matrix coefficients as {a, b, d}.
     * @param p_val The centre of the ellipse as {px, py}.
     */
    Ellipse(const std::array<double, 3> &D = {1.0, 0.0, 1.0},
            const std::array<double, 2> &p_val = {0.0, 0.0})
        : p(p_val)
    {
        a = D[0];
        b = D[1];
        d = D[2];
        z = 0.5 * std::log2(d / a) / std::log2(LAMBDA_D);
        e = std::sqrt(a * d);
    }

    bool operator==(const Ellipse &other) const
    {
        return a == other.a && b == other.b && d == other.d && p == other.p;
    }

    /**
     * @brief Create an ellipse that bounds the region u such that u.z >= 1 - ε^2 / 2,
        with u ∈ 1/√2^k * Z[ω]` and z = e^{-i*theta / 2}`. (Eq 14, arXiv:1403.2975)
     */
    static Ellipse from_region(double theta, double epsilon, int k = 0)
    {
        double t = epsilon * epsilon / 2.0;
        double scale = std::pow(2.0, static_cast<double>(k) / 2.0);
        double a_val = scale * t;
        double b_val = scale * epsilon;
        double a2 = 1.0 / (a_val * a_val);
        double b2 = 1.0 / (b_val * b_val);
        double d2 = a2 - b2;
        double zx = std::cos(theta);
        double zy = std::sin(theta);
        double new_a = d2 * zx * zx + b2;
        double new_d = d2 * zy * zy + b2;
        double new_b = d2 * zx * zy;
        double const_val = 1.0 - t;
        std::array<double, 2> p_val = {const_val * scale * zx, const_val * scale * zy};
        return Ellipse({new_a, new_b, new_d}, p_val);
    }

    /**
     * @brief Calculate the discriminant of the characteristic polynomial associated with the
     * ellipse.
     */
    double discriminant() const { return (a + d) * (a + d) - 4 * (a * d - b * b); }

    /**
     * @brief Calculate the determinant of the ellipse."
     */
    double determinant() const { return a * d - b * b; }

    /**
     * @brief Check if the ellipse is positive semi-definite.
     */
    bool positive_semi_definite() const { return (a + d) + std::sqrt(discriminant()) >= 0; }

    /**
     * @brief Calculate the uprightness of the ellipse (Eq. 32, arXiv:1403.2975).
     */
    double uprightness() const { return M_PI / (4.0 * e * e); }

    /**
     * @brief alculate the b value of the ellipse from its uprightness (Eq. 33, arXiv:1403.2975).
     */
    static double b_from_uprightness(double up)
    {
        double temp = M_PI / (4.0 * up);
        return std::sqrt(temp * temp - 1.0);
    }

    /**
     * @brief Check if the point (x, y) is inside the ellipse.
     */
    bool contains(double x, double y) const
    {
        double x_ = x - p[0];
        double y_ = y - p[1];
        return (a * x_ * x_ + 2 * b * x_ * y_ + d * y_ * y_) <= 1.0;
    }

    /**
     * @brief Normalize the ellipse so that its determinant is 1.
     */
    std::pair<Ellipse, double> normalize() const
    {
        double s_val = 1.0 / std::sqrt(determinant());
        return {scale(s_val), s_val};
    }

    /**
     * @brief Scale the ellipse by a factor of scale.
     */
    Ellipse scale(double scale_factor) const
    {
        std::array<double, 3> D = {a * scale_factor, b * scale_factor, d * scale_factor};
        return Ellipse(D, p);
    }

    /**
     * @brief Compute the x-points of the ellipse for a given y-value.
     */
    std::pair<double, double> x_points(double y) const
    {
        double y_shifted = y - p[1];
        double discriminant = y_shifted * y_shifted * (b * b - a * d) + a;

        if (discriminant < 0) {
            double det = determinant();
            double y_extent = std::sqrt(a / det);
            double y_min = p[1] - y_extent;
            double y_max = p[1] + y_extent;

            std::stringstream ss;
            ss << "Cannot compute x_points: y=" << y << " is outside ellipse bounds [" << y_min
               << ", " << y_max << "]";

            RT_FAIL(ss.str().c_str());
        }

        double d0 = std::sqrt(discriminant);
        double x1 = (-b * y_shifted - d0) / a;
        double x2 = (-b * y_shifted + d0) / a;
        return {p[0] + x1, p[0] + x2};
    }

    /**
     * @brief Compute the y-points of the ellipse for a given x-value.
     */
    std::pair<double, double> y_points(double x) const
    {
        double x_shifted = x - p[0];
        double discriminant =
            (b * x_shifted) * (b * x_shifted) - d * (a * x_shifted * x_shifted - 1.0);

        if (discriminant < 0) {
            double det = determinant();
            double x_extent = std::sqrt(d / det);
            double x_min = p[0] - x_extent;
            double x_max = p[0] + x_extent;

            std::stringstream ss;
            ss << "Cannot compute y_points: x=" << x << " is outside ellipse bounds [" << x_min
               << ", " << x_max << "]";
            RT_FAIL(ss.str().c_str());
        }

        double d0 = std::sqrt(discriminant);
        double y1 = (-b * x_shifted - d0) / d;
        double y2 = (-b * x_shifted + d0) / d;
        return {p[1] + y1, p[1] + y2};
    }

    /**
     * @brief Compute the bounding box of the ellipse as {x_min, x_max, y_min, y_max}.
     */
    std::array<double, 4> bounding_box() const
    {
        double denom = determinant();
        double x_dim = std::sqrt(d / denom);
        double y_dim = std::sqrt(a / denom);
        return {-x_dim, x_dim, -y_dim, y_dim};
    }

    /**
     * @brief Return the ellipse shifted by the offset.
     */
    Ellipse offset(double offset_val) const
    {
        std::array<double, 2> p_offset = {p[0] + offset_val, p[1] + offset_val};
        return Ellipse({a, b, d}, p_offset);
    }

    /**
     * @brief Apply a grid operation G to the ellipse E as (G^T E G).
     */
    Ellipse apply_grid_op(const GridOp &grid_op) const
    {
        const auto [ga, gb, gc, gd] = grid_op.flatten();

        std::array<double, 3> D = {ga * ga * a + 2 * ga * gc * b + d * gc * gc,
                                   ga * gb * a + (ga * gd + gb * gc) * b + gc * gd * d,
                                   gb * gb * a + 2 * gb * gd * b + d * gd * gd};

        const auto [gda, gdb, gdc, gdd] = grid_op.inverse().flatten();

        std::array<double, 2> new_p = {gda * p[0] + gdb * p[1], gdc * p[0] + gdd * p[1]};
        return Ellipse(D, new_p);
    }
};

/**
 * @class EllipseState
 * @brief A class representing a state as a pair of normalized ellipses.
 *
 * Based on Definition A.1 of arXiv:1403.2975 where the pair of ellipses are represented by real
 * symmetric positive semi-definite matrices of determinant 1
 *
 * Note: the floating point calculation in this class uses doubles, which leads to limited
 * precision. For higher precision, this needs to be replaced by multiprecision floats.
 */
struct EllipseState {
    Ellipse e1;
    Ellipse e2;

    /**
     * @brief Constructor for EllipseState.
     * @param ellipse1 The first ellipse.
     * @param ellipse2 The second ellipse.
     */
    EllipseState(const Ellipse &ellipse1, const Ellipse &ellipse2) : e1(ellipse1), e2(ellipse2) {}

    /**
     * @brief Calculate the skew of the state (Eq. 34, arXiv:1403.2975).
     */
    double skew() const { return e1.b * e1.b + e2.b * e2.b; }

    /**
     * @brief Calculate the bias of the state (Eq. 34, arXiv:1403.2975).
     */
    double bias() const { return e2.z - e1.z; }

    /**
     * @brief Calculate the special grid operation for the state for reducing the skew
     * (Lemma A.5 (Step Lemma), arXiv:1403.2975).
     */
    GridOp skew_grid_op()
    {
        GridOp grid_op = GridOp::from_string("I");
        EllipseState state = *this;
        double current_skew = state.skew();
        while (current_skew >= 15.0) {
            auto result = state.reduce_skew();
            GridOp new_grid_op = result.first;
            state = result.second;
            grid_op = grid_op * new_grid_op;
            RT_FAIL_IF(state.skew() > 0.9 * current_skew, "Skew was not decreased for state");
            current_skew = state.skew();
        }
        return grid_op;
    }

    EllipseState apply_grid_op(const GridOp &grid_op) const;

    /**
     * @brief Apply a shift operator to the state. (Definition A.6 and Lemma A.8 , arXiv:1403.2975).
     */
    std::pair<EllipseState, INT_TYPE> apply_shift_op() const
    {
        double k = std::floor((1.0 - bias()) / 2.0);
        double pk_pow = std::pow(LAMBDA_D, k);
        double nk_pow = std::pow(LAMBDA_D, -k);
        Ellipse new_e1 = e1;
        Ellipse new_e2 = e2;
        new_e1.a *= pk_pow;
        new_e1.d *= nk_pow;
        new_e1.z -= k;
        new_e2.a *= nk_pow;
        new_e2.d *= pk_pow;
        new_e2.z += k;
        if (INT_TYPE(k) % 2 != 0) {
            new_e2.b *= -1.0;
        }
        return {EllipseState(new_e1, new_e2), INT_TYPE(k)};
    }

    /**
     * @brief Phase 1 for reduce_skew (Remark A.20, arXiv:1403.2975).
     */
    GridOp reduce_skew_apply_symmetry_and_coarse_bias() const
    {
        GridOp op = GridOp::from_string("I");
        double sign = 1.0;

        if (e2.b < 0) {
            op = op * GridOp::from_string("Z");
        }

        if ((e1.z + e2.z) < 0) {
            sign = -1.0;
            op = op * GridOp::from_string("X");
        }

        if (std::abs(bias()) > 2) {
            INT_TYPE n = INT_TYPE(std::round((1.0 - sign * bias()) / 4.0));
            op = op * GridOp::from_string("U").pow(n);
        }
        return op;
    }

    /**
     * @brief Phase 2 for reduce_skew
     *
     * @param state The state to be shifted (will be modified in place).
     * @return A pair containing:
     * - The GridOp applied (Z or X).
     * - The integer shift amount `k` (needed to adjust the final operator).
     */
    std::pair<GridOp, INT_TYPE> reduce_skew_apply_shift_lemma(EllipseState &state) const
    {
        INT_TYPE k = 0;
        GridOp op = GridOp::from_string("I");

        if (std::abs(state.bias()) > 1) {
            // Apply the shift defined in Lemma A.1 / A.8
            auto shift_result = state.apply_shift_op();
            state = shift_result.first;
            k = shift_result.second;

            // Re-apply symmetries if the shift disrupted the canonical form
            if (state.e2.b < 0) {
                GridOp z_op = GridOp::from_string("Z");
                state = state.apply_grid_op(z_op);
                op = op * z_op;
            }
            if ((state.e1.z + state.e2.z) < 0) {
                GridOp x_op = GridOp::from_string("X");
                state = state.apply_grid_op(x_op);
                op = op * x_op;
            }
        }
        return {op, k};
    }

    /**
     * @brief Phase 3 for reduce_skew
     *
     * Selects the specific grid operator (R, K, A, or B) to reduce the skew
     */
    static GridOp reduce_skew_select_operator(const Ellipse &e1, const Ellipse &e2)
    {
        if (-0.8 <= e1.z && e1.z <= 0.8 && -0.8 <= e2.z && e2.z <= 0.8) {
            return GridOp::from_string("R");
        }

        if (e1.b >= 0) {
            if (e1.z <= 0.3 && e2.z >= 0.8) {
                return GridOp::from_string("K");
            }
            if (e1.z >= 0.8 && e2.z <= 0.3) {
                return GridOp::from_string("K").adj2();
            }
            if (e1.z >= 0.3 && e2.z >= 0.3) {
                INT_TYPE n = static_cast<INT_TYPE>(
                    std::max(1.0, std::floor(std::pow(LAMBDA_D, std::min(e1.z, e2.z)) / 2.0)));
                return GridOp::from_string("A").pow(n);
            }
            RT_FAIL("Skew couldn't be reduced (Case A/K failed)");
        }

        if (e1.z >= -0.2 && e2.z >= -0.2) {
            INT_TYPE n = static_cast<INT_TYPE>(
                std::max(1.0, std::floor(std::pow(LAMBDA_D, std::min(e1.z, e2.z)) / M_SQRT2)));
            return GridOp::from_string("B").pow(n);
        }

        RT_FAIL("Skew couldn't be reduced (Case B failed)");
    }

    /**
     * @brief Reduces the skew of the state (Implementation of the Step Lemma).
     *
     * This method guarantees a reduction in skew for states with skew > 15.
     * It follows the algorithm in Section A.6 of arXiv:1403.2975 by:
     * 1. Applying symmetries Remark A.20.
     * 2. Applying the Shift Lemma.
     * 3. Applying the specific Geometric Lemma (R, K, A, B).
     */
    std::pair<GridOp, EllipseState> reduce_skew()
    {
        RT_FAIL_IF(!e1.positive_semi_definite() || !e2.positive_semi_definite(),
                   "Ellipse is not positive semi-definite.");

        GridOp total_op = reduce_skew_apply_symmetry_and_coarse_bias();
        EllipseState working_state = this->apply_grid_op(total_op);

        auto [shift_adj_op, k] = reduce_skew_apply_shift_lemma(working_state);

        GridOp local_op = shift_adj_op;

        GridOp geometric_op = reduce_skew_select_operator(working_state.e1, working_state.e2);

        local_op = local_op * geometric_op;

        if (k != 0) {
            local_op = local_op.apply_shift_op(k);
        }

        total_op = total_op * local_op;

        return {total_op, this->apply_grid_op(total_op)};
    }
};
// --- Implementations requiring full class definitions ---

/**
 * @brief Apply the grid operator to an ellipse (Lemma A.4, arXiv:1403.2975).
 */
inline Ellipse GridOp::apply_to_ellipse(const Ellipse &ellipse) const
{
    return ellipse.apply_grid_op(*this);
}

/**
 * @brief Apply the grid operator to a state (Lemma A.3, arXiv:1403.2975).
 */
inline EllipseState GridOp::apply_to_state(const EllipseState &state) const
{
    return state.apply_grid_op(*this);
}

/**
 * @brief Apply a grid operation G to the state (Definition A.3, arXiv:1403.2975).
 */
inline EllipseState EllipseState::apply_grid_op(const GridOp &grid_op) const
{
    return EllipseState(e1.apply_grid_op(grid_op), e2.apply_grid_op(grid_op.adj2()));
}

} // namespace RSDecomp::GridProblem
