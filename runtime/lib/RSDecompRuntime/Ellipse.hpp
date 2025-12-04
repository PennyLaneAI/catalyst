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

#include "Rings.hpp"
#include "Utils.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace RSDecomp::GridProblem {

using namespace RSDecomp::Rings;
using namespace RSDecomp::Utils;
struct GridOp;
struct Ellipse;
struct EllipseState;

/**
 * @class GridOp
 * @brief C++ translation of the Python GridOp class.
 *
 * Represents a grid operation on a 2D grid as a 2x2 matrix.
 */
struct GridOp {
    std::array<INT_TYPE, 2> a, b, c, d;

    /**
     * @brief Constructor for GridOp.
     * @param a_val The a-coefficient as {a0, a1}.
     * @param b_val The b-coefficient as {b0, b1}.
     * @param c_val The c-coefficient as {c0, c1}.
     * @param d_val The d-coefficient as {d0, d1}.
     * @param check_valid If true, validates the grid operation properties.
     */
    GridOp(const std::array<INT_TYPE, 2> &a_val, const std::array<INT_TYPE, 2> &b_val,
           const std::array<INT_TYPE, 2> &c_val, const std::array<INT_TYPE, 2> &d_val,
           bool check_valid = true)
        : a(a_val), b(b_val), c(c_val), d(d_val)
    {
        if (check_valid) {
            if ((a[0] + b[0] + c[0] + d[0]) % 2 != 0) {
                throw std::invalid_argument("sum of a_0, b_0, c_0, d_0 must be even");
            }
            if (!(((a[1] & 1) == (b[1] & 1)) && ((b[1] & 1) == (c[1] & 1)) &&
                  ((c[1] & 1) == (d[1] & 1)))) {
                throw std::invalid_argument("a_1, b_1, c_1, d_1 must have same parity");
            }
        }
    }

    // Default constructor for map initialization
    GridOp() : a({1, 0}), b({0, 0}), c({0, 0}), d({1, 0}) {}

    static GridOp from_string(const std::string &s);

    std::array<INT_TYPE, 2> determinant() const
    {
        return {a[0] * d[0] - b[0] * c[0] + (a[1] * d[1] - b[1] * c[1]) / 2,
                a[0] * d[1] - b[0] * c[1] + a[1] * d[0] - b[1] * c[0]};
    }

    bool is_special() const
    {
        auto det = determinant();
        return det[1] == 0 && (det[0] == 1 || det[0] == -1);
    }

    std::vector<double> flatten() const
    {
        return {static_cast<double>(a[0]) + static_cast<double>(a[1]) / M_SQRT2,
                static_cast<double>(b[0]) + static_cast<double>(b[1]) / M_SQRT2,
                static_cast<double>(c[0]) + static_cast<double>(c[1]) / M_SQRT2,
                static_cast<double>(d[0]) + static_cast<double>(d[1]) / M_SQRT2};
    }

    GridOp inverse() const
    {
        auto det = determinant();
        INT_TYPE det1 = det[0];
        if (det[1] == 0 && (det1 == 1 || det1 == -1)) {
            return GridOp({d[0] / det1, d[1] / det1}, {-b[0] / det1, -b[1] / det1},
                          {-c[0] / det1, -c[1] / det1}, {a[0] / det1, a[1] / det1});
        }
        throw std::runtime_error("Grid operator needs to be special to have an inverse.");
    }

    GridOp transpose() const { return GridOp(d, b, c, a); }

    GridOp adj2() const
    {
        return GridOp({a[0], -a[1]}, {b[0], -b[1]}, {c[0], -c[1]}, {d[0], -d[1]});
    }

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

    bool operator==(const GridOp &other) const
    {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }
};

inline const std::map<std::string, GridOp> &_useful_grid_ops()
{
    static std::map<std::string, GridOp> ops;
    if (ops.empty()) {
        ops["I"] = GridOp({1, 0}, {0, 0}, {0, 0}, {1, 0});
        ops["A"] = GridOp({1, 0}, {-2, 0}, {0, 0}, {1, 0});
        ops["B"] = GridOp({1, 0}, {0, 2}, {0, 0}, {1, 0});
        ops["K"] = GridOp({-1, 1}, {0, -1}, {1, 1}, {0, 1});
        ops["R"] = GridOp({0, 1}, {0, -1}, {0, 1}, {0, 1});
        ops["U"] = GridOp({1, 2}, {0, 0}, {0, 0}, {-1, 2});
        ops["X"] = GridOp({0, 0}, {1, 0}, {1, 0}, {0, 0});
        ops["Z"] = GridOp({1, 0}, {0, 0}, {0, 0}, {-1, 0});
    }
    return ops;
}

inline GridOp GridOp::from_string(const std::string &s)
{
    const auto &ops = _useful_grid_ops();
    auto it = ops.find(s);
    if (it != ops.end()) {
        return it->second;
    }
    return ops.at("I"); // Default to identity
}

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
 * @brief C++ translation of the Python Ellipse class.
 *
 * Represents an ellipse as a positive definite matrix D = [[a, b], [b, d]].
 */
struct Ellipse {
    double a, b, d;
    std::array<double, 2> p;
    double z;
    double e;

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

    static Ellipse from_region(double theta, double epsilon, int k = 0)
    {
        double t = std::pow(epsilon, 2) / 2.0;
        double scale = std::pow(2.0, static_cast<double>(k) / 2.0);
        double a_val = scale * t;
        double b_val = scale * epsilon;
        double a2 = 1.0 / std::pow(a_val, 2);
        double b2 = 1.0 / std::pow(b_val, 2);
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

    double discriminant() const { return std::pow(a + d, 2) - 4 * (a * d - std::pow(b, 2)); }

    double determinant() const { return a * d - std::pow(b, 2); }

    bool positive_semi_definite() const { return (a + d) + std::sqrt(discriminant()) >= 0; }

    double uprightness() const { return M_PI / std::pow(2.0 * e, 2); }

    static double b_from_uprightness(double up)
    {
        return std::sqrt(std::pow(M_PI / (4.0 * up), 2) - 1.0);
    }

    bool contains(double x, double y) const
    {
        double x_ = x - p[0];
        double y_ = y - p[1];
        return (a * std::pow(x_, 2) + 2 * b * x_ * y_ + d * std::pow(y_, 2)) <= 1.0;
    }

    std::pair<Ellipse, double> normalize() const
    {
        double s_val = 1.0 / std::sqrt(determinant());
        return {scale(s_val), s_val};
    }

    Ellipse scale(double scale_factor) const
    {
        std::array<double, 3> D = {a * scale_factor, b * scale_factor, d * scale_factor};
        return Ellipse(D, p);
    }

    std::pair<double, double> x_points(double y) const
    {
        double y_shifted = y - p[1];
        double disc = std::pow(y_shifted, 2) * (std::pow(b, 2) - a * d) + a;
        if (disc < 0) {
            throw std::runtime_error("Point y is outside the ellipse");
        }
        double d0 = std::sqrt(disc);
        double x1 = (-b * y_shifted - d0) / a;
        double x2 = (-b * y_shifted + d0) / a;
        return {p[0] + x1, p[0] + x2};
    }

    std::pair<double, double> y_points(double x) const
    {
        double x_shifted = x - p[0];
        double disc = std::pow(b * x_shifted, 2) - d * (a * std::pow(x_shifted, 2) - 1.0);
        if (disc < 0) {
            throw std::runtime_error("Point x is outside the ellipse");
        }
        double d0 = std::sqrt(disc);
        double y1 = (-b * x_shifted - d0) / d;
        double y2 = (-b * x_shifted + d0) / d;
        return {p[1] + y1, p[1] + y2};
    }

    std::array<double, 4> bounding_box() const
    {
        double denom = determinant();
        double x_dim = std::sqrt(d / denom);
        double y_dim = std::sqrt(a / denom);
        return {-x_dim, x_dim, -y_dim, y_dim};
    }

    Ellipse offset(double offset_val) const
    {
        std::array<double, 2> p_offset = {p[0] + offset_val, p[1] + offset_val};
        return Ellipse({a, b, d}, p_offset);
    }

    Ellipse apply_grid_op(const GridOp &grid_op) const
    {
        auto g = grid_op.flatten();
        double ga = g[0], gb = g[1], gc = g[2], gd = g[3];
        std::array<double, 3> D = {ga * ga * a + 2 * ga * gc * b + d * gc * gc,
                                   ga * gb * a + (ga * gd + gb * gc) * b + gc * gd * d,
                                   gb * gb * a + 2 * gb * gd * b + d * gd * gd};
        auto g_inv = grid_op.inverse().flatten();
        double gda = g_inv[0], gdb = g_inv[1], gdc = g_inv[2], gdd = g_inv[3];
        std::array<double, 2> new_p = {gda * p[0] + gdb * p[1], gdc * p[0] + gdd * p[1]};
        return Ellipse(D, new_p);
    }

    bool operator==(const Ellipse &other) const
    {
        return a == other.a && b == other.b && d == other.d && p == other.p;
    }
};

/**
 * @class EllipseState
 * @brief C++ translation of the Python EllipseState class.
 */
struct EllipseState {
    Ellipse e1;
    Ellipse e2;

    EllipseState(const Ellipse &ellipse1, const Ellipse &ellipse2) : e1(ellipse1), e2(ellipse2) {}

    double skew() const { return std::pow(e1.b, 2) + std::pow(e2.b, 2); }

    double bias() const { return e2.z - e1.z; }

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
            if (state.skew() > 0.9 * current_skew) {
                throw std::runtime_error("Skew was not decreased for state");
            }
            current_skew = state.skew();
        }
        return grid_op;
    }

    EllipseState apply_grid_op(const GridOp &grid_op) const;

    std::pair<EllipseState, long long> apply_shift_op() const
    {
        long long k = static_cast<long long>(std::floor((1.0 - bias()) / 2.0));
        double pk_pow = std::pow(LAMBDA_D, static_cast<double>(k));
        double nk_pow = std::pow(LAMBDA_D, static_cast<double>(-k));
        Ellipse new_e1 = e1;
        Ellipse new_e2 = e2;
        new_e1.a *= pk_pow;
        new_e1.d *= nk_pow;
        new_e1.z -= static_cast<double>(k);
        new_e2.a *= nk_pow;
        new_e2.d *= pk_pow;
        new_e2.z += static_cast<double>(k);
        new_e2.b *= std::pow(-1.0, static_cast<double>(k));
        return {EllipseState(new_e1, new_e2), k};
    }

    std::pair<GridOp, EllipseState> reduce_skew()
    {
        if (!e1.positive_semi_definite() || !e2.positive_semi_definite()) {
            throw std::runtime_error("Ellipse is not positive semi-definite");
        }
        long long sign = 1;
        long long k = 0;
        GridOp grid_op = GridOp::from_string("I");
        if (e2.b < 0) {
            grid_op = grid_op * GridOp::from_string("Z");
        }
        if ((e1.z + e2.z) < 0) {
            sign *= -1;
            grid_op = grid_op * GridOp::from_string("X");
        }
        if (std::abs(bias()) > 2) {
            long long n =
                static_cast<long long>(round((1.0 - static_cast<double>(sign) * bias()) / 4.0));
            grid_op = grid_op * GridOp::from_string("U").pow(INT_TYPE(n));
        }
        GridOp n_grid_op = GridOp::from_string("I");
        EllipseState new_state = this->apply_grid_op(grid_op);
        if (std::abs(new_state.bias()) > 1) {
            auto shift_result = new_state.apply_shift_op();
            new_state = shift_result.first;
            k = shift_result.second;
            if (new_state.e2.b < 0) {
                GridOp grid_op_z = GridOp::from_string("Z");
                new_state = new_state.apply_grid_op(grid_op_z);
                n_grid_op = n_grid_op * grid_op_z;
            }
            if ((new_state.e1.z + new_state.e2.z) < 0) {
                GridOp grid_op_x = GridOp::from_string("X");
                new_state = new_state.apply_grid_op(grid_op_x);
                n_grid_op = n_grid_op * grid_op_x;
            }
        }
        Ellipse current_e1 = new_state.e1;
        Ellipse current_e2 = new_state.e2;
        if (-0.8 <= current_e1.z && current_e1.z <= 0.8 && -0.8 <= current_e2.z &&
            current_e2.z <= 0.8) {
            n_grid_op = n_grid_op * GridOp::from_string("R");
        }
        else {
            if (current_e1.b >= 0) {
                if (current_e1.z <= 0.3 && current_e2.z >= 0.8) {
                    n_grid_op = n_grid_op * GridOp::from_string("K");
                }
                else if (current_e1.z >= 0.8 && current_e2.z <= 0.3) {
                    n_grid_op = n_grid_op * GridOp::from_string("K").adj2();
                }
                else if (current_e1.z >= 0.3 && current_e2.z >= 0.3) {
                    INT_TYPE n = static_cast<INT_TYPE>(
                        max(1.0,
                            std::floor(std::pow(LAMBDA_D, min(current_e1.z, current_e2.z)) / 2.0)));
                    n_grid_op = n_grid_op * GridOp::from_string("A").pow(n);
                }
                else {
                    throw std::runtime_error("Skew couldn't be reduced for the state");
                }
            }
            else {
                if (current_e1.z >= -0.2 && current_e2.z >= -0.2) {
                    INT_TYPE n = static_cast<INT_TYPE>(max(
                        1.0,
                        std::floor(std::pow(LAMBDA_D, min(current_e1.z, current_e2.z)) / M_SQRT2)));
                    n_grid_op = n_grid_op * GridOp::from_string("B").pow(n);
                }
                else {
                    throw std::runtime_error("Skew couldn't be reduced for the state");
                }
            }
        }
        if (k != 0) {
            n_grid_op = n_grid_op.apply_shift_op(INT_TYPE(k));
        }
        grid_op = grid_op * n_grid_op;
        return {grid_op, this->apply_grid_op(grid_op)};
    }
};

// --- Implementations requiring full class definitions ---

inline Ellipse GridOp::apply_to_ellipse(const Ellipse &ellipse) const
{
    return ellipse.apply_grid_op(*this);
}

inline EllipseState GridOp::apply_to_state(const EllipseState &state) const
{
    return state.apply_grid_op(*this);
}

inline EllipseState EllipseState::apply_grid_op(const GridOp &grid_op) const
{
    return EllipseState(e1.apply_grid_op(grid_op), e2.apply_grid_op(grid_op.adj2()));
}

} // namespace RSDecomp::GridProblem
