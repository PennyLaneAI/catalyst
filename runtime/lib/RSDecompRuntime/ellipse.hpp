#pragma once

#include "rings.hpp" // Assumed to contain new FLOAT_TYPE definitions
#include "utils.hpp"
#include <array>
#include <cmath> // For std::abs(int)
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// These should NOW be included in "rings.hpp", but are placed here
// for clarity in case "rings.hpp" isn't modified.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>

// FIX 1: Move the literal namespace to the global scope, after includes

namespace GridProblem {
    
using namespace boost::multiprecision::literals;

struct GridOp;
struct Ellipse;
struct EllipseState;

/**
 * @class GridOp
 * @brief (Implementation unchanged, except for flatten())
 */
struct GridOp {
    std::array<INT_TYPE, 2> a, b, c, d;

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

    // --- CONVERTED ---
    std::vector<FLOAT_TYPE> flatten() const
    {
        // Use multiprecision constants and types
        return {FLOAT_TYPE(a[0]) + FLOAT_TYPE(a[1]) / MP_SQRT2,
                FLOAT_TYPE(b[0]) + FLOAT_TYPE(b[1]) / MP_SQRT2,
                FLOAT_TYPE(c[0]) + FLOAT_TYPE(c[1]) / MP_SQRT2,
                FLOAT_TYPE(d[0]) + FLOAT_TYPE(d[1]) / MP_SQRT2};
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
        INT_TYPE k_abs = abs_val(k); // abs_val from utils.hpp
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
        auto c = ZSqrtTwo(1, 1).pow(abs_val(n)); // abs_val from utils.hpp
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
 * @brief CONVERTED: All `double` members and functions replaced with `FLOAT_TYPE`.
 *
 * Represents an ellipse as a positive definite matrix D = [[a, b], [b, d]].
 */
struct Ellipse {
    FLOAT_TYPE a, b, d;
    std::array<FLOAT_TYPE, 2> p;
    FLOAT_TYPE z;
    FLOAT_TYPE e;

    // FIX 2: Remove redundant FLOAT_TYPE() wrapper around literals.
    Ellipse(const std::array<FLOAT_TYPE, 3> &D = {FLOAT_TYPE(1), FLOAT_TYPE(0), FLOAT_TYPE(1)},
            const std::array<FLOAT_TYPE, 2> &p_val = {FLOAT_TYPE(0), FLOAT_TYPE(0)})
        : p(p_val)
    {
        a = D[0];
        b = D[1];
        d = D[2];
        // Use boost::multiprecision functions and new constants
        // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
        z = FLOAT_TYPE("0.5") * boost::multiprecision::log2(d / a) /
            boost::multiprecision::log2(MP_LAMBDA);
        e = boost::multiprecision::sqrt(a * d);
    }

    static Ellipse from_region(FLOAT_TYPE theta, FLOAT_TYPE epsilon, INT_TYPE k = 0)
    {
        FLOAT_TYPE t = boost::multiprecision::pow(epsilon, 2) / 2;
        FLOAT_TYPE scale = boost::multiprecision::pow(FLOAT_TYPE(2), FLOAT_TYPE(k) / 2);
        FLOAT_TYPE a_val = scale * t;
        FLOAT_TYPE b_val = scale * epsilon;
        FLOAT_TYPE a2 = 1 / boost::multiprecision::pow(a_val, 2);
        FLOAT_TYPE b2 = 1 / boost::multiprecision::pow(b_val, 2);
        FLOAT_TYPE d2 = a2 - b2;
        FLOAT_TYPE zx = boost::multiprecision::cos(theta);
        FLOAT_TYPE zy = boost::multiprecision::sin(theta);
        FLOAT_TYPE new_a = d2 * zx * zx + b2;
        FLOAT_TYPE new_d = d2 * zy * zy + b2;
        FLOAT_TYPE new_b = d2 * zx * zy;
        FLOAT_TYPE const_val = 1 - t;
        std::array<FLOAT_TYPE, 2> p_val = {const_val * scale * zx, const_val * scale * zy};
        return Ellipse({new_a, new_b, new_d}, p_val);
    }

    FLOAT_TYPE discriminant() const
    {
        return boost::multiprecision::pow(a + d, 2) - 4 * (a * d - boost::multiprecision::pow(b, 2));
    }

    FLOAT_TYPE determinant() const { return a * d - boost::multiprecision::pow(b, 2); }

    bool positive_semi_definite() const
    {
        return (a + d) + boost::multiprecision::sqrt(discriminant()) >= 0;
    }

    FLOAT_TYPE uprightness() const { return MP_PI / boost::multiprecision::pow(2 * e, 2); }

    static FLOAT_TYPE b_from_uprightness(FLOAT_TYPE up)
    {
        return boost::multiprecision::sqrt(boost::multiprecision::pow(MP_PI / (4 * up), 2) - 1);
    }

    bool contains(FLOAT_TYPE x, FLOAT_TYPE y) const
    {
        FLOAT_TYPE x_ = x - p[0];
        FLOAT_TYPE y_ = y - p[1];
        return (a * boost::multiprecision::pow(x_, 2) + 2 * b * x_ * y_ +
                d * boost::multiprecision::pow(y_, 2)) <= 1;
    }

    std::pair<Ellipse, FLOAT_TYPE> normalize() const
    {
        FLOAT_TYPE s_val = 1 / boost::multiprecision::sqrt(determinant());
        return {scale(s_val), s_val};
    }

    Ellipse scale(FLOAT_TYPE scale_factor) const
    {
        std::array<FLOAT_TYPE, 3> D = {a * scale_factor, b * scale_factor, d * scale_factor};
        return Ellipse(D, p);
    }

    std::pair<FLOAT_TYPE, FLOAT_TYPE> x_points(FLOAT_TYPE y) const
    {
        FLOAT_TYPE y_shifted = y - p[1];
        FLOAT_TYPE disc =
            boost::multiprecision::pow(y_shifted, 2) * (boost::multiprecision::pow(b, 2) - a * d) +
            a;
        if (disc < 0) {
            throw std::runtime_error("Point y is outside the ellipse");
        }
        FLOAT_TYPE d0 = boost::multiprecision::sqrt(disc);
        FLOAT_TYPE x1 = (-b * y_shifted - d0) / a;
        FLOAT_TYPE x2 = (-b * y_shifted + d0) / a;
        return {p[0] + x1, p[0] + x2};
    }

    std::pair<FLOAT_TYPE, FLOAT_TYPE> y_points(FLOAT_TYPE x) const
    {
        FLOAT_TYPE x_shifted = x - p[0];
        FLOAT_TYPE disc =
            boost::multiprecision::pow(b * x_shifted, 2) - d * (a * boost::multiprecision::pow(x_shifted, 2) - 1);
        if (disc < 0) {
            throw std::runtime_error("Point x is outside the ellipse");
        }
        FLOAT_TYPE d0 = boost::multiprecision::sqrt(disc);
        FLOAT_TYPE y1 = (-b * x_shifted - d0) / d;
        FLOAT_TYPE y2 = (-b * x_shifted + d0) / d;
        return {p[1] + y1, p[1] + y2};
    }

    std::array<FLOAT_TYPE, 4> bounding_box() const
    {
        FLOAT_TYPE denom = determinant();
        FLOAT_TYPE x_dim = boost::multiprecision::sqrt(d / denom);
        FLOAT_TYPE y_dim = boost::multiprecision::sqrt(a / denom);
        return {-x_dim, x_dim, -y_dim, y_dim};
    }

    Ellipse offset(FLOAT_TYPE offset_val) const
    {
        std::array<FLOAT_TYPE, 2> p_offset = {p[0] + offset_val, p[1] + offset_val};
        return Ellipse({a, b, d}, p_offset);
    }

    Ellipse apply_grid_op(const GridOp &grid_op) const
    {
        auto g = grid_op.flatten(); // This now returns std::vector<FLOAT_TYPE>
        FLOAT_TYPE ga = g[0], gb = g[1], gc = g[2], gd = g[3];
        std::array<FLOAT_TYPE, 3> D = {ga * ga * a + 2 * ga * gc * b + d * gc * gc,
                                       ga * gb * a + (ga * gd + gb * gc) * b + gc * gd * d,
                                       gb * gb * a + 2 * gb * gd * b + d * gd * gd};
        auto g_inv = grid_op.inverse().flatten();
        FLOAT_TYPE gda = g_inv[0], gdb = g_inv[1], gdc = g_inv[2], gdd = g_inv[3];
        std::array<FLOAT_TYPE, 2> new_p = {gda * p[0] + gdb * p[1], gdc * p[0] + gdd * p[1]};
        return Ellipse(D, new_p);
    }

    bool operator==(const Ellipse &other) const
    {
        return a == other.a && b == other.b && d == other.d && p == other.p;
    }
};

/**
 * @class EllipseState
 * @brief CONVERTED: All `double` members and functions replaced with `FLOAT_TYPE`.
 */
struct EllipseState {
    Ellipse e1;
    Ellipse e2;

    EllipseState(const Ellipse &ellipse1, const Ellipse &ellipse2) : e1(ellipse1), e2(ellipse2) {}

    FLOAT_TYPE skew() const { return boost::multiprecision::pow(e1.b, 2) + boost::multiprecision::pow(e2.b, 2); }

    FLOAT_TYPE bias() const { return e2.z - e1.z; }

    GridOp skew_grid_op()
    {
        GridOp grid_op = GridOp::from_string("I");
        EllipseState state = *this;
        FLOAT_TYPE current_skew = state.skew();
        while (current_skew >= 15) { // 15.0 -> 15
            std::cout << "applied! current skew: " << current_skew << std::endl;
            auto result = state.reduce_skew();
            GridOp new_grid_op = result.first;
            state = result.second;
            grid_op = grid_op * new_grid_op;
            // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
            if (state.skew() > FLOAT_TYPE("0.9") * current_skew) {
                throw std::runtime_error("Skew was not decreased for state");
            }
            current_skew = state.skew();
        }
        return grid_op;
    }

    EllipseState apply_grid_op(const GridOp &grid_op) const;

    std::pair<EllipseState, INT_TYPE> apply_shift_op() const
    {
        // Use boost::multiprecision::floor and convert to long long (or int)
        INT_TYPE k = static_cast<INT_TYPE>(
            boost::multiprecision::floor((1 - bias()) / 2).convert_to<long long>());
        FLOAT_TYPE pk_pow = boost::multiprecision::pow(MP_LAMBDA, static_cast<long long>(k));
        FLOAT_TYPE nk_pow = boost::multiprecision::pow(MP_LAMBDA, -static_cast<long long>(k));
        Ellipse new_e1 = e1;
        Ellipse new_e2 = e2;
        new_e1.a *= pk_pow;
        new_e1.d *= nk_pow;
        // FIX 3: k is __int128, use static_cast, not .convert_to()
        // Also, cast to FLOAT_TYPE, not double, since .z is FLOAT_TYPE
        new_e1.z -= static_cast<FLOAT_TYPE>(k);
        new_e2.a *= nk_pow;
        new_e2.d *= pk_pow;
        // FIX 3 (cont.):
        new_e2.z += static_cast<FLOAT_TYPE>(k);
        new_e2.b *= boost::multiprecision::pow(FLOAT_TYPE(-1), static_cast<long long>(k));
        return {EllipseState(new_e1, new_e2), k};
    }

    std::pair<GridOp, EllipseState> reduce_skew()
    {
        if (!e1.positive_semi_definite() || !e2.positive_semi_definite()) {
            throw std::runtime_error("Ellipse is not positive semi-definite");
        }
        INT_TYPE sign = 1, k = 0;
        GridOp grid_op = GridOp::from_string("I");
        if (e2.b < 0) {
            grid_op = grid_op * GridOp::from_string("Z");
        }
        if ((e1.z + e2.z) < 0) {
            sign *= -1;
            grid_op = grid_op * GridOp::from_string("X");
        }
        if (boost::multiprecision::abs(bias()) > 2) {
            // FIX 4: Use boost::multiprecision::round for FLOAT_TYPE
            INT_TYPE n = static_cast<INT_TYPE>(
                boost::multiprecision::round((1 - sign * bias()) / 4).convert_to<long long>());
            grid_op = grid_op * GridOp::from_string("U").pow(n);
        }
        GridOp n_grid_op = GridOp::from_string("I");
        EllipseState new_state = this->apply_grid_op(grid_op);
        if (boost::multiprecision::abs(new_state.bias()) > 1) {
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
        // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
        if (-FLOAT_TYPE("0.8") <= current_e1.z && current_e1.z <= FLOAT_TYPE("0.8") &&
            -FLOAT_TYPE("0.8") <= current_e2.z && current_e2.z <= FLOAT_TYPE("0.8")) {
            n_grid_op = n_grid_op * GridOp::from_string("R");
        }
        else {
            if (current_e1.b >= 0) {
                // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
                if (current_e1.z <= FLOAT_TYPE("0.3") && current_e2.z >= FLOAT_TYPE("0.8")) {
                    n_grid_op = n_grid_op * GridOp::from_string("K");
                }
                // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
                else if (current_e1.z >= FLOAT_TYPE("0.8") && current_e2.z <= FLOAT_TYPE("0.3")) {
                    n_grid_op = n_grid_op * GridOp::from_string("K").adj2();
                }
                // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
                else if (current_e1.z >= FLOAT_TYPE("0.3") && current_e2.z >=FLOAT_TYPE("0.3")) {
                    INT_TYPE n = static_cast<INT_TYPE>(
                        boost::multiprecision::max(
                            FLOAT_TYPE(1),
                            boost::multiprecision::floor(
                                boost::multiprecision::pow(
                                    MP_LAMBDA,
                                    boost::multiprecision::min(current_e1.z, current_e2.z)) /
                                2))
                            .convert_to<long long>());
                    n_grid_op = n_grid_op * GridOp::from_string("A").pow(n);
                }
                else {
                    throw std::runtime_error("Skew couldn't be reduced for the state");
                }
            }
            else {
                // FIX 2 (cont.): Remove redundant FLOAT_TYPE() wrapper
                if (current_e1.z >= -FLOAT_TYPE("0.2") && current_e2.z >= -FLOAT_TYPE("0.2")) {
                    INT_TYPE n = static_cast<INT_TYPE>(
                        boost::multiprecision::max(
                            FLOAT_TYPE(1),
                            boost::multiprecision::floor(
                                boost::multiprecision::pow(
                                    MP_LAMBDA,
                                    boost::multiprecision::min(current_e1.z, current_e2.z)) /
                                MP_SQRT2))
                            .convert_to<long long>());
                    n_grid_op = n_grid_op * GridOp::from_string("B").pow(n);
                }
                else {
                    throw std::runtime_error("Skew couldn't be reduced for the state");
                }
            }
        }
        if (k != 0) {
            n_grid_op = n_grid_op.apply_shift_op(k);
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

} // namespace GridProblem
