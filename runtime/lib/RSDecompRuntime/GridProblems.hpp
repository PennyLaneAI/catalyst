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

#include <cmath>
#include <utility>

#include "Ellipse.hpp"

// Note:
// The grid problems solved here are described in https://arxiv.org/pdf/1403.2975 Section 5 and
// https://arxiv.org/pdf/1212.6253 Section 4.
//
// Solutions are passed to a visitor `visit(solution) -> bool`, and the enumeration stops as soon
// as the visitor returns false. The enumerating functions return false if they were stopped.

namespace RSDecomp::GridProblem {

/**
 * @brief Widen [lo, hi] by the working tolerance so that rounding does not drop grid points.
 */
inline void fatten(Real &lo, Real &hi) {
    Real slack = (abs(lo) + abs(hi) + 1) * tolerance();
    lo -= slack;
    hi += slack;
}

/**
 * @brief Enumerate all α ∈ Z[√2] with α ∈ [x0, x1] and α• ∈ [y0, y1], where • is the
 * √2-conjugation, up to points within the working tolerance of the boundary.
 *
 * The problem is rescaled by a power of λ = 1 + √2 such that both intervals have comparable
 * widths, and shifted close to the origin, before enumerating α = a + b√2 directly.
 * This is based on Lemmas 16 and 17 of arXiv:1212.6253.
 */
template <typename Visit>
bool solve_one_dim_problem(Real x0, Real x1, Real y0, Real y1, Visit &&visit) {
    fatten(x0, x1);
    fatten(y0, y1);
    if (x1 < x0 || y1 < y0) {
        return true;
    }

    const Real &sqrt2 = Reals::sqrt2();

    // Solutions β = λ^n α satisfy β ∈ λ^n [x0, x1] and β• = (-1)^n λ^{-n} α• ∈ (-1)^n λ^{-n} [y0,
    // y1]. Any n gives all solutions, n only balances the widths, so double precision suffices.
    long n = std::lround((approx_log(Real(y1 - y0)) - approx_log(Real(x1 - x0))) /
                         (2 * std::log(1 + std::sqrt(2.0))));
    Real lambda_n = pow(lambda(), n);
    Real conj_scale = (n % 2 == 0 ? 1 : -1) / lambda_n;
    x0 *= lambda_n;
    x1 *= lambda_n;
    y0 *= conj_scale;
    y1 *= conj_scale;
    if (y1 < y0) {
        std::swap(y0, y1);
    }

    // Shift by a nearby β0 ∈ Z[√2] so that the enumerated coordinates stay small.
    Real cx = (x0 + x1) / 2;
    Real cy = (y0 + y1) / 2;
    ZSqrtTwo beta0(floor_int(Real((cx + cy) / 2)), floor_int(Real((cx - cy) / (2 * sqrt2))));
    Real shift = to_real(beta0);
    Real shift_conj = to_real(beta0.adj2());
    x0 -= shift;
    x1 -= shift;
    y0 -= shift_conj;
    y1 -= shift_conj;

    ZSqrtTwo unscale = n >= 0 ? ZSqrtTwo(-1, 1).pow(n) : ZSqrtTwo(1, 1).pow(-n);

    // β - β0 = a + b√2 with a ∈ [(x0 + y0) / 2, (x1 + y1) / 2], and b bounded by both intervals.
    INT_TYPE a_max = floor_int(Real((x1 + y1) / 2));
    for (INT_TYPE a = ceil_int(Real((x0 + y0) / 2)); a <= a_max; ++a) {
        Real a_real(a);
        Real b_lo = max(Real(x0 - a_real), Real(a_real - y1)) / sqrt2;
        Real b_hi = min(Real(x1 - a_real), Real(a_real - y0)) / sqrt2;
        INT_TYPE b_max = floor_int(b_hi);
        for (INT_TYPE b = ceil_int(b_lo); b <= b_max; ++b) {
            if (!visit((ZSqrtTwo(a, b) + beta0) * unscale)) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Enumerate the pairs (α, β) ∈ Z[√2]^2 such that α + iβ ∈ s1 and α• + iβ• ∈ s2.
 *
 * α runs over the 1D grid problem given by the x-ranges of both sets, and for each α, β runs over
 * the 1D grid problem given by the vertical sections of the sets at α and α•. The sets provide
 * x_range() and y_section(x), as Ellipse does.
 */
template <typename Set1, typename Set2, typename Visit>
bool solve_by_sections(const Set1 &s1, const Set2 &s2, Visit &&visit) {
    auto [x0, x1] = s1.x_range();
    auto [y0, y1] = s2.x_range();
    return solve_one_dim_problem(x0, x1, y0, y1, [&](const ZSqrtTwo &alpha) {
        auto section1 = s1.y_section(alpha);
        if (!section1) {
            return true;
        }
        auto section2 = s2.y_section(alpha.adj2());
        if (!section2) {
            return true;
        }
        return solve_one_dim_problem(section1->first, section1->second, section2->first,
                                     section2->second,
                                     [&](const ZSqrtTwo &beta) { return visit(alpha, beta); });
    });
}

/**
 * @brief Enumerate the candidates u = α + iβ + shift ∈ Z[ω], with α, β ∈ Z[√2], such that
 * u - shift ∈ s1 and (u - shift)• ∈ s2, for upright sets s1 and s2.
 *
 * The outer 1D problem is taken along the axis where the product of the widths of both sets is
 * smaller. Otherwise, most of its solutions can have no solutions of the inner 1D problem.
 * The sets provide x_range(), y_range(), y_section(x) and transpose(), as Ellipse does.
 */
template <typename Set1, typename Set2, typename Visit>
bool solve_upright_coset(const Set1 &s1, const Set2 &s2, const ZOmega &shift, Visit &&visit) {
    auto [x0, x1] = s1.x_range();
    auto [x2, x3] = s2.x_range();
    auto [y0, y1] = s1.y_range();
    auto [y2, y3] = s2.y_range();
    if ((x1 - x0) * (x3 - x2) <= (y1 - y0) * (y3 - y2)) {
        return solve_by_sections(s1, s2, [&](const ZSqrtTwo &alpha, const ZSqrtTwo &beta) {
            return visit(zomega_from_sqrt_pair(alpha, beta, shift));
        });
    }
    return solve_by_sections(s1.transpose(), s2.transpose(),
                             [&](const ZSqrtTwo &beta, const ZSqrtTwo &alpha) {
                                 return visit(zomega_from_sqrt_pair(alpha, beta, shift));
                             });
}

/**
 * @brief Enumerate candidates u ∈ Z[ω] with u ∈ state.e1 and u• ∈ state.e2 for an upright state.
 *
 * Z[ω] is the union of the cosets Z[√2] + iZ[√2] and Z[√2] + iZ[√2] + ω, where ω = (1 + i)/√2
 * and ω• = -ω. Candidates can lie slightly outside the ellipses and must be checked by the caller.
 * This is based on Proposition 5.21 and Theorem 5.18 of arXiv:1403.2975.
 */
template <typename Visit> bool solve_two_dim_problem(const EllipseState &state, Visit &&visit) {
    if (!solve_upright_coset(state.e1, state.e2, ZOmega(0), visit)) {
        return false;
    }
    const Real &h = inv_sqrt2();
    return solve_upright_coset(state.e1.offset(-h), state.e2.offset(h), ZOmega(0, 0, 1, 0), visit);
}

/**
 * @brief The set of points v with G (v + shift) ∈ {u : |u|^2 <= 2^k, n · u >= min_dot} for a grid
 * operator G, with shift = (σ, σ) / √2 and an optional half-plane.
 *
 * Sections are computed from the exact intersection of the line with the set, and the ranges are
 * taken from a bounding ellipse. Compared to sections of the bounding ellipse, this avoids
 * enumerating the many grid points that can lie in the ellipse but outside the set, e.g. for
 * angles close to a multiple of π/4.
 */
struct PulledBackRegion {
    Ellipse bound;
    std::array<ZSqrtTwo, 4> g; // √2 G, which has entries in Z[√2]
    int sigma;
    int k;
    std::optional<std::array<Real, 3>> half_plane; // {n_x, n_y, min_dot}

    /**
     * @brief The entries of √2 G for the grid operator G.
     */
    static std::array<ZSqrtTwo, 4> scaled_entries(const GridOp &op) {
        return {ZSqrtTwo(op.a[1], op.a[0]), ZSqrtTwo(op.b[1], op.b[0]), ZSqrtTwo(op.c[1], op.c[0]),
                ZSqrtTwo(op.d[1], op.d[0])};
    }

    std::pair<Real, Real> x_range() const { return bound.x_range(); }
    std::pair<Real, Real> y_range() const { return bound.y_range(); }

    /**
     * @brief Return the set reflected along the diagonal, which swaps the x and y axes.
     */
    PulledBackRegion transpose() const {
        return {bound.transpose(), {g[1], g[0], g[3], g[2]}, sigma, k, half_plane};
    }

    /**
     * @brief The range of y-values of the set along the vertical line at x, if it intersects.
     */
    std::optional<std::pair<Real, Real>> y_section(const ZSqrtTwo &x) const {
        auto bound_section = bound.y_section(x);
        if (!bound_section) {
            return std::nullopt;
        }
        // Parametrize the line by y = y_ref + y' with an integer y_ref close to the set, as the
        // terms below cancel to a precision relative to |G (v + shift)| at y' = 0.
        INT_TYPE y_ref = round_int(Real((bound_section->first + bound_section->second) / 2));

        // Along the line, G (v + shift) = (U + y' * √2 W) / 2 with U = √2 G (√2 x + σ, √2 y_ref +
        // σ) and W = √2 G (0, 1) exactly in Z[√2].
        auto times_sqrt2 = [](const ZSqrtTwo &z) { return ZSqrtTwo(2 * z.b, z.a); };
        ZSqrtTwo x_scaled = times_sqrt2(x) + ZSqrtTwo(sigma, 0);
        ZSqrtTwo y_scaled(sigma, y_ref);
        ZSqrtTwo ux = g[0] * x_scaled + g[1] * y_scaled;
        ZSqrtTwo uy = g[2] * x_scaled + g[3] * y_scaled;
        const ZSqrtTwo &wx = g[1];
        const ZSqrtTwo &wy = g[3];

        // |G (v + shift)|^2 <= 2^k is A y^2 + 2 B y + C <= 0.
        ZSqrtTwo qa = (wx * wx + wy * wy) * 2;
        ZSqrtTwo qb = times_sqrt2(ux * wx + uy * wy);
        ZSqrtTwo qc = ux * ux + uy * uy - ZSqrtTwo(INT_TYPE(1) << (k + 2), 0);
        ZSqrtTwo discriminant = qb * qb - qa * qc;
        if (discriminant.sign() < 0) {
            return std::nullopt;
        }
        // Roots without cancellation.
        Real b = to_real(qb);
        Real root = sqrt(to_real(discriminant));
        Real q = b >= 0 ? Real(-b - root) : Real(-b + root);
        Real offset(y_ref);
        if (q == 0) {
            return std::make_pair(offset, offset);
        }
        Real lo = q / to_real(qa);
        Real hi = to_real(qc) / q;
        if (hi < lo) {
            std::swap(lo, hi);
        }

        if (half_plane) {
            const auto &[nx, ny, min_dot] = *half_plane;
            Real slope = nx * to_real(wx) + ny * to_real(wy);
            Real rhs = 2 * min_dot - (nx * to_real(ux) + ny * to_real(uy));
            slope *= sqrt2();
            if (slope > 0) {
                lo = max(lo, Real(rhs / slope));
            } else if (slope < 0) {
                hi = min(hi, Real(rhs / slope));
            } else if (rhs > 0) {
                return std::nullopt;
            }
        }
        if (hi < lo) {
            return std::nullopt;
        }
        return std::make_pair(Real(lo + offset), Real(hi + offset));
    }
};

/**
 * @class RzGridSearch
 * @brief Enumerates the solutions u ∈ Z[ω] of the scaled grid problem for approximating
 * z = e^{i*theta} with precision epsilon (Section 7.2, arXiv:1403.2975).
 *
 * At level k, the solutions are the u such that u / √2^k lies in the epsilon-region
 * {|w| <= 1, Re(w * conj(z)) >= 1 - ε^2 / 2} and u• / √2^k lies in the unit disk. Both sets dilate
 * uniformly with k, so the grid operator making them upright is computed once.
 *
 * Every visited solution satisfies |u|^2 <= 2^k and |u•|^2 <= 2^k exactly in Z[√2], and is not
 * divisible by √2, so that it is visited only at its smallest level.
 */
class RzGridSearch {
  public:
    RzGridSearch(const Real &theta, const Real &epsilon)
        : zx(cos(theta)), zy(sin(theta)), target(1 - epsilon * epsilon / 2) {
        Ellipse region = Ellipse::from_region(theta, epsilon);
        Ellipse disk;
        grid_op = EllipseState(region.normalize().first, disk).skew_grid_op();
        upright = EllipseState(region, disk).apply_grid_op(grid_op);
        g = PulledBackRegion::scaled_entries(grid_op);
        g_conj = PulledBackRegion::scaled_entries(grid_op.adj2());
    }

    /**
     * @brief Visit the solutions at level k as visit(u) -> bool.
     */
    template <typename Visit> bool visit_level(int k, Visit &&visit) const {
        Real scale = pow(sqrt2(), k);
        Ellipse e1 = upright.e1.dilate(scale);
        Ellipse e2 = upright.e2.dilate(scale);
        ZSqrtTwo two_pow_k(INT_TYPE(1) << k, 0);
        std::array<Real, 3> half_plane = {zx, zy, Real(target * scale)};

        auto filter = [&](const ZOmega &v) {
            // u = G v is divisible by √2 exactly if v is.
            if ((v.a + v.c) % 2 == 0 && (v.b + v.d) % 2 == 0) {
                return true;
            }
            ZOmega u = grid_op * v;
            ZSqrtTwo xi = two_pow_k - u.norm2().to_sqrt_two();
            if (xi.sign() < 0 || xi.adj2().sign() < 0) {
                return true;
            }
            if (zx * real_part(u) + zy * imag_part(u) < half_plane[2]) {
                return true;
            }
            return visit(u);
        };

        // The cosets Z[√2] + iZ[√2] and Z[√2] + iZ[√2] + ω, with ω• = -ω, see
        // solve_two_dim_problem.
        if (!solve_upright_coset(PulledBackRegion{e1, g, 0, k, half_plane},
                                 PulledBackRegion{e2, g_conj, 0, k, std::nullopt}, ZOmega(0),
                                 filter)) {
            return false;
        }
        const Real &h = inv_sqrt2();
        return solve_upright_coset(PulledBackRegion{e1.offset(-h), g, 1, k, half_plane},
                                   PulledBackRegion{e2.offset(h), g_conj, -1, k, std::nullopt},
                                   ZOmega(0, 0, 1, 0), filter);
    }

  private:
    Real zx, zy, target;
    GridOp grid_op;
    std::array<ZSqrtTwo, 4> g, g_conj;
    EllipseState upright{Ellipse(), Ellipse()};
};

} // namespace RSDecomp::GridProblem
