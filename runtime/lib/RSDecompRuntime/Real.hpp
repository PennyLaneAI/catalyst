// Copyright 2026 Xanadu Quantum Technologies Inc.

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
#include <limits>

#include "boost/multiprecision/cpp_bin_float.hpp"

#include "Rings.hpp"

// Multiprecision reals for the geometric part of the Ross-Selinger search.
namespace RSDecomp::Reals {

using namespace RSDecomp::Rings;

// The smallest epsilon supported by the search.
constexpr double MIN_EPSILON = 1e-15;

// newsynth (and pygridsynth) use 15 + 2.5 * log10(1/epsilon) significant digits, which is 53 at
// MIN_EPSILON. 192 bits are about 57 digits, and fit into three 64-bit limbs.
using Real = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<192, boost::multiprecision::digit_base_2>,
    boost::multiprecision::et_off>;

inline const Real &sqrt2() {
    static const Real value = sqrt(Real(2));
    return value;
}

inline const Real &inv_sqrt2() {
    static const Real value = sqrt2() / 2;
    return value;
}

inline const Real &lambda() {
    static const Real value = 1 + sqrt2();
    return value;
}

inline const Real &log_lambda() {
    static const Real value = log(lambda());
    return value;
}

inline const Real &log2_lambda() {
    static const Real value = log_lambda() / log(Real(2));
    return value;
}

/**
 * @brief Slack used to widen intervals so that rounding never drops a grid point.
 */
inline const Real &tolerance() {
    static const Real value = std::numeric_limits<Real>::epsilon() * 1000;
    return value;
}

inline INT_TYPE floor_int(const Real &x) { return floor(x).convert_to<INT_TYPE>(); }

inline INT_TYPE ceil_int(const Real &x) { return ceil(x).convert_to<INT_TYPE>(); }

inline INT_TYPE round_int(const Real &x) { return round(x).convert_to<INT_TYPE>(); }

inline Real to_real(const ZSqrtTwo &x) { return Real(x.a) + Real(x.b) * sqrt2(); }

/**
 * @brief Real part of a*ω^3 + b*ω^2 + c*ω + d with ω = (1 + i)/√2.
 */
inline Real real_part(const ZOmega &x) { return Real(x.d) + Real(x.c - x.a) * inv_sqrt2(); }

/**
 * @brief Imaginary part of a*ω^3 + b*ω^2 + c*ω + d with ω = (1 + i)/√2.
 */
inline Real imag_part(const ZOmega &x) { return Real(x.b) + Real(x.c + x.a) * inv_sqrt2(); }

} // namespace RSDecomp::Reals
