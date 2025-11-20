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

#include <algorithm>
#include <boost/multiprecision/miller_rabin.hpp>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "rings.hpp"
#include "utils.hpp"

namespace RSDecomp::NormSolver {
using namespace RSDecomp::Utils;
using namespace RSDecomp::Rings;
INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p);
bool primality_test(INT_TYPE n);
std::optional<INT_TYPE> sqrt_modulo_p(INT_TYPE n, INT_TYPE p);
std::optional<INT_TYPE> integer_factorize(INT_TYPE n, int max_tries = 1000);
std::optional<std::vector<INT_TYPE>> prime_factorize(INT_TYPE n, int max_trials = 1000,
                                                     bool z_sqrt_two = true);
std::optional<std::vector<ZSqrtTwo>> factorize_prime_zsqrt_two(INT_TYPE p);
std::optional<ZOmega> factorize_prime_zomega(const ZSqrtTwo &x, INT_TYPE p);
std::optional<ZOmega> solve_diophantine(const ZSqrtTwo &xi, int max_trials = 1000);

/**
 * @brief Computes GCD for standard integers.
 */
inline INT_TYPE gcd(INT_TYPE a, INT_TYPE b) { return std::gcd(a, b); }

/**
 * @brief Computes GCD for elements in the Z[sqrt(2)] ring.
 */
inline ZSqrtTwo gcd(ZSqrtTwo elem1, ZSqrtTwo elem2)
{
    while (elem2 != ZSqrtTwo(0, 0)) {
        ZSqrtTwo temp = elem1 % elem2;
        elem1 = elem2;
        elem2 = temp;
    }
    return elem1;
}

/**
 * @brief Computes GCD for elements in the Z[omega] ring.
 */
inline ZOmega gcd(ZOmega elem1, ZOmega elem2)
{
    while (elem2 != ZOmega(0, 0, 0, 0)) {
        ZOmega temp = elem1 % elem2;
        elem1 = elem2;
        elem2 = temp;
    }
    return elem1;
}

// --- Number Theoretic Algorithms ---
/**
 * @brief Probabilistic Miller-Rabin primality test.
 */
inline bool primality_test(INT_TYPE n)
{
    static lru_cache<INT_TYPE, bool, 100000> cache;
    if (auto val_opt = cache.get(n); val_opt) {
        return *val_opt;
    }
    if (boost::multiprecision::miller_rabin_test(n, 25)) {
        cache.put(n, true);
        return true;
    }
    else {
        cache.put(n, false);
        return false;
    }
}

/**
 * @brief Computes the Legendre symbol (a/p).
 */
inline INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p)
{
    static lru_cache<std::pair<INT_TYPE, INT_TYPE>, INT_TYPE, 100000> cache;
    auto key = std::make_pair(a, p);

    if (auto val_opt = cache.get(key); val_opt) {
        return *val_opt;
    }

    INT_TYPE result = mod_pow(a, (p - 1) / 2, p);
    cache.put(key, result);
    return result;
}

/**
 * @brief Computes the square root of n modulo p using the Tonelli-Shanks algorithm.
 */
inline std::optional<INT_TYPE> sqrt_modulo_p(INT_TYPE n, INT_TYPE p)
{
    INT_TYPE a = n % p;
    if (a < 0)
        a += p;
    if (a == 0)
        return 0;
    if (p == 2)
        return a;
    if (legendre_symbol(a, p) != 1 || p % 2 == 0)
        return std::nullopt;

    INT_TYPE q = p - 1;
    int s = 0;
    while (q % 2 == 0) {
        q /= 2;
        s++;
    }

    if (s == 1)
        return mod_pow(a, (p + 1) / 4, p);

    INT_TYPE z = 2;
    while (legendre_symbol(z, p) != p - 1)
        z++;

    INT_TYPE m = s;
    INT_TYPE c = mod_pow(z, q, p);
    INT_TYPE t = mod_pow(a, q, p);
    INT_TYPE r = mod_pow(a, (q + 1) / 2, p);

    while (t != 1) {
        int i = 0;
        INT_TYPE t2i = t;
        while (t2i != 1) {
            t2i = mod_mul(t2i, t2i, p);
            i++;
            if (i == m)
                return std::nullopt;
        }

        INT_TYPE b = mod_pow<INT_TYPE>(c, 1LL << (m - i - 1), p);
        m = i;
        c = mod_mul(b, b, p);
        t = mod_mul(t, c, p);
        r = mod_mul(r, b, p);
    }

    return r;
}

/**
 * @brief Finds an integer factor of n using Brent's variant of Pollard's rho algorithm.
 */
inline std::optional<INT_TYPE> integer_factorize(INT_TYPE n, int max_tries)
{
    // Use lru_cache instead of std::map
    static lru_cache<std::pair<INT_TYPE, int>, std::optional<INT_TYPE>, 100000> cache;
    auto cache_key = std::make_pair(n, max_tries);

    if (auto val_opt = cache.get(cache_key); val_opt) {
        return *val_opt;
    }

    if (n <= 2) {
        cache.put(cache_key, std::nullopt);
        return std::nullopt;
    }
    if (n % 2 == 0) {
        cache.put(cache_key, 2);
        return 2;
    }

    static std::mt19937 gen(std::random_device{}());

    // max_tries is a copy (pass-by-value), so decrementing it is fine.
    // The original value is already stored in cache_key.
    while (max_tries-- > 0) {
        std::uniform_int_distribution<INT_TYPE> distrib(1, n - 1);
        INT_TYPE y = distrib(gen);
        INT_TYPE c = distrib(gen);
        INT_TYPE m = distrib(gen);
        INT_TYPE g = 1, r = 1, q = 1, x = y, xs;

        while (g == 1) {
            x = y;
            for (int i = 0; i < r; ++i)
                y = (mod_mul(y, y, n) + c) % n;

            INT_TYPE k = 0;
            while (k < r && g == 1) {
                xs = y;
                for (INT_TYPE i = 0; i < min(m, r - k); ++i) {
                    y = (mod_mul(y, y, n) + c) % n;
                    q = mod_mul(q, abs_val(x - y), n);
                }
                g = gcd(q, n);
                k += m;
            }
            r *= 2;
        }

        if (g == n) {
            g = 1;
            y = xs;
            while (g == 1) {
                y = (mod_mul(y, y, n) + c) % n;
                g = gcd(std::abs(static_cast<long long>(x - y)), n);
            }
        }

        if (g != 1 && g != n) {
            cache.put(cache_key, g);
            return g;
        }
    }

    cache.put(cache_key, std::nullopt);
    return std::nullopt;
}
/**
 * @brief Computes the prime factorization of an integer n.
 */
inline std::optional<std::vector<INT_TYPE>> prime_factorize(INT_TYPE n, int max_trials,
                                                            bool z_sqrt_two)
{
    static lru_cache<std::tuple<INT_TYPE, int, bool>, std::optional<std::vector<INT_TYPE>>, 100000>
        cache;
    auto cache_key = std::make_tuple(n, max_trials, z_sqrt_two);

    if (auto val_opt = cache.get(cache_key); val_opt) {
        return *val_opt;
    }

    std::vector<INT_TYPE> factors;
    std::vector<INT_TYPE> stack;
    stack.push_back(n);

    while (!stack.empty()) {
        INT_TYPE p = stack.back();
        stack.pop_back();

        if (p <= 1)
            continue;

        if (primality_test(p)) {
            if (z_sqrt_two && p % 8 == 7) {
                cache.put(cache_key, std::nullopt);
                return std::nullopt;
            }
            factors.push_back(p);
            continue;
        }

        auto factor_opt = integer_factorize(p, max_trials);
        if (!factor_opt) {
            cache.put(cache_key, std::nullopt);
            return std::nullopt;
        }

        INT_TYPE factor = *factor_opt;
        if (z_sqrt_two && factor % 7 == 0) {
            cache.put(cache_key, std::nullopt);
            return std::nullopt;
        }

        stack.push_back(*factor_opt);
        stack.push_back(p / *factor_opt);
    }

    std::sort(factors.begin(), factors.end());
    cache.put(cache_key, factors);
    return factors;
}

// --- Factorization in Rings ---

/**
 * @brief Factorizes a prime p in the ring Z[sqrt(2)].
 */
inline std::optional<std::vector<ZSqrtTwo>> factorize_prime_zsqrt_two(INT_TYPE p)
{
    if (std::abs(static_cast<int>(p)) == 2) {
        return std::vector<ZSqrtTwo>{ZSqrtTwo(0, 1), ZSqrtTwo(0, (p < 0) ? -1 : 1)};
    }

    if (p % 8 == 3 || p % 8 == 5) {
        return std::vector<ZSqrtTwo>{ZSqrtTwo(p, 0)};
    }

    auto t_opt = sqrt_modulo_p(2, p);
    if (!t_opt)
        return std::nullopt;
    INT_TYPE t = *t_opt;

    ZSqrtTwo res = gcd(ZSqrtTwo(p, 0), ZSqrtTwo(min(t, p - t), 1));
    return std::vector<ZSqrtTwo>{res, res.adj2()};
}

/**
 * @brief Finds a prime factor of x in Z[omega], where x is a factor of prime p.
 */
inline std::optional<ZOmega> factorize_prime_zomega(const ZSqrtTwo &x, INT_TYPE p)
{
    if (p == 2)
        return ZOmega(0, 0, 1, 1);

    INT_TYPE a = p % 8;
    if (a % 2 == 0 || a == 7)
        return std::nullopt;

    if (a == 1 || a == 5) {
        auto h_opt = sqrt_modulo_p(-1, p);
        if (!h_opt)
            return std::nullopt;
        return gcd(ZOmega(0, 1, 0, *h_opt), ZOmega(-x.b, 0, x.b, x.a));
    }

    // Default case: a == 3
    auto h_opt = sqrt_modulo_p(-2, p);
    if (!h_opt)
        return std::nullopt;
    return gcd(ZOmega(1, 0, 1, *h_opt), ZOmega(-x.b, 0, x.b, x.a));
}

// --- Main Diophantine Solver ---

/**
 * @brief Solves the Diophantine equation t*t = xi for t in Z[omega].
 */
inline std::optional<ZOmega> solve_diophantine(const ZSqrtTwo &xi, int max_trials)
{
    // std::cout << "solving diophantine for xi: " << xi.a << " + " << xi.b << " sqrt(2)" <<
    // std::endl;
    if (xi.a == 0 && xi.b == 0)
        return ZOmega(0, 0, 0, 0);

    INT_TYPE p = xi.abs();
    if (p < 2)
        return std::nullopt;

    auto factors_opt = prime_factorize(p, max_trials);
    if (!factors_opt)
        return std::nullopt;

    ZOmega scale(0, 0, 0, 1); // Represents 1 in Z[omega]
    ZSqrtTwo next_xi = xi;
    for (INT_TYPE factor : *factors_opt) {
        auto primes_zsqrt_two_opt = factorize_prime_zsqrt_two(factor);
        if (!primes_zsqrt_two_opt)
            return std::nullopt;

        for (const auto &eta : *primes_zsqrt_two_opt) {
            next_xi = next_xi * eta.adj2();
            INT_TYPE next_ab = eta.abs();

            if ((next_xi.a % next_ab == 0) && (next_xi.b % next_ab == 0)) {
                next_xi = ZSqrtTwo(next_xi.a / next_ab, next_xi.b / next_ab);
                auto t = factorize_prime_zomega(eta, factor);
                if (!t)
                    return std::nullopt;
                scale = scale * (*t);
            }
        }
    }

    ZSqrtTwo s_val = (scale.conj() * scale).to_sqrt_two();
    INT_TYPE s_abs = s_val.abs();
    ZSqrtTwo s_new = xi * s_val.adj2();
    if (s_new.a % s_abs != 0 || s_new.b % s_abs != 0)
        return std::nullopt;

    auto t2 = xi / s_val;
    if (std::pow(t2.abs(), 2) != 1)
        return std::nullopt;

    auto t2_sqrt = t2.sqrt();
    if (!t2_sqrt)
        return std::nullopt;
    return scale * (*t2_sqrt).to_omega();
}

} // namespace RSDecomp::NormSolver
