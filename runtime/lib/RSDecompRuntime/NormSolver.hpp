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
#include <boost/random.hpp>
#include <cmath>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "RSUtils.hpp"
#include "Rings.hpp"

#define MAX_FACTORING_TRIALS 1000
#define FACTORING_CACHE_SIZE 100000

namespace RSDecomp::NormSolver {

using namespace RSDecomp::Utils;
using namespace RSDecomp::Rings;

// Forward declarations
INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p);
bool primality_test(INT_TYPE n);
std::optional<INT_TYPE> sqrt_modulo_p(INT_TYPE n, INT_TYPE p);
std::optional<INT_TYPE> integer_factorize(INT_TYPE n, int max_trials = MAX_FACTORING_TRIALS);
std::optional<std::vector<INT_TYPE>>
prime_factorize(INT_TYPE n, int max_trials = MAX_FACTORING_TRIALS, bool z_sqrt_two = true);
std::optional<std::vector<ZSqrtTwo>> factorize_prime_zsqrt_two(INT_TYPE p);
std::optional<ZOmega> factorize_prime_zomega(const ZSqrtTwo &x, INT_TYPE p);
std::optional<ZOmega> solve_diophantine(const ZSqrtTwo &xi, int max_trials = MAX_FACTORING_TRIALS);

// --- Number Theoretic Algorithms ---

/**
 * @brief Determines whether an integer is prime or not.
 *
 * This function uses the probabilistic Miller-Rabin primality test.
 *
 * @param n The number to test for primality.
 * @return true if n is likely prime, false otherwise.
 */
inline bool primality_test(INT_TYPE n)
{
    static lru_cache<INT_TYPE, bool, FACTORING_CACHE_SIZE> cache;
    if (auto val_opt = cache.get(n); val_opt) {
        return *val_opt;
    }

    // 25 iterations provides a very high probability of correctness
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
 *
 * This function uses the definition of the Legendre symbol:
 * (a/p) = a^((p-1)/2) mod p
 *
 * @param a The number to compute the Legendre symbol of.
 * @param p The prime number.
 * @return The Legendre symbol of a modulo p.
 */
inline INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p)
{
    static lru_cache<std::pair<INT_TYPE, INT_TYPE>, INT_TYPE, FACTORING_CACHE_SIZE> cache;
    auto key = std::make_pair(a, p);

    if (auto val_opt = cache.get(key); val_opt) {
        return *val_opt;
    }

    // Use boost's powm for modular exponentiation with multiprecision integers
    INT_TYPE exp = (p - 1) / 2;
    INT_TYPE result = boost::multiprecision::powm(a, exp, p);
    cache.put(key, result);
    return result;
}

/**
 * @brief Computes square root of n under modulo p if it exists.
 *
 * This uses Tonelli-Shanks algorithm to compute x, such that x^2 = n (mod p),
 * where p is an odd prime.
 * Ref: https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm
 *
 * @param n The number to compute the square root of.
 * @param p The odd prime modulus.
 * @return The square root of n under modulo p, or std::nullopt if it does not exist.
 */
inline std::optional<INT_TYPE> sqrt_modulo_p(INT_TYPE n, INT_TYPE p)
{
    // Trivial cases
    INT_TYPE a = n % p;
    if (a < 0)
        a += p; // Handle C++ negative modulo behavior
    if (a == 0)
        return INT_TYPE(0);
    if (p == 2)
        return a;
    if (legendre_symbol(a, p) != 1 || p % 2 == 0)
        return std::nullopt;

    // Factor p-1 as q*2^s with q odd
    INT_TYPE q = p - 1;
    int s = 0;
    while (q % 2 == 0) {
        q /= 2;
        s++;
    }

    // (p - 1) = 2.q ==> 2 * k
    if (s == 1) {
        INT_TYPE exp = (p + 1) / 4;
        return boost::multiprecision::powm(a, exp, p);
    }

    // Find a quadratic non-residue z, such that z^((p-1)/2) = -1 mod p
    INT_TYPE z = 2;
    while (legendre_symbol(z, p) != p - 1)
        z++;

    INT_TYPE m = s;
    INT_TYPE c = boost::multiprecision::powm(z, q, p);
    INT_TYPE t = boost::multiprecision::powm(a, q, p);
    INT_TYPE q_plus_1_div_2 = (q + 1) / 2;
    INT_TYPE r = boost::multiprecision::powm(a, q_plus_1_div_2, p);

    while (t != 1) {
        // Find least i in [1, m), such that t^(2^i) = 1 mod p
        int i = 0;
        INT_TYPE t2i = t;
        while (t2i != 1) {
            t2i = (t2i * t2i) % p;
            i++;
            if (i == m)
                return std::nullopt;
        }

        // Compute b = c^(2^(m-i-1)) mod p
        INT_TYPE m_minus_i_minus_1 = m - i - 1;

        INT_TYPE exp_val = INT_TYPE(1) << static_cast<unsigned>(m_minus_i_minus_1);
        INT_TYPE b = boost::multiprecision::powm(c, exp_val, p);

        // Update initial elements
        m = i;
        c = (b * b) % p;
        t = (t * c) % p;
        r = (r * b) % p;
    }

    return r;
}

/**
 * @brief Computes an integer factor of a number n.
 *
 * This function implements the Brent's variant of the Pollard's rho algorithm
 * for integer factorization.
 * Ref: https://doi.org/10.1007/BF01933190
 *
 * @param n The number to factor.
 * @param max_trials The maximum number of attempts to find a factor.
 * @return An integer factor of n, or std::nullopt if no factors are found.
 */
inline std::optional<INT_TYPE> integer_factorize(INT_TYPE n, int max_trials)
{
    static lru_cache<std::pair<INT_TYPE, int>, std::optional<INT_TYPE>, FACTORING_CACHE_SIZE> cache;
    auto cache_key = std::make_pair(n, max_trials);

    if (auto val_opt = cache.get(cache_key); val_opt) {
        return *val_opt;
    }

    if (n <= 2) {
        cache.put(cache_key, std::nullopt);
        return std::nullopt;
    }
    if (n % 2 == 0) {
        cache.put(cache_key, INT_TYPE(2));
        return INT_TYPE(2);
    }

    static boost::random::mt11213b gen(std::random_device{}());

    // Main loop: retry with different parameters on failure
    while (max_trials-- > 0) {
        INT_TYPE n_minus_1 = n - 1;
        boost::random::uniform_int_distribution<INT_TYPE> dist(1, n_minus_1);

        INT_TYPE y = dist(gen);
        INT_TYPE c = dist(gen);
        INT_TYPE m = dist(gen);

        INT_TYPE g = 1, r = 1, q = 1, x = y, xs;

        while (g == 1) {
            x = y;
            // Process next `r` steps
            for (INT_TYPE i = 0; i < r; ++i)
                y = ((y * y) % n + c) % n;

            INT_TYPE k = 0;
            while (k < r && g == 1) {
                xs = y;
                // Process next `min(m, r-k)` steps
                INT_TYPE loop_limit = min(m, r - k);
                for (INT_TYPE i = 0; i < loop_limit; ++i) {
                    y = ((y * y) % n + c) % n;
                    INT_TYPE diff = x > y ? x - y : y - x;
                    q = (q * diff) % n;
                }
                g = gcd(q, n);
                k += m;
            }
            r *= 2;
        }

        // Linear search for a factor if the above doesn't yield a result.
        if (g == n) {
            g = 1;
            y = xs;
            while (g == 1) {
                y = ((y * y) % n + c) % n;
                INT_TYPE diff = x > y ? x - y : y - x;
                g = gcd(diff, n);
            }
        }

        // If we found a valid integer factor, such that it is neither 1 nor n.
        if (g != 1 && g != n) {
            cache.put(cache_key, g);
            return g;
        }
    }

    cache.put(cache_key, std::nullopt);
    return std::nullopt;
}

/**
 * @brief Computes the prime factorization of a number n.
 *
 * This function uses a combination of the Brent's variant of Pollard's rho algorithm
 * for integer factorization and Miller-Rabin primality test.
 *
 * Note: The function returns std::nullopt in cases where a prime factor might be 7,
 * as it cannot be expressed in the ring Z[sqrt(2)] if z_sqrt_two is true.
 *
 * @param n The number to factor.
 * @param max_trials The maximum number of attempts to find a factor.
 * @param z_sqrt_two When true, returns factors only if expressible in Z[sqrt(2)].
 * (Returns nullopt if factor == 7 mod 8).
 * @return A vector of sorted prime factors, or std::nullopt.
 */
inline std::optional<std::vector<INT_TYPE>> prime_factorize(INT_TYPE n, int max_trials,
                                                            bool z_sqrt_two)
{
    static lru_cache<std::tuple<INT_TYPE, int, bool>, std::optional<std::vector<INT_TYPE>>,
                     FACTORING_CACHE_SIZE>
        cache;
    auto cache_key = std::make_tuple(n, max_trials, z_sqrt_two);

    if (auto val_opt = cache.get(cache_key); val_opt) {
        return *val_opt;
    }

    std::vector<INT_TYPE> factors;
    std::vector<INT_TYPE> stack;
    stack.emplace_back(n);

    while (!stack.empty()) {
        INT_TYPE p = stack.back();
        stack.pop_back();

        // Trivial case
        if (p <= 1)
            continue;

        // Check if p is prime
        if (primality_test(p)) {
            // Cannot split in Z[sqrt(2)] if p = 7 mod 8
            if (z_sqrt_two && p % 8 == 7) {
                cache.put(cache_key, std::nullopt);
                return std::nullopt;
            }
            factors.emplace_back(p);
            continue;
        }

        // Find a factor
        auto factor_opt = integer_factorize(p, max_trials);
        if (!factor_opt) {
            cache.put(cache_key, std::nullopt);
            return std::nullopt;
        }

        INT_TYPE factor = *factor_opt;
        // Check Z[sqrt(2)] constraint on the found factor
        if (z_sqrt_two && factor % 7 == 0) {
            cache.put(cache_key, std::nullopt);
            return std::nullopt;
        }

        // Push factor and its complement onto the stack
        stack.emplace_back(*factor_opt);
        stack.emplace_back(p / *factor_opt);
    }

    std::sort(factors.begin(), factors.end());
    cache.put(cache_key, factors);
    return factors;
}

// --- Factorization in Rings ---

/**
 * @brief Find the factorization of a prime number p in ring Z[sqrt(2)].
 *
 * Uses theory from Appendix C.2 of arXiv:1403.2975.
 * - Lemma C.7: Prime factorization of int prime p in Z[sqrt(2)] has 1 or 2 factors.
 * - Lemma C.8: Factorization of +/-2.
 * - Lemma C.9: p = 3, 5 mod 8 has 1 factor.
 * - Lemma C.11: p = 1, 7 mod 8 has 2 factors.
 * @param p A prime integer.
 * @return A vector of factors in Z[sqrt(2)], or std::nullopt if factorization fails.
 */
inline std::optional<std::vector<ZSqrtTwo>> factorize_prime_zsqrt_two(INT_TYPE p)
{
    // Lemma C.8: ±2 = (0 + 1√2)(0 ± 1√2)
    if (abs_val(p) == INT_TYPE(2)) {
        return std::vector<ZSqrtTwo>{ZSqrtTwo(0, 1), ZSqrtTwo(0, (p < 0) ? -1 : 1)};
    }

    // Lemma C.9: p = (p + 0√2) is prime in Z[√2]
    if (p % 8 == 3 || p % 8 == 5) {
        return std::vector<ZSqrtTwo>{ZSqrtTwo(p, 0)};
    }

    // Default case (Lemma C.11): p = ±1 mod 8
    // Solve t^2 = 2 (mod p)
    auto t_opt = sqrt_modulo_p(2, p);
    if (!t_opt)
        return std::nullopt;
    INT_TYPE t = *t_opt;

    // Perform ring GCD to get (a + b√2)(a - b√2)
    ZSqrtTwo res = gcd(ZSqrtTwo(p, 0), ZSqrtTwo(min(t, p - t), 1));
    return std::vector<ZSqrtTwo>{res, res.adj2()};
}

/**
 * @brief Find a prime factor of an element x in the ring Z[omega],
 * where x divides a prime integer p.
 *
 * Uses theory from Appendix C.3 of arXiv:1403.2975.
 * - Lemma C.13: Prime factorization of a prime p in Z[sqrt(2)] has one or two factors in Z[omega].
 * - Lemma C.20-21: Prime factorization of a prime p in Z[sqrt(2)] can be used as a possible
 * solution to the Diophantine equation t^* t = \xi iff p==2 or p == 1,3,5 mod 8.
 * @param x An element of Z[sqrt(2)] such that x | p.
 * @param p A prime integer.
 * @return A factor in Z[omega], or std::nullopt if factorization fails
 */
inline std::optional<ZOmega> factorize_prime_zomega(const ZSqrtTwo &x, INT_TYPE p)
{
    // Basic cases
    if (p == 2)
        return ZOmega(0, 0, 1, 1);

    INT_TYPE a = p % 8;
    // p = 2k or p = 7 mod 8, no factorization in Z[omega]
    if (a % 2 == 0 || a == 7)
        return std::nullopt;

    // p = 1, 5 mod 8, use h = sqrt(-1) mod p
    if (a == 1 || a == 5) {
        auto h_opt = sqrt_modulo_p(-1, p);
        if (!h_opt)
            return std::nullopt;
        return gcd(ZOmega(0, 1, 0, *h_opt), ZOmega(-x.b, 0, x.b, x.a));
    }

    // Default case: a == 3
    // p = 3 mod 8, use h = sqrt(-2) mod p
    auto h_opt = sqrt_modulo_p(-2, p);
    if (!h_opt)
        return std::nullopt;
    return gcd(ZOmega(1, 0, 1, *h_opt), ZOmega(-x.b, 0, x.b, x.a));
}

// --- Main Diophantine Solver ---

/**
 * @brief Solve the Diophantine equation t*t = xi for t in Z[omega] and xi in Z[sqrt(2)].
 *
 * Uses theory from Appendix C and D (Proof of Lemma 8.4) of arXiv:1403.2975.
 *
 * @param xi An element of the ring Z[sqrt(2)].
 * @param max_trials Maximum attempts for factorization.
 * @return An element of Z[omega] satisfying the equation, or std::nullopt.
 */
inline std::optional<ZOmega> solve_diophantine(const ZSqrtTwo &xi, int max_trials)
{
    if (xi.a == 0 && xi.b == 0)
        return ZOmega(0, 0, 0, 0);

    INT_TYPE p = xi.norm();
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
            // Scale the next_xi by the factor in Z[sqrt(2)]
            next_xi = next_xi * eta.adj2();
            INT_TYPE next_ab = eta.norm();

            // Check if next_xi is divisible by eta
            if ((next_xi.a % next_ab == 0) && (next_xi.b % next_ab == 0)) {
                next_xi = ZSqrtTwo(next_xi.a / next_ab, next_xi.b / next_ab);
                auto t = factorize_prime_zomega(eta, factor);
                if (!t)
                    return std::nullopt;
                scale = scale * (*t);
            }
        }
    }

    // The remaining quotient should be divisible
    ZSqrtTwo s_val = (scale.conj() * scale).to_sqrt_two();
    INT_TYPE s_abs = s_val.norm();
    ZSqrtTwo s_new = xi * s_val.adj2();

    if (s_new.a % s_abs != 0 || s_new.b % s_abs != 0)
        return std::nullopt;

    // The remaining quotient should be a unit in Z[sqrt(2)]
    auto t2 = xi / s_val;
    INT_TYPE t2_abs = t2.norm();
    if (t2_abs * t2_abs != 1)
        return std::nullopt;

    auto t2_sqrt = t2.sqrt();
    if (!t2_sqrt)
        return std::nullopt;
    return scale * (*t2_sqrt).to_omega();
}

} // namespace RSDecomp::NormSolver
