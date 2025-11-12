#pragma once

#include <algorithm>
#include <boost/multiprecision/miller_rabin.hpp>
#include <cmath>
#include <map>
#include <numeric> // For std::gcd
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

// Assumes rings.h defines ZSqrtTwo and ZOmega classes with all necessary
// arithmetic operators (*, /, %), methods (adj2, conj, norm_squared, sqrt, to_omega),
// and comparison operators (e.g., operator!=).
#include "rings.hpp"

namespace NormSolver {

// Forward declarations
INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p);
bool primality_test(INT_TYPE n);
std::optional<INT_TYPE> sqrt_modulo_p(INT_TYPE n, INT_TYPE p);
std::optional<INT_TYPE> integer_factorize(INT_TYPE n, int max_tries = 1000);
std::optional<std::vector<INT_TYPE>> prime_factorize(INT_TYPE n, int max_trials = 1000,
                                                     bool z_sqrt_two = true);
std::optional<std::vector<ZSqrtTwo>> factorize_prime_zsqrt_two(INT_TYPE p);
std::optional<ZOmega> factorize_prime_zomega(const ZSqrtTwo &x, INT_TYPE p);
std::optional<ZOmega> solve_diophantine(const ZSqrtTwo &xi, int max_trials = 1000);

// --- Helper Functions for Modular Arithmetic ---

/**
 * @brief Performs modular multiplication (a * b) % mod, preventing overflow.
 */
inline INT_TYPE mod_mul(INT_TYPE a, INT_TYPE b, INT_TYPE mod)
{
    return static_cast<INT_TYPE>((static_cast<__int128>(a) * b) % mod);
}

/**
 * @brief Performs modular exponentiation (base^exp) % mod.
 */
inline INT_TYPE mod_pow(INT_TYPE base, INT_TYPE exp, INT_TYPE mod)
{
    INT_TYPE res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            res = mod_mul(res, base, mod);
        base = mod_mul(base, base, mod);
        exp /= 2;
    }
    return res;
}

// --- Greatest Common Divisor (GCD) ---

/**
 * @brief Computes GCD for standard integers.
 */
inline INT_TYPE gcd(INT_TYPE a, INT_TYPE b) { return std::gcd(a, b); }

/**
 * @brief Computes GCD for elements in the Z[sqrt(2)] ring.
 */
inline ZSqrtTwo gcd(ZSqrtTwo elem1, ZSqrtTwo elem2)
{
    // std::cout << "Computing GCD of ZSqrtTwo elements" << std::endl;
    // std::cout << "elem1: " << elem1 << std::endl;
    // std::cout << "elem2: " << elem2 << std::endl;
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
    // std::cout << "Computing GCD of ZOmega elements" << std::endl;
    // std::cout << "elem1: " << elem1 << std::endl;
    // std::cout << "elem2: " << elem2 << std::endl;
    while (elem2 != ZOmega(0, 0, 0, 0)) {
        ZOmega temp = elem1 % elem2;
        elem1 = elem2;
        elem2 = temp;
    }
    return elem1;
}

// --- Number Theoretic Algorithms ---

/**
 * @brief Deterministic Miller-Rabin primality test for 64-bit integers.
 */
inline bool primality_test(INT_TYPE n)
{
    static std::map<INT_TYPE, bool> cache;
    if (auto it = cache.find(n); it != cache.end()) {
        return it->second;
    }

    if (n < 2 || n == 4)
        return cache[n] = false;
    if (n < 4)
        return cache[n] = true;

    const std::vector<INT_TYPE> small_primes = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
                                                43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    for (INT_TYPE p : small_primes) {
        if (n == p)
            return cache[n] = true;
        if (n % p == 0)
            return cache[n] = false;
    }

    INT_TYPE d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    const std::vector<INT_TYPE> bases = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (INT_TYPE base : bases) {
        if (base >= n)
            continue;

        INT_TYPE x = mod_pow(base, d, n);
        if (x == 1 || x == n - 1)
            continue;

        bool is_composite = true;
        for (int r = 1; r < s; ++r) {
            x = mod_mul(x, x, n);
            if (x == n - 1) {
                is_composite = false;
                break;
            }
        }
        if (is_composite)
            return cache[n] = false;
    }

    return cache[n] = true;
}

// inline bool primality_test(INT_TYPE n) {
//     // std::cout << "Performing primality test for n: " << n << std::endl;

//     static std::map<INT_TYPE, bool> cache;
//     if (auto it = cache.find(n); it != cache.end()) {
//         return it->second;
//     }
//    if (boost::multiprecision::miller_rabin_test(n, 25)) {
//         return cache[n] = true;
//     }
//     else {
//         return cache[n] = false;
//     }
// }

/**
 * @brief Computes the Legendre symbol (a/p).
 */
inline INT_TYPE legendre_symbol(INT_TYPE a, INT_TYPE p)
{
    static std::map<std::pair<INT_TYPE, INT_TYPE>, INT_TYPE> cache;
    if (auto it = cache.find({a, p}); it != cache.end()) {
        return it->second;
    }
    return cache[{a, p}] = mod_pow(a, (p - 1) / 2, p);
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

        INT_TYPE b = mod_pow(c, 1LL << (m - i - 1), p);
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
    // FIX 1: The cache key must include max_tries.
    // std::pair is simpler than std::tuple for two items.
    static std::map<std::pair<INT_TYPE, int>, std::optional<INT_TYPE>> cache;
    auto cache_key = std::make_pair(n, max_tries);

    if (auto it = cache.find(cache_key); it != cache.end()) {
        return it->second;
    }

    // FIX 2: Cache the base cases.
    if (n <= 2)
        return cache[cache_key] = std::nullopt;
    if (n % 2 == 0)
        return cache[cache_key] = 2;

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
                for (int i = 0; i < std::min((long long)m, (long long)(r - k)); ++i) {
                    y = (mod_mul(y, y, n) + c) % n;
                    q = mod_mul(q, std::abs((long long)(x - y)), n);
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

        // FIX 1 (cont.): Use the correct composite key for caching.
        if (g != 1 && g != n)
            return cache[cache_key] = g;
    }

    // FIX 1 (cont.): Use the correct composite key for caching.
    return cache[cache_key] = std::nullopt;
}
/**
 * @brief Computes the prime factorization of an integer n.
 */
inline std::optional<std::vector<INT_TYPE>> prime_factorize(INT_TYPE n, int max_trials,
                                                            bool z_sqrt_two)
{
    static std::map<std::tuple<INT_TYPE, int, bool>, std::optional<std::vector<INT_TYPE>>> cache;
    if (auto it = cache.find({n, max_trials, z_sqrt_two}); it != cache.end()) {
        return it->second;
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
                return cache[{n, max_trials, z_sqrt_two}] = std::nullopt;
            }
            factors.push_back(p);
            continue;
        }

        auto factor_opt = integer_factorize(p, max_trials);
        if (!factor_opt)
            return cache[{n, max_trials, z_sqrt_two}] = std::nullopt;

        INT_TYPE factor = *factor_opt;
        if (z_sqrt_two && factor % 7 == 0) {
            return cache[{n, max_trials, z_sqrt_two}] = std::nullopt;
        }

        stack.push_back(*factor_opt);
        stack.push_back(p / *factor_opt);
    }

    std::sort(factors.begin(), factors.end());
    return cache[{n, max_trials, z_sqrt_two}] = factors;
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

    ZSqrtTwo res = gcd(ZSqrtTwo(p, 0), ZSqrtTwo(std::min(t, p - t), 1));
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
                // std::cout << "before factorization eta: " << eta.a << " + " << eta.b << "
                // sqrt(2)" << std::endl; std::cout << "before factorization factor: " << factor <<
                // std::endl;
                auto t = factorize_prime_zomega(eta, factor);
                // std::cout << "after factorization" << std::endl;
                if (!t)
                    return std::nullopt;
                // std::cout << "found t factor: " << (*t).a << " + " << (*t).b << " omega^3 + " <<
                // (*t).c << " omega^2 + " << (*t).d << std::endl;
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

} // namespace NormSolver
