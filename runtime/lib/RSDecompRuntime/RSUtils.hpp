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
#include <list>
#include <map>
#include <optional>

// We define a lot of these utilities here to support multiprecision INTs and FLOATs, which cannot
// use the std algorithms.
namespace RSDecomp::Utils {
template <typename T> inline T min(T x, T y) { return (x > y) ? y : x; }

template <typename T> inline T max(T x, T y) { return (x > y) ? x : y; }

template <typename T> inline T abs_val(T x) { return (x < 0) ? -x : x; }

template <typename T> inline T floor_div(T a, T b)
{
    RT_FAIL_IF(b == T(0), "Division by zero");
    T q = a / b;
    T r = a % b;
    if ((r != T(0)) && ((r < T(0)) != (b < T(0)))) {
        q -= T(1);
    }
    return q;
}

/**
 * @brief Computes GCD for generic types supporting % and comparison with 0.
 */
template <typename T> T gcd(T a, T b)
{
    while (b != T(0)) {
        T temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

/**
 * @brief Simple LRU (Least Recently Used) Cache implementation.
 *
 * This cache stores key-value pairs up to a maximum size. When the cache exceeds this size,
 * the least recently used item is evicted.
 *
 * @tparam Key The type of the keys.
 * @tparam Value The type of the values.
 * @tparam MaxSize The maximum number of items the cache can hold.
 */
template <typename Key, typename Value, size_t MaxSize> class lru_cache {
    static_assert(MaxSize > 0, "LRU cache MaxSize must be greater than 0");

  public:
    using list_pair_t = std::pair<Key, Value>;
    using list_iterator_t = typename std::list<list_pair_t>::iterator;
    using map_t = std::map<Key, list_iterator_t>;

    /**
     * @brief Gets a value from the cache.
     *
     * If the key is found, the item is marked as most-recently-used
     * and its value is returned.
     *
     * @param key The key to look up.
     * @return std::optional<Value> The value if found, otherwise std::nullopt.
     */
    std::optional<Value> get(const Key &key)
    {
        auto map_it = cache_map.find(key);

        if (map_it == cache_map.end()) {
            return std::nullopt;
        }

        cache_list.splice(cache_list.begin(), cache_list, map_it->second);

        return map_it->second->second;
    }

    /**
     * @brief Puts a new (key, value) pair into the cache.
     *
     * If the key already exists, its value is updated.
     * If the cache is full, the least-recently-used item is evicted.
     * The new/updated item is marked as the most-recently-used.
     *
     * @param key The key of the item.
     * @param value The value of the item.
     */
    void put(const Key &key, const Value &value)
    {
        auto map_it = cache_map.find(key);

        if (map_it != cache_map.end()) {
            map_it->second->second = value;
            cache_list.splice(cache_list.begin(), cache_list, map_it->second);
        }
        else {
            if (cache_map.size() >= MaxSize) {
                const auto &[lru_key, lru_value] = cache_list.back();
                cache_map.erase(lru_key);
                cache_list.pop_back();
            }
            cache_list.push_front({key, value});
            cache_map[key] = cache_list.begin();
        }
    }

    /**
     * @brief Returns the current number of items in the cache.
     */
    size_t size() const { return cache_map.size(); }

    /**
     * @brief Clears all items from the cache.
     */
    void clear()
    {
        cache_map.clear();
        cache_list.clear();
    }

  private:
    std::list<list_pair_t> cache_list;
    map_t cache_map;
};

} // namespace RSDecomp::Utils
