#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>

template <typename T> inline T abs_val(T x) { return (x < 0) ? -x : x; }

template <typename T> inline T math_mod(T a, T n)
{
    if (n == 0) {
        throw std::invalid_argument("Modulo by zero");
    }
    if (n < 0) {
        n = -n;
    }
    T r = a % n;
    return r < 0 ? r + n : r;
}

template <typename T> inline T floor_div(T a, T b)
{
    if (b == 0) {
        throw std::invalid_argument("Division by zero");
    }
    T q = a / b;
    T r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
        q -= 1;
    }
    return q;
}

inline std::ostream &operator<<(std::ostream &os, __int128_t value)
{
    if (value == 0) {
        os << "0";
        return os;
    }
    std::string str;
    bool is_negative = false;
    if (value < 0) {
        is_negative = true;
    }
    while (value != 0) {
        int digit;
        if (is_negative) {
            digit = -(value % 10);
            value /= 10;
        }
        else {
            digit = value % 10;
            value /= 10;
        }
        str += (char)('0' + digit);
    }
    if (is_negative) {
        str += '-';
    }
    std::reverse(str.begin(), str.end());
    os << str;
    return os;
}

inline std::ostream &operator<<(std::ostream &os, unsigned __int128 n)
{
    if (n == 0) {
        os << "0";
        return os;
    }
    std::string str;
    while (n != 0) {
        str += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(str.begin(), str.end());
    os << str;
    return os;
}

template <typename Key, typename Value, size_t MaxSize>
class lru_cache {
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
                const auto &lru_item = cache_list.back();
                cache_map.erase(lru_item.first);
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
