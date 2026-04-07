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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <cstdio>

#include "RSUtils.hpp"
#include "Rings.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::Rings;
using namespace RSDecomp::Utils;

TEST_CASE("Test Math helper", "[RSDecomp][Utils]")
{
    SECTION("Min/Max/Abs_val")
    {
        CHECK(min(INT_TYPE(3), INT_TYPE(-5)) == INT_TYPE(-5));
        CHECK(max(INT_TYPE(3), INT_TYPE(-5)) == INT_TYPE(3));

        CHECK(abs_val(INT_TYPE(-10)) == INT_TYPE(10));
        CHECK(abs_val(INT_TYPE(7)) == INT_TYPE(7));
    }

    SECTION("floor_div")
    {
        CHECK(floor_div(INT_TYPE(10), INT_TYPE(3)) == INT_TYPE(3));
        CHECK(floor_div(INT_TYPE(1), INT_TYPE(2)) == INT_TYPE(0));
        CHECK(floor_div(INT_TYPE(-10), INT_TYPE(3)) == INT_TYPE(-4));
        CHECK(floor_div(INT_TYPE(-1), INT_TYPE(2)) == INT_TYPE(-1));
        CHECK(floor_div(INT_TYPE(10), INT_TYPE(-3)) == INT_TYPE(-4));
        CHECK(floor_div(INT_TYPE(-10), INT_TYPE(-3)) == INT_TYPE(3));
        CHECK(floor_div(INT_TYPE(-9), INT_TYPE(3)) == INT_TYPE(-3));
        CHECK(floor_div(INT_TYPE(9), INT_TYPE(-3)) == INT_TYPE(-3));
        CHECK(floor_div(INT_TYPE(0), INT_TYPE(5)) == INT_TYPE(0));
        CHECK(floor_div(INT_TYPE(5), INT_TYPE(-1)) == INT_TYPE(-5));
        REQUIRE_THROWS_WITH(floor_div(INT_TYPE(5), INT_TYPE(0)),
                            ContainsSubstring("Division by zero"));
    }

    SECTION("GCD")
    {
        CHECK(gcd(INT_TYPE(48), INT_TYPE(18)) == INT_TYPE(6));
        CHECK(gcd(ZSqrtTwo(31, 13), ZSqrtTwo(65, 33)) == ZSqrtTwo(-3, -7));
        CHECK(gcd(ZOmega(31, 32, 32, -11), ZOmega(96, 106, 74, -44)) == ZOmega(-3, 5, 2, 2));
    }
}

TEST_CASE("LRU Cache Basic Operations", "[LRUCache]")
{
    lru_cache<int, std::string, 3> cache;

    SECTION("Starts empty")
    {
        CHECK(cache.size() == 0);
        CHECK(cache.get(1) == std::nullopt);
    }

    SECTION("Put and Get")
    {
        cache.put(1, "one");
        cache.put(2, "two");

        CHECK(cache.size() == 2);
        CHECK(cache.get(1) == "one");
        CHECK(cache.get(2) == "two");
        CHECK(cache.get(3) == std::nullopt);
    }

    SECTION("Update existing key")
    {
        cache.put(1, "one");
        cache.put(1, "ONE_UPDATED");

        CHECK(cache.size() == 1);
        CHECK(cache.get(1) == "ONE_UPDATED");
    }
}

TEST_CASE("LRU Eviction Logic", "[LRUCache]")
{
    lru_cache<int, int, 3> cache;

    cache.put(1, 100);
    cache.put(2, 200);
    cache.put(3, 300);

    CHECK(cache.size() == 3);

    SECTION("Evicts oldest inserted when no access happens")
    {
        // Cache is [3, 2, 1] (Most Recent -> Least Recent)

        // Add 4th item. 1 should be evicted.
        cache.put(4, 400);

        CHECK(cache.size() == 3);
        CHECK(cache.get(4) == 400);
        CHECK(cache.get(3) == 300);
        CHECK(cache.get(2) == 200);
        CHECK(cache.get(1) == std::nullopt); // 1 is gone
    }

    SECTION("Accessing an item prevents its eviction")
    {
        // Cache is [3, 2, 1]

        // Access 1. It moves to the front.
        // Cache becomes [1, 3, 2]
        cache.get(1);

        // Add 4th item. 2 (now least recent) should be evicted.
        cache.put(4, 400);

        CHECK(cache.get(1) == 100);          // 1 is still there
        CHECK(cache.get(2) == std::nullopt); // 2 is gone
        CHECK(cache.get(4) == 400);
    }

    SECTION("Updating an item prevents its eviction")
    {
        // Cache is [3, 2, 1]

        // Update 1. It moves to front.
        // Cache becomes [1, 3, 2]
        cache.put(1, 101);

        // Add 4th item. 2 should be evicted.
        cache.put(4, 400);

        CHECK(cache.get(1) == 101);
        CHECK(cache.get(2) == std::nullopt);
    }
}

TEST_CASE("LRU Cache Edge Cases", "[LRUCache]")
{
    SECTION("Cache with MaxSize 1")
    {
        lru_cache<int, int, 1> tiny_cache;

        tiny_cache.put(1, 10);
        CHECK(tiny_cache.get(1) == 10);

        tiny_cache.put(2, 20); // Should evict 1 immediately
        CHECK(tiny_cache.get(2) == 20);
        CHECK(tiny_cache.get(1) == std::nullopt);
    }

    SECTION("Clearing the cache")
    {
        lru_cache<int, int, 3> cache;
        cache.put(1, 10);
        cache.put(2, 20);

        CHECK(cache.size() == 2);

        cache.clear();

        CHECK(cache.size() == 0);
        CHECK(cache.get(1) == std::nullopt);

        // Ensure we can fill it again
        cache.put(1, 10);
        CHECK(cache.size() == 1);
    }
}

TEST_CASE("LRU Cache Complex Types", "[LRUCache]")
{
    using KeyType = std::pair<int, int>;
    lru_cache<KeyType, int, 2> pair_cache;

    KeyType k1 = {1, 2};
    KeyType k2 = {3, 4};
    KeyType k3 = {5, 6};

    pair_cache.put(k1, 100);
    pair_cache.put(k2, 200);

    CHECK(pair_cache.get({1, 2}) == 100);

    // Evict k2 (since k1 was just accessed)
    pair_cache.put(k3, 300);

    CHECK(pair_cache.get(k3) == 300);
    CHECK(pair_cache.get(k1) == 100);          // k1 should still be there
    CHECK(pair_cache.get(k2) == std::nullopt); // k2 was least recent
}
