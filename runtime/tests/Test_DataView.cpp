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

#include <cstdio>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DataView.hpp"

using namespace Catch::Matchers;

using namespace Catalyst::Runtime;

TEST_CASE("Test DataView Pre-Increment Iterator - double, 1", "[DataView]")
{
    double data_aligned[3] = {1.0, 1.1, 1.2};
    size_t offset = 0U;
    size_t sizes[1] = {3};
    size_t strides[1] = {1};

    DataView<double, 1> view(data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    CHECK(*view_iter == Catch::Approx(1.0).epsilon(1e-5));
    CHECK(*++view_iter == Catch::Approx(1.1).epsilon(1e-5));
    CHECK(*++view_iter == Catch::Approx(1.2).epsilon(1e-5));
}

TEST_CASE("Test DataView Pre-Increment Iterator - int, 2", "[DataView]")
{
    int data_aligned[3][3] = {{0, 1, 2}, {3, 4, 5}};
    size_t offset = 0U;
    size_t sizes[2] = {3, 3};
    size_t strides[2] = {3, 1};

    DataView<int, 2> view(*data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    CHECK(*view_iter == 0);
    for (int i = 1; i < 6; i++) {
        CHECK(*++view_iter == i);
    }
}

TEST_CASE("Test DataView Pre-Increment Iterator - int, 3", "[DataView]")
{
    int data_aligned[18] = {0};
    for (int i = 0; i < 18; i++) {
        data_aligned[i] = i;
    }

    size_t offset = 0U;
    size_t sizes[3] = {3, 2, 3};
    size_t strides[3] = {6, 3, 1};

    DataView<int, 3> view(data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    CHECK(*view_iter == 0);
    for (int i = 1; i < 18; i++) {
        CHECK(*++view_iter == i);
    }
}

TEST_CASE("Test DataView Post-Increment Iterator - double, 1", "[DataView]")
{
    double data_aligned[3] = {3.2, 4.1, 1.6};
    size_t offset = 0;
    size_t sizes[1] = {3};
    size_t strides[1] = {1};

    DataView<double, 1> view(data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    CHECK(*view_iter++ == Catch::Approx(3.2).epsilon(1e-5));
    CHECK(*view_iter++ == Catch::Approx(4.1).epsilon(1e-5));
    CHECK(*view_iter == Catch::Approx(1.6).epsilon(1e-5));
}

TEST_CASE("Test DataView Post-Increment Iterator - int, 2", "[DataView]")
{
    int data_aligned[3][3] = {{0, 1, 2}, {3, 4, 5}};
    size_t offset = 0U;
    size_t sizes[2] = {3, 3};
    size_t strides[2] = {3, 1};

    DataView<int, 2> view(*data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    for (int i = 0; i < 6; i++) {
        CHECK(*view_iter++ == i);
    }
}

TEST_CASE("Test DataView Post-Increment Iterator - int, 3", "[DataView]")
{
    int data_aligned[18] = {0};
    for (int i = 0; i < 18; i++) {
        data_aligned[i] = i;
    }

    size_t offset = 0U;
    size_t sizes[3] = {3, 2, 3};
    size_t strides[3] = {6, 3, 1};

    DataView<int, 3> view(data_aligned, offset, sizes, strides);

    auto view_iter = view.begin();

    for (int i = 0; i < 18; i++) {
        CHECK(*view_iter++ == i);
    }
}

TEST_CASE("DataView Iterator Distance 0 - 0 first axis", "[DataView]")
{
    int *data_aligned = nullptr;
    size_t offset = 0;
    size_t sizes[2] = {0, 10};
    size_t strides[2] = {0, 0};

    DataView<int, 2> view(data_aligned, offset, sizes, strides);

    CHECK(std::distance(view.begin(), view.end()) == 0);
}

TEST_CASE("DataView Iterator Distance 0 - 0 second axis", "[DataView]")
{
    int *data_aligned = nullptr;
    size_t offset = 0;
    size_t sizes[2] = {10, 0};
    size_t strides[2] = {0, 0};

    DataView<int, 2> view(data_aligned, offset, sizes, strides);

    CHECK(std::distance(view.begin(), view.end()) == 0);
}

TEST_CASE("DataView Iterator Distance 4 - int, 2", "[DataView]")
{
    int data_aligned[2][2] = {{1, 2}, {3, 4}};
    size_t offset = 0;
    size_t sizes[2] = {2, 2};
    size_t strides[2] = {2, 1};

    DataView<int, 2> view(*data_aligned, offset, sizes, strides);

    CHECK(std::distance(view.begin(), view.end()) == 4);
}

TEST_CASE("DataView Iterator Distance 12 - double, 3", "[DataView]")
{
    double data_aligned[2][2][3] = {{{3.1, 2.6, 9.5}, {5.4, 2.3, 8.1}},
                                    {{9.8, 8.2, 7.2}, {0.7, 9.6, 6.6}}};
    size_t offset = 0;
    size_t sizes[3] = {2, 2, 3};
    size_t strides[3] = {6, 3, 1};

    DataView<double, 3> view(**data_aligned, offset, sizes, strides);

    CHECK(std::distance(view.begin(), view.end()) == 12);
}

TEST_CASE("DataView Size - 0 first axis", "[DataView]")
{
    int *data_aligned = nullptr;
    size_t offset = 0;
    size_t sizes[2] = {0, 10};
    size_t strides[2] = {0, 0};

    DataView<int, 2> view(data_aligned, offset, sizes, strides);

    CHECK(view.size() == 0);
}
