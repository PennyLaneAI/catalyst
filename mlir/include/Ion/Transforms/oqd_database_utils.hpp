// Copyright 2024 Xanadu Quantum Technologies Inc.

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

#include <type_traits>
#include <vector>

#include <toml++/toml.hpp>

namespace {

template <typename T> std::vector<T> tomlArray2StdVector(const toml::array &arr)
{
    // A toml node can contain toml objects of arbitrary types, even other toml nodes
    // i.e. toml nodes are similar to pytrees
    // Therefore, toml++ does not provide a simple "toml array to std vector" converter
    //
    // For a "leaf" array node, whose contents are now simple values,
    // such a utility would come in handy.

    std::vector<T> vec;

    if constexpr (std::is_same_v<T, int64_t>) {
        for (const auto &elem : arr) {
            vec.push_back(elem.as_integer()->get());
        }
    }
    else if constexpr (std::is_same_v<T, double>) {
        for (const auto &elem : arr) {
            vec.push_back(elem.as_floating_point()->get());
        }
    }

    return vec;
}
} // namespace
