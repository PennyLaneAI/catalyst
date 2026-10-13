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

// Functions for the runtime_call tests.

#include <cstdint>
#include <cstring>

static std::uint64_t stored;

extern "C" {

std::int32_t native_add(std::int32_t a, std::int32_t b) { return a + b; }
double native_scale(double value, float factor) { return value * factor; }
const void *native_echo(const void *ptr) { return ptr; }
std::uint64_t native_strlen(const char *text) { return std::strlen(text); }
void native_store(std::uint64_t value) { stored = value; }
std::uint64_t native_load() { return stored; }

std::int32_t native_sum(const std::int8_t *data, std::uint64_t size) {
    std::int32_t sum = 0;
    for (std::uint64_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

std::int32_t native_fill(std::uint8_t *out, std::uint64_t size, std::uint8_t first) {
    for (std::uint64_t i = 0; i < size; ++i) {
        out[i] = static_cast<std::uint8_t>(first + i);
    }
    return 0;
}

} // extern "C"

namespace demo {

std::int32_t add(std::int32_t a, std::int32_t b) { return a + b; } // _ZN4demo3addEii
std::int32_t twice(std::int32_t value) { return 2 * value; }
double twice(double value) { return 2 * value; }

} // namespace demo
