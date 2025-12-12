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

#include "CliffordData.hpp"

namespace RSDecomp::CliffordData {
const std::map<std::vector<GateType>, SO3Matrix> clifford_group_to_SO3 = {
    {{I}, SO3Matrix{DyadicMatrix{{0, 0, 0, 1}, {}, {}, {0, 0, 0, 1}}}},
    {{H}, SO3Matrix{-DyadicMatrix{{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, 1}}},
    {{S}, SO3Matrix{DyadicMatrix{{-1, 0, 0, 0}, {}, {}, {0, 0, 1, 0}}}},
    {{X}, SO3Matrix{DyadicMatrix{{}, {0, -1, 0, 0}, {0, -1, 0, 0}, {}}}},
    {{Y}, SO3Matrix{DyadicMatrix{{}, {0, 0, 0, -1}, {0, 0, 0, 1}, {}}}},
    {{Z}, SO3Matrix{DyadicMatrix{{0, -1, 0, 0}, {}, {}, {0, 1, 0, 0}}}},
    {{Sd}, SO3Matrix{DyadicMatrix{{0, 0, -1, 0}, {}, {}, {1, 0, 0, 0}}}},
    {{H, S}, SO3Matrix{DyadicMatrix{{0, 0, -1, 0}, {-1, 0, 0, 0}, {0, 0, -1, 0}, {1, 0, 0, 0}, 1}}},
    {{H, Z}, SO3Matrix{DyadicMatrix{{0, 0, 0, 1}, {0, 0, 0, -1}, {0, 0, 0, 1}, {0, 0, 0, 1}, 1}}},
    {{H, Sd},
     SO3Matrix{DyadicMatrix{{-1, 0, 0, 0}, {0, 0, -1, 0}, {-1, 0, 0, 0}, {0, 0, 1, 0}, 1}}},
    {{S, H}, SO3Matrix{DyadicMatrix{{0, 0, -1, 0}, {0, 0, -1, 0}, {-1, 0, 0, 0}, {1, 0, 0, 0}, 1}}},
    {{S, X}, SO3Matrix{DyadicMatrix{{}, {0, 0, -1, 0}, {-1, 0, 0, 0}, {}}}},
    {{S, Y}, SO3Matrix{DyadicMatrix{{}, {1, 0, 0, 0}, {0, 0, 1, 0}, {}}}},
    {{Z, H}, SO3Matrix{DyadicMatrix{{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, -1}, {0, 0, 0, 1}, 1}}},
    {{Sd, H},
     SO3Matrix{DyadicMatrix{{-1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 0, -1, 0}, {0, 0, 1, 0}, 1}}},
    {{S, H, S}, SO3Matrix{DyadicMatrix{{0, 0, 0, 1}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, 1}}},
    {{S, H, Z},
     SO3Matrix{DyadicMatrix{{-1, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}, 1}}},
    {{S, H, Sd},
     SO3Matrix{DyadicMatrix{{0, -1, 0, 0}, {0, 0, 0, -1}, {0, 0, 0, 1}, {0, 1, 0, 0}, 1}}},
    {{Z, H, S},
     SO3Matrix{DyadicMatrix{{-1, 0, 0, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, 1}}},
    {{Z, H, Z},
     SO3Matrix{DyadicMatrix{{0, -1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, 1}}},
    {{Z, H, Sd},
     SO3Matrix{DyadicMatrix{{0, 0, -1, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, 1}}},
    {{Sd, H, S},
     SO3Matrix{DyadicMatrix{{0, -1, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, -1}, {0, 1, 0, 0}, 1}}},
    {{Sd, H, Z},
     SO3Matrix{DyadicMatrix{{0, 0, -1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, 1}}},
    {{Sd, H, Sd},
     SO3Matrix{DyadicMatrix{{0, 0, 0, 1}, {0, -1, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 1}, 1}}},
};

const std::unordered_map<GateType, std::pair<DyadicMatrix, double>> clifford_gates_to_SU2 = {
    {I, {DyadicMatrix{{0, 0, 0, 1}, {}, {}, {0, 0, 0, 1}}, 0.0}},
    {H, {-DyadicMatrix{{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, 1}, 0.5}},
    {S, {DyadicMatrix{{-1, 0, 0, 0}, {}, {}, {0, 0, 1, 0}}, 0.25}},
    {X, {DyadicMatrix{{}, {0, -1, 0, 0}, {0, -1, 0, 0}, {}}, 0.5}},
    {Y, {DyadicMatrix{{}, {0, 0, 0, -1}, {0, 0, 0, 1}, {}}, 0.5}},
    {Z, {DyadicMatrix{{0, -1, 0, 0}, {}, {}, {0, 1, 0, 0}}, 0.5}},
    {Sd, {DyadicMatrix{{0, 0, -1, 0}, {}, {}, {1, 0, 0, 0}}, 0.75}}};

const std::map<std::array<int, 3>, ParityTransformInfo> parity_transforms = {
    // "C"
    {{1, 1, 1}, {SO3Matrix(C_DATA), {I}, 0.0}},
    // "T"
    {{2, 2, 0}, {SO3Matrix(T_DATA), {T}, 0.0}},
    // "HT"
    {{0, 2, 2}, {SO3Matrix(HT_DATA), {HT}, 1.5}},
    // "SHT"
    {{2, 0, 2}, {SO3Matrix(SHT_DATA), {SHT}, 1.25}}};

} // namespace RSDecomp::CliffordData
