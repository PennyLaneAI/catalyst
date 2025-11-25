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
#include <map>

#include "Rings.hpp"

namespace RSDecomp::CliffordData {
using namespace RSDecomp::Rings;
enum class GateType { T = 0, HT, SHT, I, X, Y, Z, H, S, Sd };
enum class PPRGateType {
    I = 0,
    X2,
    X4,
    X8,
    adjX2,
    adjX4,
    adjX8,
    Y2,
    Y4,
    Y8,
    adjY2,
    adjY4,
    adjY8,
    Z2,
    Z4,
    Z8,
    adjZ2,
    adjZ4,
    adjZ8
};

using enum CliffordData::GateType;

std::string_view gateTypeToString(GateType type);
std::ostream &operator<<(std::ostream &os, GateType type);
void printGateVector(const std::vector<GateType> &gates);

extern const std::map<std::vector<GateType>, SO3Matrix> clifford_group_to_SO3;

extern const std::unordered_map<GateType, std::pair<DyadicMatrix, double>> clifford_gates_to_SU2;

struct ParityTransformInfo {
    SO3Matrix so3_val;
    std::vector<GateType> op_gates;
    double op_phase; // Global phase scaled by 1/pi
};

// Helper function to initialize ParityTransforms
const DyadicMatrix C_DATA({0, 0, 0, 1}, {}, {}, {0, 0, 0, 1});

const DyadicMatrix T_DATA({0, 0, 0, 1}, {}, {}, {-1, 0, 0, 0});

const DyadicMatrix HT_DATA({0, 0, 0, 1}, {0, 0, 0, 1}, {-1, 0, 0, 0}, {1, 0, 0, 0}, 1);

const DyadicMatrix SHT_DATA({0, 0, 0, 1}, {0, -1, 0, 0}, {-1, 0, 0, 0}, {0, 0, 1, 0}, 1);

// --- Step 2: Use these named variables in your map initialization ---

extern const std::map<std::array<int, 3>, ParityTransformInfo> parity_transforms;

std::vector<PPRGateType> HSTtoPPR(const std::vector<GateType> &vector);

} // namespace RSDecomp::CliffordData
