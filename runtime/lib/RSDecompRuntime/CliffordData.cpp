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
#include "Exception.hpp"

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

std::vector<PPRGateType> HSTtoPPR(const std::vector<GateType> &input_gates)
{
    std::vector<PPRGateType> output_gates;
    output_gates.reserve(input_gates.size());

    size_t i = 0;
    while (i < input_gates.size()) {
        const GateType &current_gate = input_gates[i];
        if ((current_gate == GateType::HT || current_gate == GateType::SHT) &&
            (i + 1 < input_gates.size())) {
            const GateType &next_gate = input_gates[i + 1];

            // Rule: HT, HT -> X8, Z8
            if (current_gate == GateType::HT && next_gate == GateType::HT) {
                output_gates.push_back(PPRGateType::X8);
                output_gates.push_back(PPRGateType::Z8);
                i += 2; // Skip both processed gates
                continue;
            }

            // Rule: HT, SHT -> X4, X8, Z8
            if (current_gate == GateType::HT && next_gate == GateType::SHT) {
                output_gates.push_back(PPRGateType::X4);
                output_gates.push_back(PPRGateType::X8);
                output_gates.push_back(PPRGateType::Z8);
                i += 2;
                continue;
            }

            // Rule: SHT, HT -> Z4, X8, Z8
            if (current_gate == GateType::SHT && next_gate == GateType::HT) {
                output_gates.push_back(PPRGateType::Z4);
                output_gates.push_back(PPRGateType::X8);
                output_gates.push_back(PPRGateType::Z8);
                i += 2;
                continue;
            }

            // Rule: SHT, SHT -> Z4, X4, X8, Z8
            if (current_gate == GateType::SHT && next_gate == GateType::SHT) {
                output_gates.push_back(PPRGateType::Z4);
                output_gates.push_back(PPRGateType::X4);
                output_gates.push_back(PPRGateType::X8);
                output_gates.push_back(PPRGateType::Z8);
                i += 2;
                continue;
            }
        }

        // If we're here, no pair rule was matched.
        // We handle the 1-to-1 mappings.
        switch (current_gate) {
        case GateType::T:
            output_gates.push_back(PPRGateType::Z8);
            break;
        case GateType::I:
            output_gates.push_back(PPRGateType::I);
            break;
        case GateType::X:
            output_gates.push_back(PPRGateType::X2);
            break;
        case GateType::Y:
            output_gates.push_back(PPRGateType::Y2);
            break;
        case GateType::Z:
            output_gates.push_back(PPRGateType::Z2);
            break;
        case GateType::H:
            output_gates.push_back(PPRGateType::Z4);
            output_gates.push_back(PPRGateType::X4);
            output_gates.push_back(PPRGateType::Z4);
            break;
        case GateType::S:
            output_gates.push_back(PPRGateType::Z4);
            break;
        case GateType::Sd:
            output_gates.push_back(PPRGateType::adjZ4);
            break;
        case GateType::HT: {
            output_gates.push_back(PPRGateType::X8);
            output_gates.push_back(PPRGateType::Z4);
            output_gates.push_back(PPRGateType::X4);
            output_gates.push_back(PPRGateType::Z4);
            break;
        }
        case GateType::SHT: {
            output_gates.push_back(PPRGateType::adjY8);
            output_gates.push_back(PPRGateType::adjX4);
            output_gates.push_back(PPRGateType::Z4);
            output_gates.push_back(PPRGateType::Z2);
            break;
        }

        default:
            RT_FAIL("Unknown GateType encountered.");
        }

        i += 1; // Skip the single processed gate
    }

    return output_gates;
}

} // namespace RSDecomp::CliffordData
