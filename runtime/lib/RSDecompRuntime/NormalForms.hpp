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

#include "CliffordData.hpp"
#include "Rings.hpp"

namespace RSDecomp::NormalForms {
using namespace RSDecomp::Rings;
using RSDecomp::CliffordData::GateType;

/**
 * @brief Decompose an SO(3) matrix into Matsumoto-Amano normal form.
 *
 *  A Matsumoto-Amano normal form - (T | ε) (HT | SHT)^* C, consists of a rightmost
    Clifford operator, followed by any number of syllables of the form HT or SHT, followed by an
    optional syllable T arXiv:1312.6584.
 */
std::pair<std::vector<GateType>, double> ma_normal_form(const SO3Matrix &op)
{
    ZOmega a{0, 0, 0, 1};
    ZOmega c{0, 0, 0, 0};
    INT_TYPE k = 0;

    SO3Matrix so3_op = op;

    std::vector<GateType> decomposition;
    double g_phase = 0.0;

    auto parity_vec = so3_op.parity_vec();

    while (parity_vec != std::array<int, 3>{1, 1, 1}) {
        const auto &[so3_val, op_gate, op_phase] = CliffordData::parity_transforms.at(parity_vec);
        so3_op = so3_matrix_mul(so3_val, so3_op);

        decomposition.insert(decomposition.end(), op_gate.begin(), op_gate.end());
        g_phase += op_phase;
        if (parity_vec == std::array<int, 3>{2, 2, 0}) {
            // T
            c = c * ZOmega(0, 0, 1, 0);
        }
        else if (parity_vec == std::array<int, 3>{0, 2, 2}) {
            // HT
            auto a_temp = ZOmega(0, -1, 0, 0) * (a + c);
            auto c_temp = ZOmega(-1, 0, 0, 0) * (a - c);
            a = a_temp;
            c = c_temp;
            k += 1;
        }
        else {
            // SHT
            ZOmega ic = ZOmega(0, 1, 0, 0) * c;
            auto a_temp = ZOmega(0, 0, -1, 0) * (a + ic);
            auto c_temp = ZOmega(0, -1, 0, 0) * (a - ic);
            a = a_temp;
            c = c_temp;
            k += 1;
        }

        parity_vec = so3_op.parity_vec();
    }
    for (const auto &[clifford_ops, clifford_so3] : CliffordData::clifford_group_to_SO3) {
        if (clifford_so3 == so3_op) {
            for (const auto &clf_op : clifford_ops) {
                decomposition.emplace_back(clf_op);
                auto [su2, gp] = CliffordData::clifford_gates_to_SU2.at(clf_op);
                a = su2.a * a + su2.b * c;
                c = su2.c * a + su2.d * c;
                k += su2.k;
                g_phase -= gp;
            }
            break;
        }
    }

    auto su2mat = op.dyadic_mat;
    auto g_angle = -std::arg(su2mat.a.to_complex() / a.to_complex());
    g_phase = g_angle / M_PI - g_phase;

    g_phase = std::fmod(g_phase, 2.0);
    if (g_phase < 0.0) {
        g_phase += 2.0;
    }
    g_phase *= M_PI;

    return {decomposition, g_phase};
}

} // namespace RSDecomp::NormalForms
