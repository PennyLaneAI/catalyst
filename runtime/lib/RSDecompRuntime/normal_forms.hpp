#include "clifford-data.hpp"
#include "rings.hpp"
#include <complex>
namespace normal_forms {

using CliffordData::GateType;

std::pair<std::vector<GateType>, double> ma_normal_form(SO3Matrix &op)
{
    ZOmega a{0, 0, 0, 1};
    ZOmega c{0, 0, 0, 0};
    INT_TYPE k = 0;

    SO3Matrix so3_op = op;

    std::vector<GateType> decomposition;
    double g_phase = 0.0;

    auto parity_vec = so3_op.parity_vec();

    while (parity_vec != std::array<int, 3>{1, 1, 1}) {
        CliffordData::ParityTransformInfo pt_info = CliffordData::parity_transforms.at(parity_vec);
        auto so3_val = pt_info.so3_val;
        auto op_gate = pt_info.op_gates;
        auto op_phase = pt_info.op_phase;
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
                decomposition.push_back(clf_op);
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
    auto g_angle =
        -std::arg(su2mat.a.to_complex() / a.to_complex() * std::pow(M_SQRT2, k - su2mat.k));
    g_phase = g_angle / M_PI - g_phase;

    g_phase = std::fmod(g_phase, 2.0);
    if (g_phase < 0.0) {
        g_phase += 2.0;
    }
    g_phase *= M_PI;

    return {decomposition, g_phase};
}

} // namespace normal_forms
