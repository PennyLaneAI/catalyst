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
        // std::cout << "Current c = " << c << ", a = " << a << ", k = " << k << std::endl;
        // std::cout << "Before parity vec: ";
        // for (const auto vec_data : parity_vec) {
        //     std::cout << vec_data << " ";
        // }

        // std::cout << std::endl;
        CliffordData::ParityTransformInfo pt_info = CliffordData::parity_transforms.at(parity_vec);
        auto so3_val = pt_info.so3_val;
        // std::cout << "so3_val: " << so3_val << std::endl;
        auto op_gate = pt_info.op_gates;
        auto op_phase = pt_info.op_phase;
        so3_op = so3_matrix_mul(so3_val, so3_op);
        // std::cout << "After SO3: " << so3_op << std::endl;

        decomposition.insert(decomposition.end(), op_gate.begin(), op_gate.end());
        g_phase += op_phase;
        // std::cout << "HERE A" << std::endl;
        if (parity_vec == std::array<int, 3>{2, 2, 0}) {
            // T
            // std::cout << "Overflow investigation 1" << std::endl;
            c = c * ZOmega(0, 0, 1, 0);
        }
        else if (parity_vec == std::array<int, 3>{0, 2, 2}) {
            // HT
            // std::cout << "Overflow investigation 2" << std::endl;
            auto a_temp = ZOmega(0, -1, 0, 0) * (a + c);
            auto c_temp = ZOmega(-1, 0, 0, 0) * (a - c);
            a = a_temp;
            c = c_temp;
            // std::cout << "Overflow investigation 3" << std::endl;
            k += 1;
        }
        else {
            // SHT
            // std::cout << "Overflow investigation 4" << std::endl;
            ZOmega ic = ZOmega(0, 1, 0, 0) * c;
            // std::cout << "Overflow investigation 5" << std::endl;
            auto a_temp = ZOmega(0, 0, -1, 0) * (a + ic);
            // std::cout << "Overflow investigation 6" << std::endl;
            auto c_temp = ZOmega(0, -1, 0, 0) * (a - ic);
            a = a_temp;
            c = c_temp;
            k += 1;
        }
        // std::cout << "HERE B" << std::endl;

        parity_vec = so3_op.parity_vec();
        // std::cout << "HERE C" << std::endl;
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

    auto su2mat = so3_op.dyadic_mat;
    double g_angle = -static_cast<double>(std::arg(su2mat.a.to_complex() / a.to_complex()));
    g_angle -= M_PI_4 * (k - su2mat.k);
    g_phase = g_angle / M_PI - g_phase;

    return {decomposition, g_phase};
}

} // namespace normal_forms
