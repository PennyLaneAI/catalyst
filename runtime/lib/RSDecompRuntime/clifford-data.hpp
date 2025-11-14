#pragma once
#include "rings.hpp"
#include <map>
// CHECK ALL VALUES in this file
namespace CliffordData {
enum class GateType { T = 0, HT, SHT, I, X, Y, Z, H, S, Sd };

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

} // namespace CliffordData
