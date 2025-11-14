#include "clifford-data.hpp"
// CHECK ALL VALUES in this file

// Overload the << operator for std::ostream.
// This is the idiomatic C++ way to make your custom types printable.
namespace CliffordData {

std::ostream &operator<<(std::ostream &os, GateType type)
{
    os << gateTypeToString(type);
    return os;
}

std::string_view gateTypeToString(GateType type)
{
    switch (type) {
    case GateType::I:
        return "I";
    case GateType::X:
        return "X";
    case GateType::Y:
        return "Y";
    case GateType::Z:
        return "Z";
    case GateType::H:
        return "H";
    case GateType::S:
        return "S";
    case GateType::Sd:
        return "Sd";
    case GateType::T:
        return "T";
    case GateType::HT:
        return "H, T";
    case GateType::SHT:
        return "S, H, T";
    default:
        return "Unknown";
    }
}

// The function to print your vector
void printGateVector(const std::vector<GateType> &gates)
{
    std::cout << "[";
    // Loop through the vector, printing each element
    for (size_t i = 0; i < gates.size(); ++i) {
        // Use our overloaded << operator
        std::cout << gates[i];

        // Add a comma and space, but not after the last element
        if (i < gates.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

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
    {{1, 1, 1},
     {SO3Matrix(C_DATA), // Much cleaner and unambiguous
      {I},
      0.0}},
    // "T"
    {{2, 2, 0}, {SO3Matrix(T_DATA), {T}, 0.0}},
    // "HT"
    {{0, 2, 2}, {SO3Matrix(HT_DATA), {HT}, 1.5}},
    // "SHT"
    {{2, 0, 2}, {SO3Matrix(SHT_DATA), {SHT}, 1.25}}};

} // namespace CliffordData
