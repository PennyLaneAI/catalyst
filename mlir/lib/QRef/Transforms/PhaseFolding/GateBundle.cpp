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

#include "GateBundle.hpp"

using namespace catalyst::phase_folding;

/*
    Operators:
*/
GateBundle &GateBundle::operator+=(const GateBundle &rhs) {
    zeroAffineGates.reserve(zeroAffineGates.size() + rhs.zeroAffineGates.size());
    zeroAffineGates.insert(zeroAffineGates.end(), rhs.zeroAffineGates.begin(),
                           rhs.zeroAffineGates.end());

    oneAffineGates.reserve(oneAffineGates.size() + rhs.oneAffineGates.size());
    oneAffineGates.insert(oneAffineGates.end(), rhs.oneAffineGates.begin(),
                          rhs.oneAffineGates.end());

    return *this;
}

GateBundle GateBundle::operator+(const GateBundle &rhs) const {
    GateBundle res = *this;
    res += rhs;
    return res;
}

std::string GateBundle::toString() const {
    auto gatesToString = [](const std::vector<GateID> &gates) -> std::string {
        std::string res = ": (";
        for (size_t i = 0; i < gates.size(); ++i) {
            if (i > 0) {
                res += ", ";
            }
            res += std::to_string(gates[i]);
        }
        res += ")";
        return res;
    };

    return "[0" + gatesToString(zeroAffineGates) + " __ 1" + gatesToString(oneAffineGates) + ']';
}

/*
    Methods:
*/
GateID GateBundle::getMergeTarget() const {
    if (!zeroAffineGates.empty()) {
        return zeroAffineGates[0];
    }
    if (!oneAffineGates.empty()) {
        return oneAffineGates[0];
    }
    return -1;
}

/*
    Print:
*/
namespace catalyst::phase_folding {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const std::vector<GateID> &gates) {
    os << ": (";
    for (size_t i = 0; i < gates.size(); i++) {
        if (i > 0) {
            os << ", ";
        }
        os << gates[i];
    }
    os << ")";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const GateBundle &bundle) {
    os << '[';
    os << "0" << bundle.zeroAffineGates;
    os << " __ ";
    os << "1" << bundle.oneAffineGates;
    os << ']';
    return os;
}
} // namespace catalyst::phase_folding
