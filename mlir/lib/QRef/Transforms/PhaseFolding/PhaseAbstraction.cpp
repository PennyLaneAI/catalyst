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

#include "PhaseAbstraction.hpp"

#include "AffineRelation.hpp"
#include "AffineSchema.hpp"

using namespace catalyst::phase_folding;

/*
    Operators:
*/
PhaseAbstraction &PhaseAbstraction::operator+=(const PhaseAbstraction &rhs) {
    addActiveBundlesWith(rhs.activeBundles);
    addOrphanBundlesWith(rhs.orphanBundles);
    return *this;
}

PhaseAbstraction PhaseAbstraction::operator+(const PhaseAbstraction &rhs) const {
    PhaseAbstraction res = *this;
    res += rhs;
    return res;
}

/*
    Insertion:
*/
void PhaseAbstraction::addOrphanBundlesWith(const std::vector<GateBundle> &rhsOrphans) {
    orphanBundles.reserve(orphanBundles.size() + rhsOrphans.size());
    orphanBundles.insert(orphanBundles.end(), rhsOrphans.begin(), rhsOrphans.end());
}

void PhaseAbstraction::addActiveBundlesWith(const llvm::DenseMap<Parity, GateBundle> &rhsActives) {
    if (activeBundles.empty()) {
        activeBundles.reserve(rhsActives.size());
    }

    for (const auto &[parity, contributors] : rhsActives) {
        insertActiveBundle(contributors, parity);
    }
}

void PhaseAbstraction::insertActiveBundle(const GateBundle &contributor, const Parity &parity) {
    auto it = activeBundles.find(parity);
    if (it != activeBundles.end()) {
        it->second += contributor;
    } else {
        activeBundles[parity] = contributor;
    }
}

void PhaseAbstraction::insertContributor(const GateBundle &contributor, const Parity &parity) {
    // if (parity.isUnsat(schema.affVal)) {
    //     orphanBundles.push_back(contributor);
    // } else {
    insertActiveBundle(contributor, parity);
    // }

    // BitLocation affValLoc = schema.affVal;
    // bool affineValue = parity.getBitAtLoc(affValLoc);

    // if (affineValue) {
    //     parity.clearBitAtLoc(affValLoc);
    // }
    // insertActiveBundle(contributor, parity);
    // if (affineValue) {
    //     parity.assignBitAtLoc(affValLoc, affineValue);
    // }
}

/*
    Normalization:
*/
void PhaseAbstraction::normalizeByCond(const AffineRelation &cond, const AffineSchema &paritySchema,
                                       bool isPrecond, bool isProjectOutAuxVars) {
    llvm::DenseMap<Parity, GateBundle> oldBundles;
    std::swap(activeBundles, oldBundles);

    for (auto &[parity, contributors] : oldBundles) {
        Parity reducedPar = cond.reduce(parity, paritySchema, isPrecond, isProjectOutAuxVars);

        BitLocation affValLoc = cond.getSchema().affVal;
        if (reducedPar.getBitAtLoc(affValLoc)) {
            reducedPar.clearBitAtLoc(affValLoc);
            contributors.flipGatesAffineValues();
        }
        insertActiveBundle(contributors, reducedPar);
    }
}

void PhaseAbstraction::nullifyByPrecond(const AffineRelation &precond,
                                        const AffineSchema &paritySchema) {
    normalizeByCond(precond, paritySchema, true);
    orphanNonTrivialBundles();
}

void PhaseAbstraction::normalizeByPostcond(const AffineRelation &postcond,
                                           const AffineSchema &paritySchema) {
    normalizeByCond(postcond, paritySchema, false);
}

void PhaseAbstraction::projectOutAuxVars(const AffineRelation &cond) {
    normalizeByCond(cond, cond.getSchema(), false, true);
    orphanBundlesIf(
        [&](const Parity &parity) { return !parity.isTrivialInRange(cond.getSchema().auxVars); });
}

std::string PhaseAbstraction::toString(const AffineSchema &schema) const {
    std::string res = "";
    for (const auto &[parity, contributors] : activeBundles) {

        std::string pre = parity.toStringWithOrder(schema.preVars);
        std::string aux = parity.toStringWithOrder(schema.auxVars);

        res += (pre + " " + aux + " -> " + contributors.toString() + "\n");
    }

    if (!orphanBundles.empty()) {
        res += "Unsat -> ";
        for (const GateBundle &contributors : orphanBundles) {
            res += contributors.toString() + ", ";
        }
        res += "\n";
    }
    return res;
}

/*
    Print:
*/
namespace catalyst::phase_folding {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhaseAbstraction &pp) {
    for (const auto &[parity, contributors] : pp.activeBundles) {
        os << parity << " -> " << contributors << "\n";
    }
    if (!pp.orphanBundles.empty()) {
        os << "Unsats ->\n";
        for (const GateBundle &contributors : pp.orphanBundles) {
            os << contributors << "\n";
        }
    }
    return os;
}
} // namespace catalyst::phase_folding
