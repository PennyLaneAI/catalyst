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

/*
    Operators:
*/
PhaseAbstraction &PhaseAbstraction::operator+=(const PhaseAbstraction &rhs)
{
    addActiveBundlesWith(rhs.activeBundles);
    addOrphanBundlesWith(rhs.orphanBundles);
    return *this;
}

PhaseAbstraction PhaseAbstraction::operator+(const PhaseAbstraction &rhs) const
{
    PhaseAbstraction res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhaseAbstraction &pp)
{
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

/*
    Methods:
*/
void PhaseAbstraction::addOrphanBundlesWith(const std::vector<GateBundle> &rhsOrphans)
{
    orphanBundles.reserve(orphanBundles.size() + rhsOrphans.size());
    orphanBundles.insert(orphanBundles.end(), rhsOrphans.begin(), rhsOrphans.end());
}

void PhaseAbstraction::addActiveBundlesWith(const llvm::DenseMap<Parity, GateBundle> &rhsActives)
{
    if (activeBundles.empty())  activeBundles.reserve(rhsActives.size());
    
    for (const auto &[parity, contributors] : rhsActives) {
        insertActiveBundle(contributors, parity);
    }
}

void PhaseAbstraction::insertActiveBundle(const GateBundle &contributor, const Parity &parity)
{
    auto it = activeBundles.find(parity);
    if (it != activeBundles.end()) {
        it->second += contributor;
    }
    else {
        activeBundles[parity] = contributor;
    }
}

void PhaseAbstraction::insertContributor(const GateBundle &contributor, const Parity &parity)
{
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

void PhaseAbstraction::normalizeByCond(const AffineRelation& cond, const AffineSchema& paritySchema, bool isPrecond, bool isProjectOutAuxVars)
{
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
