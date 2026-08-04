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
}

void PhaseAbstraction::orphanNonTrivialBundles()
{
    for (auto &[parity, contributors] : activeBundles) {
        if (!parity.isTrivial()) {
            orphanBundles.push_back(std::move(contributors));
            activeBundles.erase(parity);
        }
    }
}

void PhaseAbstraction::nullifyByPrecond(const AffineRelation& precond, const AffineSchema& paritySchema)
{
    llvm::DenseMap<Parity, GateBundle> trivialCondBundles;
    GateBundle &trivContributors = trivialCondBundles[Parity::Trivial(precond.numQubits())];
        
    for (auto &[parity, contributors] : activeBundles) {
        if (precond.reduce(parity, paritySchema).isTrivial()) { // might become unsat instead of trivial; in that case, we should look at affVal 1 gates, they become 0!
            trivContributors += std::move(contributors);
        } else {
            orphanBundles.push_back(std::move(contributors));
        }
    }

    if (trivContributors.gateCount() > 0) {
        activeBundles = std::move(trivialCondBundles);
    } else {
        activeBundles.clear();
    }
}

void PhaseAbstraction::normalizeByPostcond(const AffineRelation& postcond, const AffineSchema& paritySchema)
{
    // if it was empty initially, if result became unsat,...
    llvm::DenseMap<Parity, GateBundle> normalizedBundles;

    // llvm::errs() << "\normalizeByPostcond:\n";
    // llvm::errs() << "current phases:\n" << *this << "\n";

    for (auto &[parity, contributors] : activeBundles) {
        // llvm::errs() << "original parity: " << parity << "\n";
        Parity reducedPar = postcond.reduce(parity, paritySchema);
        // llvm::errs() << "reduced parity: " << reducedPar << "\n";
        normalizedBundles[reducedPar] = std::move(contributors);
    }

    activeBundles = std::move(normalizedBundles);
    // llvm::errs() << "%%%%%%%%%%%%%%%%%%%%\n";
}

void PhaseAbstraction::reSchema(const AffineSchema &oldSchm, const AffineSchema &newSchm)
{
    llvm::DenseMap<Parity, GateBundle> newActiveBundles;
    Parity newPar(newSchm.maxBlock());

    for (auto &[oldPar, contributors] : activeBundles) {
        newPar.reset();
        newPar.mapBitsFrom(oldPar, oldSchm.preVars, newSchm.preVars);
        newPar.mapBitsFrom(oldPar, oldSchm.auxVars, newSchm.auxVars);
        newPar.mapBitFrom(oldPar, oldSchm.affVal, newSchm.affVal);

        newActiveBundles[newPar] = contributors;
    }
    activeBundles = std::move(newActiveBundles);
}
