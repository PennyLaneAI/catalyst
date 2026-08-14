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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "GateBundle.hpp"
#include "Parity.hpp"

namespace catalyst::phase_folding {

class AffineRelation;
struct AffineSchema;

struct PhaseAbstraction {
    llvm::DenseMap<Parity, GateBundle> activeBundles;
    std::vector<GateBundle> orphanBundles;

    // Constructors
    PhaseAbstraction() = default;

    // Operators
    PhaseAbstraction &operator+=(const PhaseAbstraction &rhs);
    PhaseAbstraction operator+(const PhaseAbstraction &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhaseAbstraction &pp);
    // template <typename ColOrderRange>
    // std::string toString(ColOrderRange colOrder) const;
    std::string toString(const AffineSchema &schema) const;

    // Methods
    void insertContributor(const GateBundle &contributor, const Parity &parity);
    void orphanNonTrivialBundles();
    void nullifyByPrecond(const AffineRelation &precond, const AffineSchema &paritySchema);
    void normalizeByPostcond(const AffineRelation &postcond, const AffineSchema &paritySchema);
    void projectOutAuxVars(const AffineRelation &cond);

  private:
    void addActiveBundlesWith(const llvm::DenseMap<Parity, GateBundle> &rhsActives);
    void addOrphanBundlesWith(const std::vector<GateBundle> &rhsOrphans);
    void insertActiveBundle(const GateBundle &contributor, const Parity &parity);
    void normalizeByCond(const AffineRelation &cond, const AffineSchema &paritySchema,
                         bool isPrecond, bool isProjectOutAuxVars = false);

    template <typename Predicate> void orphanBundlesIf(Predicate cond);
};

// template <typename ColOrderRange>
// std::string PhaseAbstraction::toString(ColOrderRange colOrder) const
// {
//     std::string res = "";
//     for (const auto &[parity, contributors] : activeBundles) {
//         res += (parity.toStringWithOrder(colOrder) + " -> " + contributors.toString() + "\n");
//     }

//     if (!orphanBundles.empty()) {
//         res += "Unsat -> ";
//         for (const GateBundle &contributors : orphanBundles) {
//             res += contributors.toString() + ", ";
//         }
//         res += "\n";
//     }
//     return res;
// }

template <typename Predicate> void PhaseAbstraction::orphanBundlesIf(Predicate cond) {
    for (auto &[parity, contributors] : activeBundles) {
        if (cond(parity)) {
            orphanBundles.push_back(std::move(contributors));
            activeBundles.erase(parity);
        }
    }
}

inline void PhaseAbstraction::orphanNonTrivialBundles() {
    orphanBundlesIf([&](const Parity &parity) { return !parity.isTrivial(); });
}

// in all these normalizations, I should be able to create the constraint matrix and toREF() it,
// then call reduce function for all parities on this same matrix. won't I?

} // namespace catalyst::phase_folding
