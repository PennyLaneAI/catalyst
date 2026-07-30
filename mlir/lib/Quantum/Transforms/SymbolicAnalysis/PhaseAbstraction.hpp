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

#include <utility>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "Parity.hpp"
#include "GateBundle.hpp"
#include "AffineRelation.hpp"

struct PhaseAbstraction {
    llvm::DenseMap<Parity, GateBundle> activeBundles;
    std::vector<GateBundle> orphanBundles;

    // Constructors
    PhaseAbstraction() = default;
    
    // Operators
    PhaseAbstraction &operator+=(const PhaseAbstraction &rhs);
    PhaseAbstraction operator+(const PhaseAbstraction &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhaseAbstraction &pp);
    template <typename ColOrderRange>
    std::string toString(ColOrderRange colOrder) const;

    // Methods
    void insertContributor(const GateBundle &contributor, const Parity &parity);
    void orphanAllBundles();
    void nullifyByPrecond(const AffineRelation& precond, const AffineSchema& paritySchema);
    void reSchema(const AffineSchema &oldSchm, const AffineSchema &newSchm);

  private:
    void addActiveBundlesWith(const llvm::DenseMap<Parity, GateBundle> &rhsActives);
    void addOrphanBundlesWith(const std::vector<GateBundle> &rhsOrphans);
    void insertActiveBundle(const GateBundle &contributor, const Parity &parity);
};

template <typename ColOrderRange>
std::string PhaseAbstraction::toString(ColOrderRange colOrder) const
{
    std::string res = "";
    for (const auto &[parity, contributors] : activeBundles) {
        res += (parity.toStringWithOrder(colOrder) + " -> " + contributors.toString() + "\n");
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
