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

#include "AffineRelation.hpp"
#include "PhaseAbstraction.hpp"

#include "llvm/Support/raw_ostream.h"

struct ProgramAbstraction;

enum class RegionType { Conditional, Loop, Procedure };

struct RegionSummary {
    PhaseAbstraction phases;
    PhaseAbstraction falseBranchPhases; // for Conditional
    AffineRelation affineRel;
    RegionType type;

    // Constructors
    RegionSummary() = default;
    RegionSummary(RegionType type, ProgramAbstraction &circ1, ProgramAbstraction *circ2 = nullptr);
    
    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const RegionSummary &sum);

    // Methods
    void nullifyPhasesUnder(const AffineRelation &precondRel);
    void accumulatePhasesInto(PhaseAbstraction &trgtPhases, const TransformSchema &trgtSchm);

  private:
    void summarizeIf(const TransformSchema &ifBodySchm);
    void summarizeIfElse(const TransformSchema &ifBodySchm, ProgramAbstraction &elseBody);
    void summarizeCond(const TransformSchema &ifBodySchm, const AffineRelation &elseRel);
    void summarizeLoop(const TransformSchema &bodySchm);
    void summarizeProc();
};

inline void RegionSummary::summarizeIf(const TransformSchema &ifBodySchm)
{
    summarizeCond(ifBodySchm, AffineRelation::Identity(ifBodySchm.numQubits()));
}

inline void RegionSummary::summarizeCond(const TransformSchema &ifBodySchm, const AffineRelation &elseRel)
{
    // llvm::errs() << "thenRel:\n" << affineRel << "\n";
    // llvm::errs() << "elseRel:\n" << elseRel << "\n";

    affineRel.joinWith(elseRel);
    // llvm::errs() << "joinedRel:\n" << affineRel << "\n";
    phases.reSchema(ifBodySchm, affineRel.getSchema());
}

inline void RegionSummary::summarizeLoop(const TransformSchema &bodySchm)
{
    affineRel.applyKleeneStar();
    phases.reSchema(bodySchm, affineRel.getSchema());
}

inline void RegionSummary::summarizeProc()
{
    affineRel.projectOutAuxVars();
    phases.orphanAllBundles();
}
