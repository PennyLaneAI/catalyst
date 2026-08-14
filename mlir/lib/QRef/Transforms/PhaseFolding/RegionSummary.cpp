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

#include "RegionSummary.hpp"

#include "ProgramAbstraction.hpp"

using namespace catalyst::phase_folding;

/*
    Constructors:
*/
RegionSummary::RegionSummary(RegionType type, ProgramAbstraction &circ1,
                             ProgramAbstraction *circ2) {
    this->phasesSchm = circ1.getSchema();
    this->phases = std::move(circ1.phases);
    this->affineRel = AffineRelation(std::move(circ1.stateTransform));
    phases.projectOutAuxVars(affineRel);
    this->type = type;

    switch (type) {
    case RegionType::Conditional:
        summarizeCond(circ2);
        break;
    case RegionType::Loop:
        summarizeLoop();
        break;
    case RegionType::Procedure:
        summarizeProc();
        break;
    }
}

void RegionSummary::summarizeCond(ProgramAbstraction *elseBody) {
    AffineRelation elseRel;
    if (elseBody) {
        this->falseBranchPhasesSchm = elseBody->getSchema();
        this->falseBranchPhases = std::move(elseBody->phases);
        elseRel = AffineRelation(std::move(elseBody->stateTransform));
    } else {
        elseRel = AffineRelation::Identity(phasesSchm.numQubits());
    }
    affineRel.joinWith(elseRel);
}

void RegionSummary::summarizeLoop() { affineRel.applyKleeneStar(); }

void RegionSummary::summarizeProc() {
    llvm::errs() << "summarizeProc...\n";
    affineRel.projectOutAuxVars();
    // llvm::errs() << "affineRel:\n" << affineRel << "\n";
    phases.orphanNonTrivialBundles();
    // llvm::errs() << "phases:\n" << phases << "\n";
}

/*
    Methods:
*/
void RegionSummary::nullifyPhasesUnder(const AffineRelation &precondRel) {
    phases.nullifyByPrecond(precondRel, phasesSchm);

    if (type == RegionType::Conditional) {
        falseBranchPhases.nullifyByPrecond(precondRel, falseBranchPhasesSchm);
    }
}

void RegionSummary::accumulatePhasesInto(PhaseAbstraction &trgtPhases,
                                         const TransformSchema &trgtSchm) {
    trgtPhases += phases;

    if (type == RegionType::Conditional) {
        trgtPhases += falseBranchPhases;
    }
} // += ops could become more optimized by actually consuming the phases

/*
    Print:
*/
namespace catalyst::phase_folding {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const RegionSummary &sum) {
    os << ".Phase abstraction:\n" << sum.phases.toString(sum.affineRel.getSchema());
    if (sum.type == RegionType::Conditional) {
        os << "--\n" << sum.falseBranchPhases.toString(sum.affineRel.getSchema());
    }
    os << ".Affine relation:\n" << sum.affineRel;
    return os;
}
} // namespace catalyst::phase_folding
