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

/*
    Constructors:
*/
RegionSummary::RegionSummary(RegionType type, ProgramAbstraction &circ1, ProgramAbstraction *circ2)
{
    TransformSchema circ1Schm;
    if (type != RegionType::Procedure) {
        circ1Schm = circ1.getSchema();
    }
        
    this->affineRel = AffineRelation(std::move(circ1.stateTransform));
    this->phases = std::move(circ1.phases);
    this->type = type;

    switch (type) {
    case RegionType::Conditional:
        if (circ2){
            summarizeIfElse(circ1Schm, *circ2);
        } else {
            summarizeIf(circ1Schm);
        }
        break;
    case RegionType::Loop:
        summarizeLoop(circ1Schm);
        break;
    case RegionType::Procedure:
        summarizeProc();
        break;
    }
}

void RegionSummary::summarizeIfElse(const TransformSchema &ifBodySchm, ProgramAbstraction &elseBody)
{
    TransformSchema elseSchm = elseBody.getSchema();
    AffineRelation elseRel(std::move(elseBody.stateTransform));
    
    summarizeCond(ifBodySchm, elseRel);
    
    elseBody.phases.reSchema(elseSchm, affineRel.getSchema());
    this->falseBranchPhases = std::move(elseBody.phases);
}

/*
    Print:
*/
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const RegionSummary &sum)
{
    auto colOrd = sum.affineRel.getSchema().getOrder();

    os << ".Phase abstraction:\n" << sum.phases.toString(colOrd);
    if (sum.type == RegionType::Conditional) {
        os << "--\n" << sum.falseBranchPhases.toString(colOrd);
    }
    os << ".Affine relation:\n" << sum.affineRel;
    return os;
}

/*
    Methods:
*/
void RegionSummary::nullifyPhasesUnder(const AffineRelation &precondRel)
{
    phases.nullifyByPrecond(precondRel, affineRel.getSchema());

    if (type == RegionType::Conditional) {
        falseBranchPhases.nullifyByPrecond(precondRel, affineRel.getSchema());
    }
}

void RegionSummary::accumulatePhasesInto(PhaseAbstraction &trgtPhases, const TransformSchema &trgtSchm)
{
    // phases.reSchema(affineRel.getSchema(), trgtSchm);   // is it necessary? no, bc all phasee are either 0 or orphaned
    trgtPhases += phases;

    if (type == RegionType::Conditional) {
        // falseBranchPhases.reSchema(affineRel.getSchema(), trgtSchm);
        trgtPhases += falseBranchPhases;
    }
}   // += ops could become more optimized by actually consuming the phases
