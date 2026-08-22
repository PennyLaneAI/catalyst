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

#include "llvm/Support/raw_ostream.h"

#include "AffineRelation.hpp"
#include "PhaseAbstraction.hpp"

namespace catalyst::phase_folding {

struct ProgramAbstraction;

enum class RegionType { Conditional, Loop, Procedure };

struct RegionSummary {
    PhaseAbstraction phases;
    TransformSchema phasesSchm;

    PhaseAbstraction falseBranchPhases; // for Conditional
    TransformSchema falseBranchPhasesSchm;

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
    void summarizeCond(ProgramAbstraction *elseBody = nullptr);
    void summarizeLoop();
    void summarizeProc();
};

// I'm thinking of keeping all phases, and nullifying at the and of all nested blocks. think about
// multiple blocks?

} // namespace catalyst::phase_folding
