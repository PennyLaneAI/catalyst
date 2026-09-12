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

#include "ProgramAbstraction.hpp"

#include "AffineRelation.hpp"
#include "RegionSummary.hpp"

using namespace catalyst::phase_folding;

/*
    Methods:
*/
bool ProgramAbstraction::areWiresInBound(llvm::ArrayRef<size_t> wires) {
    size_t maxWire = 0;
    for (size_t wire : wires) {
        if (wire > maxWire) {
            maxWire = wire;
        }
    }
    if (maxWire >= numQubits()) {
        extendQubitsBy(maxWire - numQubits() + 1);
    }
    return (maxWire < numQubits());
}

void ProgramAbstraction::applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> wires,
                                   std::optional<GateID> gateId) {
    assert(areWiresInBound(wires));
    assert(arity(gate) == DYNAMIC_ARITY || wires.size() == arity(gate));
    assert(!isPhaseGate(gate) || gateId.has_value());

    switch (gate) {
    case Gate::I:
        break;
    case Gate::H:
        stateTransform.applyGateH(wires[0]);
        break;
    case Gate::X:
        stateTransform.applyGateX(wires[0]);
        break;
    case Gate::Y:
        if (isAdjoint) {
            applyGateY_dag(wires[0], gateId.value());
        } else {
            applyGateY(wires[0], gateId.value());
        }
        break;
    case Gate::Z:
    case Gate::S:
    case Gate::T:
    case Gate::RZ:
        applyGateRZ(wires[0], gateId.value());
        break;
    case Gate::CNOT:
        stateTransform.applyGateCNOT(wires[0], wires[1]);
        break;
    case Gate::SWAP:
        stateTransform.applyGateSWAP(wires[0], wires[1]);
        break;
    case Gate::U:
        stateTransform.applyGateU(wires);
        break;
    case Gate::GP:
        break; // figure out later.
    }
}

void ProgramAbstraction::applyGateRZ(size_t wire, GateID gateId) {
    Parity &parity = stateTransform.getExprMutable(wire);
    BitLocation affValLoc = getSchema().affVal;
    bool affineVal = parity.getBitAtLoc(affValLoc);
    GateBundle contributor(gateId, affineVal);

    parity.clearBitAtLoc(affValLoc);
    phases.insertContributor(contributor, parity);
    parity.assignBitAtLoc(affValLoc, affineVal);
}

void ProgramAbstraction::applyGateY(size_t wire, GateID gateId) {
    stateTransform.applyGateX(wire);
    applyGateRZ(wire, gateId);
    // global phase of +i.
}

void ProgramAbstraction::applyGateY_dag(size_t wire, GateID gateId) {
    applyGateRZ(wire, gateId);
    stateTransform.applyGateX(wire);
    // global phase of -i.
}

void ProgramAbstraction::applySummary(RegionSummary &&summary) {
    // llvm::errs() << "heyyyyyyyyyyyy\n*************\n";

    // llvm::errs() << "\nProgramAbstraction::applySummary...\n";
    // llvm::errs() << "currentAbstraction:\n" << *this << "\n";
    // llvm::errs() << "summary:\n" << summary << "\n";

    AffineRelation precondition(std::move(stateTransform));
    // llvm::errs() << "precondition:\n" << precondition << "\n";

    summary.nullifyPhasesUnder(precondition);
    // llvm::errs() << "\nnullifiedSummary:\n" << summary << "\n";

    precondition.propagateThrough(summary.affineRel);
    // llvm::errs() << "\npropagatedThrough:\n" << precondition << "\n";

    this->normalizePhasesUnder(precondition);
    // llvm::errs() << "\nnormalizedPhases:\n" << phases << "\n";

    this->stateTransform = precondition.solveRelation();
    // llvm::errs() << "\nsolvedRelation:\n" << stateTransform << "\n";

    summary.accumulatePhasesInto(this->phases, getSchema());
    // llvm::errs() << "\naccumulatedPhases:\n" << phases << "\n";

    // llvm::errs() << "\nfinalAbstraction:\n" << *this << "\n";
}

inline void ProgramAbstraction::normalizePhasesUnder(const AffineRelation &postcondRel) {
    phases.normalizeByPostcond(postcondRel, getSchema());
}

/*
    Print:
*/
namespace catalyst::phase_folding {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ProgramAbstraction &progAbs) {
    os << ".Phase abstraction:\n" << progAbs.phases.toString(progAbs.getSchema());
    os << ".Affine transformation:\n" << progAbs.stateTransform;
    return os;
}
} // namespace catalyst::phase_folding