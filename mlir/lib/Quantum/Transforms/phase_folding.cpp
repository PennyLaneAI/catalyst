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

// This algorithm is taken from https://arxiv.org/pdf/1303.2042

#define DEBUG_TYPE "phase-folding"

#include <cassert>
#include <cmath> // std::abs
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // arith::ConstantOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h" // mlir::matchPattern, mlir::m_Constant
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

#include "SymbolicAnalysis/Gate.h"
#include "SymbolicAnalysis/SymbolicCircuit.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_PHASEFOLDINGPASS
#define GEN_PASS_DEF_PHASEFOLDINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct PhaseFoldingPass : impl::PhaseFoldingPassBase<PhaseFoldingPass> {
    using PhaseFoldingPassBase::PhaseFoldingPassBase;

    llvm::DenseMap<mlir::Value, size_t> ssaToWireMap;
    llvm::DenseMap<mlir::Value, size_t> qregToBaseMap;

    std::vector<CustomOp> phaseOps;

    // Gate Statistics:
    void updateStats(Gate gate, int incr) { insertedGateCount[static_cast<size_t>(gate)] += incr; }

    void reportStats()
    {
        llvm::outs() << "Stats:\n";
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if (insertedGateCount[i] != 0) {
                llvm::outs() << GATE_NAME[i] << ": initial-> " << initialGateCount[i]
                             << ",  final-> " << (initialGateCount[i] + insertedGateCount[i])
                             << ". difference-> " << insertedGateCount[i] << "\n";
            }
        }
        llvm::outs() << "\n";
    }

    // Qubit Extraction:
    void allocateRegister(mlir::Value qreg, auto regSize, SymbolicCircuit &symCirc)
    {
        qregToBaseMap[qreg] = symCirc.qubitNum;
        symCirc.extendQubitsBy(static_cast<size_t>(regSize.value_or(0)));
    }

    void allocateQubit(mlir::Value qubit, SymbolicCircuit &symCirc)
    {
        ssaToWireMap[qubit] = symCirc.qubitNum;
        symCirc.extendQubitsBy(1);
    }

    void extractFromQreg(ExtractOp extractOp)
    {
        mlir::Value qreg = extractOp.getQreg();
        auto regIt = qregToBaseMap.find(qreg);
        if (regIt == qregToBaseMap.end()) {
            llvm::errs() << "Error: ExtractOp references an untracked register.\n";
            assert(false);
        }
        size_t baseIndex = regIt->second;

        auto staticIdx = extractOp.getIdxAttr();
        if (!staticIdx.has_value()) {
            // auto dynamicIdx = extractOp.getIdx();
            llvm::errs() << "Error: Dynamic qubit extraction indices are not supported.\n";
            assert(false);
        }
        ssaToWireMap[extractOp.getQubit()] = baseIndex + static_cast<size_t>(staticIdx.value_or(0));
    }

    mlir::DenseElementsAttr extractBasisState(SetBasisStateOp basisOp)
    {
        mlir::Value basisStateTensor = basisOp.getBasisState();
        mlir::Operation *defOp = basisStateTensor.getDefiningOp();

        if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(defOp)) {
            if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
                return denseAttr;
            }
            else {
                llvm::errs() << "Error: Basis state constant is not a DenseElementsAttr.\n";
                assert(false);
            }
        }
        else {
            llvm::errs() << "Error: Dynamic basis state initialization is not supported.\n";
            assert(false);
        }
    }

    void initQubitsState(SetBasisStateOp basisOp, SymbolicCircuit &symCirc)
    {
        llvm::SmallVector<size_t, 4> qubitIndices =
            getQubitIndices(basisOp.getInQubits(), basisOp.getOutQubits());
        mlir::DenseElementsAttr basisState = extractBasisState(basisOp);

        assert(static_cast<size_t>(basisState.getNumElements()) == qubitIndices.size());

        size_t i = 0;
        for (const llvm::APInt &val : basisState.getValues<llvm::APInt>()) {
            symCirc.initQubit(qubitIndices[i], val.getBoolValue());
            i++;
        }
    }

    void applyUndefinedOp(Operation *op, SymbolicCircuit &symCirc)
    {
        llvm::SmallVector<size_t, 4> qubitIndices;
        if (auto qGate = dyn_cast<QuantumGate>(op)) {
            qubitIndices =
                getQubitIndices(qGate.getNonCtrlQubitOperands(), qGate.getNonCtrlQubitResults());
        }
        else if (auto stateOp = dyn_cast<SetStateOp>(op)) {
            qubitIndices = getQubitIndices(stateOp.getInQubits(), stateOp.getOutQubits());
        }
        else {
            return;
        }

        Gate gate = (isa<MultiRZOp>(op) || isa<PCPhaseOp>(op)) ? Gate::I : Gate::U;
        symCirc.applyGate(gate, false, qubitIndices);
    }

    llvm::SmallVector<size_t, 4> getQubitIndices(mlir::ValueRange ins, mlir::ResultRange outs)
    {
        assert(ins.size() == outs.size());
        size_t n = ins.size();

        llvm::SmallVector<size_t, 4> indices;
        indices.reserve(n);

        for (size_t i = 0; i < n; i++) {
            mlir::Value inValue = ins[i];
            mlir::OpResult outValue = outs[i];

            size_t index;
            auto it = ssaToWireMap.find(inValue);
            if (it == ssaToWireMap.end()) {
                llvm::errs() << "Error: Operation references an untracked value.\n";
                assert(false);
            }
            index = it->second;
            indices.push_back(index);
            ssaToWireMap.erase(it);
            ssaToWireMap[outValue] = index;
        }
        return indices;
    }

    // Gate Recognitions:
    Gate gateFromName(llvm::StringRef gateName)
    {
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if ((GATE_NAME[i] == gateName)) {
                return static_cast<Gate>(i);
            }
        }
        return Gate::U;
    }

    Gate extractCliffTGate(CustomOp &op)
    {
        Gate gate = gateFromName(op.getGateName());
        if (!op.getInCtrlQubits().empty() || !op.getInCtrlValues().empty()) {
            if (isPhaseGate(gate)) {
                return Gate::I; // C-Rz gates don't alter state space, but alter phase space
                                // non-linearly, which I'm not going to track for now, but should be
                                // trackable using xy = x + y - (x \oplus y).
            }
            // if (gate == Gate::X && op.getInCtrlQubits().size() == 1 &&
            // op.getInCtrlValues().empty()) {
            //     return Gate::CNOT;
            // }    should pass getInCtrlQubit as qubitIn. will do it later.
            return Gate::U;
        }
        return gate;
    }

    // Rotation Angle Computations:
    double getPhase(CustomOp &op)
    {
        double c = (op.getAdjointFlag() ? -1 : 1);
        double angle = rotAngle(gateFromName(op.getGateName()));

        return ((angle != UNKNOWN_ANGLE ? angle : extractRZAngle(op.getParams())) * c);
    }

    double extractRZAngle(mlir::ValueRange params)
    {
        if (params.empty()) {
            return 0.0;
        }

        mlir::FloatAttr floatAttr;
        if (mlir::matchPattern(params.front(), mlir::m_Constant(&floatAttr))) {
            return floatAttr.getValueAsDouble();
        }
        else { // dynamic param
            return 0.0;
        }
    }

    double sumAngles(const PhaseBucket &contributors)
    {
        double sum = 0.0;
        for (GateID id : contributors.zeroAffineRZs) {
            sum += getPhase(phaseOps[id]);
        }
        for (GateID id : contributors.oneAffineRZs) {
            sum -= getPhase(phaseOps[id]);
        }
        return sum;
    }

    double normalizeAngle(double angle)
    { // returns equivalent angle between [-PI, PI]
        return std::remainder(angle, 2 * PI);
    }

    // IR Modifications:
    void updateTargetOp(CustomOp &targetOp, double sumAngle)
    {
        double normAngle = normalizeAngle(sumAngle);
        bool isAdjoint = (normAngle < 0.0);
        normAngle = std::abs(normAngle);
        Gate gate = gateWithAngle(normAngle);

        if (gate == Gate::I) {
            killOp(targetOp);
        }
        else {
            replaceOpWith(targetOp, gate, isAdjoint, normAngle);
        }
    }

    void removePhaseOp(CustomOp &op)
    {
        if (gateFromName(op.getGateName()) == Gate::Y) {
            replaceOpWith(op, Gate::X, false, 0.0);
        }
        else {
            killOp(op);
        }
    }

    void replaceOpWith(CustomOp &preOp, Gate newGate, bool isAdjoint, double angle)
    {
        updateStats(gateFromName(preOp.getGateName()), -1);

        mlir::IRRewriter rewriter(preOp.getContext());
        rewriter.setInsertionPoint(preOp.getOperation());
        mlir::Value angleVal;
        if (newGate == Gate::RZ) {
            angleVal = arith::ConstantOp::create(rewriter, preOp.getLoc(),
                                                 rewriter.getF64FloatAttr(angle));
        }

        CustomOp newOp = CustomOp::create(
            rewriter, preOp.getLoc(),
            /*gate_name=*/GATE_NAME[static_cast<size_t>(newGate)],
            /*in_qubits=*/preOp.getInQubits(),
            /*params=*/(newGate == Gate::RZ) ? mlir::ValueRange({angleVal}) : mlir::ValueRange({}),
            /*adjoint=*/isAdjoint);

        rewriter.replaceOp(preOp, newOp);
        updateStats(newGate, +1);
    }

    void killOp(CustomOp &op)
    {
        assert(op.getInCtrlQubits().empty() &&
               op.getInCtrlValues().empty()); // move to somewhere better
        updateStats(gateFromName(op.getGateName()), -1);

        mlir::ValueRange qIns = op.getInQubits();
        op.replaceAllUsesWith(qIns);
        op.erase();
    }

    // Phase-folding Algorithm:
    void phaseAnalysis(CustomOp customOp, SymbolicCircuit &symCirc, GateID &gateID)
    {
        llvm::SmallVector<size_t, 4> qubitIndices =
            getQubitIndices(customOp.getInQubits(), customOp.getOutQubits());

        Gate gate = extractCliffTGate(customOp);
        if (isPhaseGate(gate)) {
            phaseOps.push_back(customOp);
            gateID++;
        }
        initialGateCount[static_cast<size_t>(gate)]++;

        symCirc.applyGate(gate, customOp.getAdjointFlag(), qubitIndices, gateID);
    }

    void phaseMerge(SymbolicCircuit &symCirc)
    {
        auto removeGates = [&](PhaseBucket &contributors,
                               std::optional<GateID> skipID = std::nullopt) {
            for (GateID id : contributors.getAllGatesMutable()) {
                if (id != skipID) {
                    removePhaseOp(phaseOps[id]);
                }
            }
        };
        for (auto &[parity, contributors] : symCirc.phasePoly.terms) {
            if (parity.isTrivial()) {
                removeGates(contributors);
            }
            else if (!parity.isUnsat() && contributors.gateCount() > 1) {
                double angleSum = sumAngles(contributors);
                if (!contributors.isMergeTargetAffineZero()) {
                    angleSum = -angleSum;
                }

                GateID targetOpID = contributors.getMergeTarget();
                updateTargetOp(phaseOps[targetOpID], angleSum);
                removeGates(contributors, targetOpID);
            }
        }
    }

    void runOnOperation() override
    {
        llvm::outs() << "Hello phase-folding world!\n";

        SymbolicCircuit symCirc = SymbolicCircuit();
        GateID gateID = -1;

        getOperation()->walk([&](Operation *op) {
            if (auto customOp = dyn_cast<CustomOp>(op)) {
                phaseAnalysis(customOp, symCirc, gateID);
            }
            else if (auto extractOp = dyn_cast<ExtractOp>(op)) {
                extractFromQreg(extractOp);
            }
            else if (auto allocQbOp = dyn_cast<AllocQubitOp>(op)) {
                allocateQubit(allocQbOp.getResult(), symCirc);
            }
            else if (auto allocOp = dyn_cast<AllocOp>(op)) {
                allocateRegister(allocOp.getResult(), allocOp.getNqubitsAttr(), symCirc);
            }
            else if (auto basisOp = dyn_cast<SetBasisStateOp>(op)) {
                initQubitsState(basisOp, symCirc);
            }
            else if (auto gpOp = dyn_cast<GlobalPhaseOp>(op)) {
                llvm::outs() << "GlobalPhaseOp\n";
            }
            else {
                // llvm::outs() << "QuantumOperation\n";
                applyUndefinedOp(op, symCirc);
            }
        });

        llvm::outs() << symCirc << "\n";

        phaseMerge(symCirc);
        reportStats();
    }
};

} // namespace quantum
} // namespace catalyst

// Currently ignoring any blocks or dynamic allocations, only capturing pure quantum circuits.
// quantum.insert?