// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "gridsynth-patterns"

#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::SmallVector;
using llvm::StringRef;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

// --- Helper Functions to build switch cases ---

/**
 * @brief Helper to create a chain of parameter-less single qubit quantum custom gates.
 */
mlir::Value createGateChain(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value qregIn,
                            mlir::Value qbitIndex, mlir::ArrayRef<StringRef> gateNames,
                            bool isAdjoint = false)
{
    auto qregType = qregIn.getType();
    auto qbitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    mlir::Value currentQbit = rewriter.create<ExtractOp>(loc, qbitType, qregIn, qbitIndex, nullptr);

    for (StringRef gateName : gateNames) {
        mlir::NamedAttrList newAttrs;
        newAttrs.append(rewriter.getNamedAttr("gate_name", rewriter.getStringAttr(gateName)));
        newAttrs.append(rewriter.getNamedAttr("operandSegmentSizes",
                                              rewriter.getDenseI32ArrayAttr({0, 1, 0, 0})));
        if (isAdjoint) {
            newAttrs.append(rewriter.getNamedAttr("adjoint", rewriter.getUnitAttr()));
        }
        newAttrs.append(
            rewriter.getNamedAttr("resultSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0})));

        auto newOp = rewriter.create<CustomOp>(loc, qbitType, mlir::ValueRange{currentQbit},
                                               newAttrs.getAttrs());
        currentQbit = newOp.getResult(0);
    }

    mlir::Value qregOut =
        rewriter.create<InsertOp>(loc, qregType, qregIn, qbitIndex, nullptr, currentQbit);
    return qregOut;
}

/**
 * @brief Populates the scf.index_switch op for the Clifford+T basis.
 */
void populateCliffordTSwitchCases(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                  mlir::scf::IndexSwitchOp switchOp, mlir::Value qregIn,
                                  mlir::Value qbitIndex)
{
    static StringRef gates0[] = {"T"};
    static StringRef gates1[] = {"Hadamard", "T"};
    static StringRef gates2[] = {"S", "Hadamard", "T"};
    static StringRef gates3[] = {"Identity"};
    static StringRef gates4[] = {"PauliX"};
    static StringRef gates5[] = {"PauliY"};
    static StringRef gates6[] = {"PauliZ"};
    static StringRef gates7[] = {"Hadamard"};
    static StringRef gates8[] = {"S"}; // Used for both Case 8 and 9

    SmallVector<std::pair<llvm::ArrayRef<StringRef>, bool>, 10> caseConfigs = {
        {gates0, /*isAdjoint=*/false}, // Case 0
        {gates1, /*isAdjoint=*/false}, // Case 1
        {gates2, /*isAdjoint=*/false}, // Case 2
        {gates3, /*isAdjoint=*/false}, // Case 3
        {gates4, /*isAdjoint=*/false}, // Case 4
        {gates5, /*isAdjoint=*/false}, // Case 5
        {gates6, /*isAdjoint=*/false}, // Case 6
        {gates7, /*isAdjoint=*/false}, // Case 7
        {gates8, /*isAdjoint=*/false}, // Case 8
        {gates8, /*isAdjoint=*/true},  // Case 9
    };

    // Populate Switch Cases
    assert(caseConfigs.size() == switchOp.getCases().size() &&
           "Mismatch in case config and case values");
    for (size_t i = 0; i < caseConfigs.size(); ++i) {
        const auto &config = caseConfigs[i];
        Region &caseRegion = switchOp.getCaseRegions()[i];
        caseRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion.front());

        mlir::Value qregCase =
            createGateChain(rewriter, loc, qregIn, qbitIndex, config.first, config.second);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase);
    }

    // Populate Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    static StringRef gatesDefault[] = {"Identity"};
    mlir::Value qregDefault =
        createGateChain(rewriter, loc, qregIn, qbitIndex, gatesDefault, /*isAdjoint=*/false);
    rewriter.create<mlir::scf::YieldOp>(loc, qregDefault);
}

/**
 * @brief Populates the scf.index_switch op for the PPR-Basis.
 * Maps the new enum (I, X2...adjZ8) to PPRotationOps.
 */
void populatePPRBasisSwitchCases(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                 mlir::scf::IndexSwitchOp switchOp, mlir::Value qregIn,
                                 mlir::Value qbitIndex)
{
    auto qregType = qregIn.getType();
    auto qbitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Helper to create a single PPRotationOp.
    // Handles 'adjoint' by negating the rotationKind.
    auto createPPROp = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> pauliWord,
                           uint16_t rotationKind, bool isAdjoint,
                           mlir::Value currentLoopReg) -> mlir::Value {
        mlir::Value currentQbit =
            builder.create<ExtractOp>(loc, qbitType, currentLoopReg, qbitIndex, nullptr);

        // Convert rotation kind to signed integer and negate if adjoint
        int16_t signedRotation = static_cast<int16_t>(rotationKind);
        if (isAdjoint) {
            signedRotation = -signedRotation;
        }

        // We need to cast back to uint16_t for the C++ builder signature
        uint16_t finalRotationArg = static_cast<uint16_t>(signedRotation);

        auto pprOp = builder.create<catalyst::qec::PPRotationOp>(
            loc, pauliWord, finalRotationArg, mlir::ValueRange{currentQbit}, nullptr);

        mlir::Value resultQbit = pprOp.getResult(0);

        mlir::Value qregOut =
            builder.create<InsertOp>(loc, qregType, currentLoopReg, qbitIndex, nullptr, resultQbit);
        return qregOut;
    };

    struct PPRConfig {
        bool isIdentity;
        ArrayRef<StringRef> pauli;
        uint16_t n;     // The denominator (2, 4, 8)
        bool isAdjoint; // True if adjX, adjY, etc.
    };

    static StringRef xPauli[] = {"X"};
    static StringRef yPauli[] = {"Y"};
    static StringRef zPauli[] = {"Z"};

    // Map indices 0..18 to configs
    SmallVector<PPRConfig> caseConfigs;

    // Case 0: I
    caseConfigs.push_back({true, {}, 0, false});

    // Helper to push X, Y, Z: (2, 4, 8, adj2, adj4, adj8)
    auto pushSeries = [&](ArrayRef<StringRef> pauli) {
        caseConfigs.push_back({false, pauli, 2, false});
        caseConfigs.push_back({false, pauli, 4, false});
        caseConfigs.push_back({false, pauli, 8, false});
        caseConfigs.push_back({false, pauli, 2, true});
        caseConfigs.push_back({false, pauli, 4, true});
        caseConfigs.push_back({false, pauli, 8, true});
    };

    pushSeries(xPauli);
    pushSeries(yPauli);
    pushSeries(zPauli);

    for (size_t i = 0; i < caseConfigs.size(); ++i) {
        const auto &config = caseConfigs[i];
        Region &caseRegion = switchOp.getCaseRegions()[i];
        caseRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion.front());

        mlir::Value qregCase = config.isIdentity ? qregIn
                                                 : createPPROp(rewriter, config.pauli, config.n,
                                                               config.isAdjoint, qregIn);

        rewriter.create<mlir::scf::YieldOp>(loc, qregCase);
    }

    // Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    rewriter.create<mlir::scf::YieldOp>(loc, qregIn);
}

/**
 * @brief Gets or creates the RZ/PhaseShift decomposition function.
 * This function contains the loop and switch logic.
 */
mlir::func::FuncOp getOrCreateDecompositionFunc(mlir::ModuleOp module,
                                                mlir::PatternRewriter &rewriter, double epsilon,
                                                bool pprBasis)
{
    const char *funcName =
        pprBasis ? "__catalyst_decompose_RZ_ppr_basis" : "__catalyst_decompose_RZ";

    // Check if it exists
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    // Define function type: (qureg, i64, f64) -> (qureg, f64)
    // inputs:
    // qureg: input quantum register
    // i64: qubit index
    // f64: rotation angle
    // returns:
    // qureg: output quantum register
    // f64: runtime phase
    auto qregType = catalyst::quantum::QuregType::get(rewriter.getContext());
    auto indexType = rewriter.getI64Type();
    auto f64Type = rewriter.getF64Type();
    auto funcType = rewriter.getFunctionType({qregType, indexType, f64Type}, {qregType, f64Type});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    // Build function body
    auto *entryBlock = func.addEntryBlock();
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    mlir::Location loc = func.getLoc();
    mlir::Value qregIn = entryBlock->getArgument(0);
    mlir::Value qbitIndex = entryBlock->getArgument(1);
    mlir::Value angle = entryBlock->getArgument(2);
    auto i1Type = rewriter.getI1Type();

    // Ensure or declare Get Size
    auto getSizeType =
        rewriter.getFunctionType({f64Type, f64Type, i1Type}, {rewriter.getIndexType()});
    auto getSizeFunc =
        catalyst::ensurefuncOrDeclare(rewriter, func, "rs_decomposition_get_size", getSizeType);

    // Ensure or declare Get Gates
    auto rankedMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    auto getGatesType = rewriter.getFunctionType({rankedMemRefType, f64Type, f64Type, i1Type}, {});
    auto getGatesFunc =
        catalyst::ensurefuncOrDeclare(rewriter, func, "rs_decomposition_get_gates", getGatesType);

    // Ensure or declare Get Phase
    auto getPhaseType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {f64Type});
    auto getPhaseFunc =
        catalyst::ensurefuncOrDeclare(rewriter, func, "rs_decomposition_get_phase", getPhaseType);

    // Parameters for compilation
    mlir::Value epsilonVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(epsilon));
    mlir::Value pprBasisVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(pprBasis));

    // Call GetSize
    auto callGetSizeOp = rewriter.create<mlir::func::CallOp>(
        loc, getSizeFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});
    mlir::Value num_gates = callGetSizeOp.getResult(0);

    // Call GetGates
    auto gatesMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    mlir::Value gatesMemref =
        rewriter.create<mlir::memref::AllocaOp>(loc, gatesMemRefType, num_gates);
    rewriter.create<mlir::func::CallOp>(
        loc, getGatesFunc, mlir::ValueRange{gatesMemref, angle, epsilonVal, pprBasisVal});

    // Call GetPhase
    auto callGetPhaseOp = rewriter.create<mlir::func::CallOp>(
        loc, getPhaseFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});
    mlir::Value runtimePhase = callGetPhaseOp.getResult(0);

    // Create the scf.for loop over gate sequence indices
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto forOp = rewriter.create<mlir::scf::ForOp>(loc, c0, num_gates, c1, ValueRange{qregIn});

    mlir::OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    mlir::Value iv = forOp.getInductionVar();
    mlir::Value currentLoopReg = forOp.getRegionIterArg(0);

    mlir::Value currentGateIndex =
        rewriter.create<mlir::memref::LoadOp>(loc, gatesMemref, ValueRange{iv});

    const int64_t numCases = pprBasis ? 19 : 10;
    SmallVector<int64_t> caseValues;
    caseValues.reserve(numCases);
    for (int64_t i = 0; i < numCases; ++i) {
        caseValues.push_back(i);
    }
    // Create the switch operation inside the loop
    mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
    auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
        loc, mlir::TypeRange{qregType}, currentGateIndex, caseValuesAttr, caseValues.size());

    // Populate Switch Cases
    if (pprBasis) {
        populatePPRBasisSwitchCases(rewriter, loc, switchOp, currentLoopReg, qbitIndex);
    }
    else {
        populateCliffordTSwitchCases(rewriter, loc, switchOp, currentLoopReg, qbitIndex);
    }

    // Yield the result of the switch op from the for loop
    rewriter.setInsertionPointAfter(switchOp);
    rewriter.create<mlir::scf::YieldOp>(loc, switchOp.getResults());

    rewriter.setInsertionPointAfter(forOp);
    mlir::Value finalReg = forOp.getResult(0);

    // Return the final register and the computed runtime phase
    rewriter.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{finalReg, runtimePhase});

    return func;
}

// --- Rewrite Pattern ---

struct DecomposeCustomOpPattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    const double epsilon;
    const bool pprBasis;

    DecomposeCustomOpPattern(mlir::MLIRContext *context, double epsilon, bool pprBasis)
        : mlir::OpRewritePattern<CustomOp>(context), epsilon(epsilon), pprBasis(pprBasis)
    {
    }

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        StringRef gateName = op.getGateName();
        bool isRZ = gateName == "RZ";
        bool isPhaseShift = gateName == "PhaseShift";
        if (!isRZ && !isPhaseShift) {
            return failure();
        }

        mlir::Value qbitOperand;
        SmallVector<mlir::Value> paramOperands;

        for (mlir::Value operand : op->getOperands()) {
            if (mlir::isa<catalyst::quantum::QubitType>(operand.getType())) {
                qbitOperand = operand;
            }
            else {
                paramOperands.push_back(operand);
            }
        }
        if (paramOperands.size() != 1) {
            return failure();
        }
        mlir::Value angle = paramOperands[0];
        if (!qbitOperand) {
            return failure();
        }

        catalyst::quantum::ExtractOp extractOp = findSourceExtract(qbitOperand);
        if (!extractOp) {
            return failure();
        }

        mlir::Value qregOperand = extractOp.getQreg();
        mlir::Value qbitIndex;

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(extractOp);

        if (extractOp.getIdx()) {
            qbitIndex = extractOp.getIdx();
        }
        else if (extractOp.getIdxAttr()) {
            qbitIndex = rewriter.create<mlir::arith::ConstantOp>(
                extractOp.getLoc(), rewriter.getI64IntegerAttr(*extractOp.getIdxAttr()));
        }
        else {
            return failure();
        }

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        mlir::Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        mlir::Value regWithQubit = rewriter.create<InsertOp>(
            loc, qregOperand.getType(), qregOperand, qbitIndex, nullptr, qbitOperand);

        mlir::func::FuncOp decompFunc =
            getOrCreateDecompositionFunc(module, rewriter, epsilon, pprBasis);

        auto callDecompOp = rewriter.create<mlir::func::CallOp>(
            loc, decompFunc, mlir::ValueRange{regWithQubit, qbitIndex, angle});
        mlir::Value finalReg = callDecompOp.getResult(0);
        mlir::Value runtimePhase = callDecompOp.getResult(1);

        mlir::Value finalQbitResult =
            rewriter.create<ExtractOp>(loc, qbitOperand.getType(), finalReg, qbitIndex, nullptr);

        rewriter.setInsertionPointAfter(finalQbitResult.getDefiningOp());

        mlir::Value finalPhase;
        if (isPhaseShift) {
            finalPhase = rewriter.create<mlir::arith::AddFOp>(loc, runtimePhase, angle);
        }
        else {
            finalPhase = runtimePhase;
        }

        mlir::NamedAttrList gphaseAttrs;
        gphaseAttrs.append(
            rewriter.getNamedAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 0})));
        rewriter.create<GlobalPhaseOp>(loc, TypeRange{}, ValueRange{finalPhase},
                                       gphaseAttrs.getAttrs());

        rewriter.replaceAllUsesWith(op->getResults(), finalQbitResult);
        rewriter.eraseOp(op);
        return success();
    }

  private:
    /**
     * @brief Traces a qubit value backward to find its source ExtractOp.
     */
    catalyst::quantum::ExtractOp findSourceExtract(mlir::Value qbit) const
    {
        if (!qbit) {
            return nullptr;
        }
        if (auto extractOp = qbit.getDefiningOp<ExtractOp>())
            return extractOp;

        auto definingOp = qbit.getDefiningOp();
        if (!definingOp) {
            return nullptr;
        }

        if (auto customOp = dyn_cast<CustomOp>(definingOp)) {
            auto opResult = mlir::dyn_cast<mlir::OpResult>(qbit);
            if (!opResult) {
                return nullptr;
            }

            std::vector<mlir::Value> qubitOperands = customOp.getQubitOperands();
            std::vector<mlir::OpResult> qubitResults = customOp.getQubitResults();

            if (qubitOperands.size() != qubitResults.size()) {
                return nullptr;
            }

            int64_t qubitResultIndex = -1;
            for (size_t i = 0; i < qubitResults.size(); ++i) {
                if (qubitResults[i] == opResult) {
                    qubitResultIndex = i;
                    break;
                }
            }
            if (qubitResultIndex == -1) {
                return nullptr;
            }

            mlir::Value nextQubit = qubitOperands[qubitResultIndex];
            if (nextQubit)
                return findSourceExtract(nextQubit);
        }
        {
            return nullptr;
        }
    }
};

} // anonymous namespace

namespace catalyst {
namespace quantum {

void populateGridsynthPatterns(RewritePatternSet &patterns, double epsilon, bool pprBasis)
{
    patterns.add<DecomposeCustomOpPattern>(patterns.getContext(), epsilon, pprBasis);
}

} // namespace quantum
} // namespace catalyst
