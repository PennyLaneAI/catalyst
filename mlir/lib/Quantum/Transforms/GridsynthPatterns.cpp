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
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

// --- Helper Functions to build switch cases ---

/**
 * @brief Helper to create a chain of parameter-less single qubit quantum custom gates.
 */
Value createGateChain(PatternRewriter &rewriter, Location loc, Value qbitIn,
                      ArrayRef<StringRef> gateNames, bool isAdjoint = false)
{
    auto qbitType = QubitType::get(rewriter.getContext());
    Value currentQbit = qbitIn;

    for (StringRef gateName : gateNames) {
        NamedAttrList newAttrs;
        newAttrs.append(rewriter.getNamedAttr("gate_name", rewriter.getStringAttr(gateName)));
        newAttrs.append(rewriter.getNamedAttr("operandSegmentSizes",
                                              rewriter.getDenseI32ArrayAttr({0, 1, 0, 0})));
        if (isAdjoint) {
            newAttrs.append(rewriter.getNamedAttr("adjoint", rewriter.getUnitAttr()));
        }
        newAttrs.append(
            rewriter.getNamedAttr("resultSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0})));

        auto newOp =
            CustomOp::create(rewriter, loc, qbitType, ValueRange{currentQbit}, newAttrs.getAttrs());
        currentQbit = newOp.getResult(0);
    }

    return currentQbit;
}

/**
 * @brief Populates the scf.index_switch op for the Clifford+T basis.
 */
void populateCliffordTSwitchCases(PatternRewriter &rewriter, Location loc,
                                  scf::IndexSwitchOp switchOp, Value qbitIn)
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
    for (size_t i = 0; i < caseConfigs.size(); i++) {
        const auto &config = caseConfigs[i];
        Region &caseRegion = switchOp.getCaseRegions()[i];
        caseRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion.front());

        // Pass the qubit through the chain
        Value qbitOut = createGateChain(rewriter, loc, qbitIn, config.first, config.second);
        scf::YieldOp::create(rewriter, loc, qbitOut);
    }

    // Populate Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    static StringRef gatesDefault[] = {"Identity"};
    Value qbitDefault = createGateChain(rewriter, loc, qbitIn, gatesDefault, /*isAdjoint=*/false);
    scf::YieldOp::create(rewriter, loc, qbitDefault);
}

/**
 * @brief Populates the scf.index_switch op for the PPR-Basis.
 * Maps the new enum (I, X2...adjZ8) to PPRotationOps.
 */
void populatePPRBasisSwitchCases(PatternRewriter &rewriter, Location loc,
                                 scf::IndexSwitchOp switchOp, Value qbitIn)
{
    // Helper to create a single PPRotationOp directly on the qubit.
    auto createPPROp = [&](OpBuilder &builder, ArrayRef<StringRef> pauliWord, uint16_t rotationKind,
                           bool isAdjoint, Value currentQbit) -> Value {
        // Convert rotation kind to signed integer and negate if adjoint
        int16_t signedRotation = static_cast<int16_t>(rotationKind);
        if (isAdjoint) {
            signedRotation = -signedRotation;
        }

        // We need to cast back to uint16_t for the C++ builder signature
        uint16_t finalRotationArg = static_cast<uint16_t>(signedRotation);

        auto pprOp = catalyst::pbc::PPRotationOp::create(builder, loc, pauliWord, finalRotationArg,
                                                         ValueRange{currentQbit}, nullptr);

        return pprOp->getResult(0);
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

    SmallVector<PPRConfig> caseConfigs;

    // Case 0: I
    caseConfigs.push_back({true, {}, 0, false});

    // Helper to push X, Y, Z series
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

    for (size_t i = 0; i < caseConfigs.size(); i++) {
        const auto &config = caseConfigs[i];
        Region &caseRegion = switchOp.getCaseRegions()[i];
        caseRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion.front());

        Value qbitOut = config.isIdentity ? qbitIn
                                          : createPPROp(rewriter, config.pauli, config.n,
                                                        config.isAdjoint, qbitIn);

        scf::YieldOp::create(rewriter, loc, qbitOut);
    }

    // Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    scf::YieldOp::create(rewriter, loc, qbitIn);
}

struct DecompositionExternalFuncs {
    func::FuncOp getSize;
    func::FuncOp getGates;
    func::FuncOp getPhase;
};

DecompositionExternalFuncs getOrDeclareExternalFuncs(PatternRewriter &rewriter, func::FuncOp func)
{
    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type();
    auto indexType = rewriter.getIndexType();
    auto rankedMemRefType = MemRefType::get({ShapedType::kDynamic}, indexType);

    // Ensure or declare Get Size: (f64, f64, i1) -> index
    auto getSizeType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {indexType});
    auto getSizeFunc = catalyst::ensureFunctionDeclaration<func::FuncOp>(
        rewriter, func, "rs_decomposition_get_size", getSizeType);

    // Ensure or declare Get Gates: (memref, f64, f64, i1) -> void
    auto getGatesType = rewriter.getFunctionType({rankedMemRefType, f64Type, f64Type, i1Type}, {});
    auto getGatesFunc = catalyst::ensureFunctionDeclaration<func::FuncOp>(
        rewriter, func, "rs_decomposition_get_gates", getGatesType);

    // Ensure or declare Get Phase: (f64, f64, i1) -> f64
    auto getPhaseType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {f64Type});
    auto getPhaseFunc = catalyst::ensureFunctionDeclaration<func::FuncOp>(
        rewriter, func, "rs_decomposition_get_phase", getPhaseType);

    return {getSizeFunc, getGatesFunc, getPhaseFunc};
}

/**
 * @brief Builds the main loop that iterates over the gate sequence and applies the quantum gates.
 */
Value buildDecompositionLoop(PatternRewriter &rewriter, Location loc, Value qbitIn,
                             Value gatesMemref, Value numGates, double epsilon, bool pprBasis)
{
    auto qbitType = QubitType::get(rewriter.getContext());
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Create the scf.for loop over gate sequence indices
    // The loop carries the Qubit as an argument
    auto forOp = scf::ForOp::create(rewriter, loc, c0, numGates, c1, ValueRange{qbitIn});

    // Add attribute to the for op to indicate the estimated iterations of the loop
    auto estimatedRanges = static_cast<int64_t>(std::ceil(10 * std::log2(1 / epsilon)));
    auto estimatedRangesAttr = rewriter.getI16IntegerAttr(estimatedRanges);
    forOp->setAttr("estimated_iterations", estimatedRangesAttr);

    {
        OpBuilder::InsertionGuard loopGuard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value currentQbit = forOp.getRegionIterArg(0);

        Value currentGateIndex = memref::LoadOp::create(rewriter, loc, gatesMemref, ValueRange{iv});

        // 19 cases for PPR basis: Identity + (X, Y, Z) x (2, 4, 8) x (normal, adjoint)
        // 10 cases for Clifford+T basis: {T, H T, S H T, I, X, Y, Z, H, S, adjS}
        const int64_t numCases = pprBasis ? 19 : 10;
        SmallVector<int64_t> caseValues;
        caseValues.reserve(numCases);
        for (int64_t i = 0; i < numCases; i++) {
            caseValues.push_back(i);
        }

        // Create the switch operation inside the loop
        DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
        auto switchOp =
            scf::IndexSwitchOp::create(rewriter, loc, TypeRange{qbitType}, currentGateIndex,
                                       caseValuesAttr, caseValues.size());

        // Populate Switch Cases
        if (pprBasis) {
            populatePPRBasisSwitchCases(rewriter, loc, switchOp, currentQbit);
        }
        else {
            populateCliffordTSwitchCases(rewriter, loc, switchOp, currentQbit);
        }

        // Yield the result of the switch op from the for loop
        rewriter.setInsertionPointAfter(switchOp);
        scf::YieldOp::create(rewriter, loc, switchOp->getResults());
    }

    // Return the result of the loop (the final qubit state)
    return forOp.getResult(0);
}

/**
 * @brief Gets or creates the RZ/PhaseShift decomposition function.
 * This function contains the loop and switch logic acting on a Qubit.
 */
func::FuncOp getOrCreateDecompositionFunc(ModuleOp module, PatternRewriter &rewriter,
                                          double epsilon, bool pprBasis)
{
    StringRef funcName = pprBasis ? "__catalyst_decompose_RZ_ppr_basis" : "__catalyst_decompose_RZ";

    // Check if it exists
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    // Define function type: (qubit, f64) -> (qubit, f64)
    // inputs:
    // qubit: input qubit wire
    // f64: rotation angle
    // returns:
    // qubit: output qubit wire
    // f64: runtime phase
    auto qbitType = QubitType::get(rewriter.getContext());
    auto f64Type = rewriter.getF64Type();
    auto funcType = rewriter.getFunctionType({qbitType, f64Type}, {qbitType, f64Type});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = func::FuncOp::create(rewriter, module.getLoc(), funcName, funcType);
    func.setPrivate();

    // Get or declare external functions (GetSize, GetGates, GetPhase)
    DecompositionExternalFuncs extFuncs = getOrDeclareExternalFuncs(rewriter, func);

    // Build function body
    Block *entryBlock = func.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    Location loc = func.getLoc();
    Value qbitIn = entryBlock->getArgument(0);
    Value angle = entryBlock->getArgument(1);

    // Parameters for compilation
    Value epsilonVal = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(epsilon));
    Value pprBasisVal = arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(pprBasis));

    // Call GetSize
    auto callGetSizeOp = func::CallOp::create(rewriter, loc, extFuncs.getSize,
                                              ValueRange{angle, epsilonVal, pprBasisVal});
    Value num_gates = callGetSizeOp->getResult(0);

    // Call GetGates
    // Use memref.alloc (Heap) instead of alloca (Stack) because num_gates is dynamic.
    auto gatesMemRefType = MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
    Value gatesMemref = memref::AllocOp::create(rewriter, loc, gatesMemRefType, num_gates);

    func::CallOp::create(rewriter, loc, extFuncs.getGates,
                         ValueRange{gatesMemref, angle, epsilonVal, pprBasisVal});

    // Call GetPhase
    auto callGetPhaseOp = func::CallOp::create(rewriter, loc, extFuncs.getPhase,
                                               ValueRange{angle, epsilonVal, pprBasisVal});
    Value runtimePhase = callGetPhaseOp->getResult(0);

    // Build the Loop logic
    Value finalQbit =
        buildDecompositionLoop(rewriter, loc, qbitIn, gatesMemref, num_gates, epsilon, pprBasis);

    // Clean up heap memory
    memref::DeallocOp::create(rewriter, loc, gatesMemref);

    // Return the final qubit and the computed runtime phase
    func::ReturnOp::create(rewriter, loc, ValueRange{finalQbit, runtimePhase});

    return func;
}

// --- Rewrite Pattern for CustomOp (RZ/PhaseShift) ---

struct DecomposeCustomOpPattern : public OpRewritePattern<CustomOp> {
    using OpRewritePattern<CustomOp>::OpRewritePattern;

    const double epsilon;
    const bool pprBasis;

    DecomposeCustomOpPattern(MLIRContext *context, double epsilon, bool pprBasis)
        : OpRewritePattern<CustomOp>(context), epsilon(epsilon), pprBasis(pprBasis)
    {
    }

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        StringRef gateName = op.getGateName();
        bool isRZ = gateName == "RZ";
        bool isPhaseShift = gateName == "PhaseShift";
        if (!isRZ && !isPhaseShift) {
            return failure();
        }

        assert(op.getQubitOperands().size() == 1 && op.getAllParams().size() == 1 &&
               "only RZ and PhaseShift are allowed in Gridsynth decomposition");

        // Directly grab the SSA value of the qubit. No need to look up ExtractOps.
        Value qbitOperand = op.getQubitOperands()[0];
        Value angle = op.getAllParams()[0];

        ModuleOp mod = op->getParentOfType<ModuleOp>();
        Location loc = op.getLoc();

        func::FuncOp decompFunc = getOrCreateDecompositionFunc(mod, rewriter, epsilon, pprBasis);

        // Call the function using the qubit directly
        auto callDecompOp =
            func::CallOp::create(rewriter, loc, decompFunc, ValueRange{qbitOperand, angle});

        Value finalQbitResult = callDecompOp->getResult(0);
        Value runtimePhase = callDecompOp.getResult(1);

        // Handle Global Phase for PhaseShift
        Value finalPhase;
        if (isPhaseShift) {
            // PhaseShift(phi) = RZ(phi) * GlobalPhase(-phi/2)
            Value c2 = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(2.0));
            Value halfAngle = arith::DivFOp::create(rewriter, loc, angle, c2);
            finalPhase = arith::SubFOp::create(rewriter, loc, runtimePhase, halfAngle);
        }
        else {
            finalPhase = runtimePhase;
        }

        NamedAttrList gphaseAttrs;
        gphaseAttrs.append(
            rewriter.getNamedAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 0})));
        GlobalPhaseOp::create(rewriter, loc, TypeRange{}, ValueRange{finalPhase},
                              gphaseAttrs.getAttrs());

        // Replace the RZ/PhaseShift op with the resulting qubit
        rewriter.replaceOp(op, finalQbitResult);
        return success();
    }
};

// --- Rewrite Pattern for PPRotationArbitraryOp ---

struct DecomposePPRArbitraryOpPattern
    : public OpRewritePattern<catalyst::pbc::PPRotationArbitraryOp> {
    using OpRewritePattern<catalyst::pbc::PPRotationArbitraryOp>::OpRewritePattern;

    const double epsilon;
    const bool pprBasis;

    DecomposePPRArbitraryOpPattern(MLIRContext *context, double epsilon, bool pprBasis)
        : OpRewritePattern<catalyst::pbc::PPRotationArbitraryOp>(context), epsilon(epsilon),
          pprBasis(pprBasis)
    {
    }

    LogicalResult matchAndRewrite(catalyst::pbc::PPRotationArbitraryOp op,
                                  PatternRewriter &rewriter) const override
    {
        if (op.getPauliProduct() != rewriter.getStrArrayAttr({"Z"})) {
            return failure();
        }

        if (op.getCondition()) {
            return failure();
        }

        Value qbitOperand = op.getInQubits()[0];
        Value angle = op.getArbitraryAngle();

        ModuleOp mod = op->getParentOfType<ModuleOp>();
        Location loc = op.getLoc();

        // PPR(theta, Z) = exp(-i * theta * Z)
        // RZ(phi)       = exp(-i * phi/2 * Z)
        // phi = 2 * theta
        Value cMinus2 = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(2.0));
        Value rzAngle = arith::MulFOp::create(rewriter, loc, angle, cMinus2);

        func::FuncOp decompFunc = getOrCreateDecompositionFunc(mod, rewriter, epsilon, pprBasis);

        auto callDecompOp =
            func::CallOp::create(rewriter, loc, decompFunc, ValueRange{qbitOperand, rzAngle});

        Value finalQbitResult = callDecompOp->getResult(0);
        Value runtimePhase = callDecompOp.getResult(1);

        NamedAttrList gphaseAttrs;
        gphaseAttrs.append(
            rewriter.getNamedAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 0})));
        GlobalPhaseOp::create(rewriter, loc, TypeRange{}, ValueRange{runtimePhase},
                              gphaseAttrs.getAttrs());

        rewriter.replaceOp(op, finalQbitResult);
        return success();
    }
};

} // anonymous namespace

namespace catalyst {
namespace quantum {

void populateGridsynthPatterns(RewritePatternSet &patterns, double epsilon, bool pprBasis)
{
    patterns.add<DecomposeCustomOpPattern>(patterns.getContext(), epsilon, pprBasis);
    patterns.add<DecomposePPRArbitraryOpPattern>(patterns.getContext(), epsilon, pprBasis);
}

} // namespace quantum
} // namespace catalyst
