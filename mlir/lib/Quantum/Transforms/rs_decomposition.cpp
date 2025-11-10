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

#define DEBUG_TYPE "rs-decomposition"

#include <memory>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumOps.h"

using llvm::SmallVector;
using llvm::StringRef;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_RSDECOMPOSITIONPASS
#define GEN_PASS_DEF_RSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

// --- Helper Functions to Declare External Runtime Functions ---

/**
 * @brief Gets or declares the external runtime function `rs_decomposition`.
 * This is used to get the decomposed gate sequence.
 * MLIR signature: func.func @rs_decomposition(f64, f64, i1) -> memref<?xindex>
 */
mlir::func::FuncOp getOrDeclareGetGatesFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    // Because of later annotation pass, the C++ runtime function
    // will be matched with `rs_decomposition_0` function
    const char *funcName = "rs_decomposition";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type(); // For ppr-basis flag
    auto rankedMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    // Runtime function takes two f64 inputs and one i1 input
    // corresponding to theta, epsilon, and ppr-basis
    auto funcType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {rankedMemRefType});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `rs_decomposition_get_phase`.
 * This is used to get the residual phase from the decomposition.
 * MLIR signature: func.func @rs_decomposition_get_phase(f64, f64, i1) -> f64
 * Will be used for global phase
 */
mlir::func::FuncOp getOrDeclareGetPhaseFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    // Because of later annotation pass, the C++ runtime function
    // will be matched with `rs_decomposition_get_phase_0` function
    const char *funcName = "rs_decomposition_get_phase";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type(); // For ppr-basis flag
    // Runtime function takes two f64 inputs and one i1 input
    // corresponding to theta, epsilon, and ppr-basis
    auto funcType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {f64Type});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `free_memref`.
 * MLIR signature: func.func @free_memref(memref<?xindex>) -> ()
 */
mlir::func::FuncOp getOrDeclareFreeMemrefFunc(mlir::ModuleOp module,
                                              mlir::PatternRewriter &rewriter)
{
    const char *funcName = "free_memref";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto rankedMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    auto funcType = rewriter.getFunctionType({rankedMemRefType}, {});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the RZ/PhaseShift decomposition function.
 * This function contains the loop and switch logic.
 * MLIR signature: func.func @__catalyst_decompose_RZ(qureg, i64, f64) -> (qureg, f64)
 * or @__catalyst_decompose_RZ_ppr_basis(qureg, i64, f64) -> (qureg, f64)
 * Returns the final register and the computed runtime phase.
 */
mlir::func::FuncOp getOrDeclareDecompositionFunc(mlir::ModuleOp module,
                                                 mlir::PatternRewriter &rewriter, double epsilon,
                                                 bool pprBasis)
{
    const char *funcName =
        pprBasis ? "__catalyst_decompose_RZ_ppr_basis" : "__catalyst_decompose_RZ";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    // Define function type: (qureg, i64, f64) -> (qureg, f64)
    auto qregType = catalyst::quantum::QuregType::get(rewriter.getContext());
    auto indexType = rewriter.getI64Type();
    auto f64Type = rewriter.getF64Type();
    auto funcType = rewriter.getFunctionType({qregType, indexType, f64Type}, {qregType, f64Type});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    auto *entryBlock = func.addEntryBlock();
    mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    mlir::Location loc = func.getLoc();
    mlir::Value qregIn = entryBlock->getArgument(0);
    mlir::Value qbitIndex = entryBlock->getArgument(1);
    mlir::Value angle = entryBlock->getArgument(2);

    auto qbitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Declare runtime functions
    mlir::func::FuncOp getGatesFunc = getOrDeclareGetGatesFunc(module, rewriter);
    mlir::func::FuncOp getPhaseFunc = getOrDeclareGetPhaseFunc(module, rewriter);
    mlir::func::FuncOp freeMemrefFunc = getOrDeclareFreeMemrefFunc(module, rewriter);

    mlir::Value epsilonVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(epsilon));
    mlir::Value pprBasisVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(pprBasis));

    // Call runtime functions
    auto callGetGatesOp = rewriter.create<mlir::func::CallOp>(
        loc, getGatesFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});
    auto callGetPhaseOp = rewriter.create<mlir::func::CallOp>(
        loc, getPhaseFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});

    mlir::Value rankedMemref = callGetGatesOp.getResult(0);
    mlir::Value runtimePhase = callGetPhaseOp.getResult(0);

    // Create the scf.for loop over gate sequence indices
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value memrefSize = rewriter.create<mlir::memref::DimOp>(loc, rankedMemref, c0);
    auto forOp = rewriter.create<mlir::scf::ForOp>(loc, c0, memrefSize, c1, ValueRange{qregIn});

    mlir::OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    mlir::Value iv = forOp.getInductionVar();
    mlir::Value currentLoopReg = forOp.getRegionIterArg(0);

    // Load the gate index inside the loop
    mlir::Value currentGateIndex =
        rewriter.create<mlir::memref::LoadOp>(loc, rankedMemref, ValueRange{iv});

    // --- START: Lambda Helpers for Op Creation ---

    // Helper lambda to create a chain of parameter-less single qubit CUSTOM gates.
    // Used for Clifford+T basis.
    auto createGateChain = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> gateNames,
                               mlir::Value qregIn, bool isAdjoint = false) -> mlir::Value {
        // Extract the qubit
        mlir::Value currentQbit =
            builder.create<ExtractOp>(loc, qbitType, qregIn, qbitIndex, /*idx_attr=*/nullptr);

        // Apply the chain of gates
        for (StringRef gateName : gateNames) {
            mlir::NamedAttrList newAttrs;
            newAttrs.append(builder.getNamedAttr("gate_name", builder.getStringAttr(gateName)));

            // Format is [params, qubits, ctrl_wires, ctrl_values]
            auto segmentSizes = builder.getDenseI32ArrayAttr({0, 1, 0, 0});
            newAttrs.append(builder.getNamedAttr("operandSegmentSizes", segmentSizes));

            // Add adjoint flag if isAdjoint is true
            if (isAdjoint) {
                newAttrs.append(builder.getNamedAttr("adjoint", builder.getUnitAttr()));
            }

            // Add resultSegmentSizes. Assume [qubits, classical_values]
            // We return 1 qubit and 0 classical values.
            auto resultSegmentSizes = builder.getDenseI32ArrayAttr({1, 0});
            newAttrs.append(builder.getNamedAttr("resultSegmentSizes", resultSegmentSizes));

            // Create the CustomOp.
            auto newOp = builder.create<CustomOp>(loc, qbitType, mlir::ValueRange{currentQbit},
                                                  newAttrs.getAttrs());
            currentQbit = newOp.getResult(0); // Chain the result
        }

        // Insert the final resulting qubit back into the register
        mlir::Value qregOut = builder.create<InsertOp>(loc, qregType, qregIn, qbitIndex,
                                                       /*idx_attr=*/nullptr, currentQbit);

        return qregOut;
    };

    // Helper lambda to create a single PPRotationOp.
    // Used for PPR basis.
    auto createPPROp = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> pauliWord,
                           uint16_t rotationKind, mlir::Value qregIn) -> mlir::Value {
        // Extract the qubit
        mlir::Value currentQbit =
            builder.create<ExtractOp>(loc, qbitType, qregIn, qbitIndex, /*idx_attr=*/nullptr);

        // Create the PPRotationOp
        auto pprOp = builder.create<qec::PPRotationOp>(
            loc,
            pauliWord,
            rotationKind,
            mlir::ValueRange{currentQbit},
            /*condition=*/nullptr);

        mlir::Value resultQbit = pprOp.getResult(0);

        // Insert the final resulting qubit back into the register
        mlir::Value qregOut = builder.create<InsertOp>(loc, qregType, qregIn, qbitIndex,
                                                       /*idx_attr=*/nullptr, resultQbit);
        return qregOut;
    };

    // --- END: Lambda Helpers for Op Creation ---

    // --- Define cases based on pprBasis ---
    SmallVector<int64_t> caseValues;
    std::vector<std::pair<llvm::ArrayRef<StringRef>, bool>> caseConfigs;

    if (pprBasis) {
        // PPR Basis cases: {X_pi/8, X_pi/4, Z_pi/8, Z_pi/4}
        caseValues = {0, 1, 2, 3};
        // We don't use caseConfigs for this path, we build cases manually
    }
    else {
        // Standard Clifford+T cases
        caseValues = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        // Add 'static' to give these arrays a permanent lifetime
        static StringRef gates0[] = {"T"};
        static StringRef gates1[] = {"Hadamard", "T"};
        static StringRef gates2[] = {"S", "Hadamard", "T"};
        static StringRef gates3[] = {"Identity"};
        static StringRef gates4[] = {"PauliX"};
        static StringRef gates5[] = {"PauliY"};
        static StringRef gates6[] = {"PauliZ"};
        static StringRef gates7[] = {"Hadamard"};
        static StringRef gates8[] = {"S"}; // Used for both Case 8 and 9

        caseConfigs = {
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
    }

    // Create the switch operation inside the loop
    mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
    auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
        loc, mlir::TypeRange{qregType}, currentGateIndex, caseValuesAttr, caseValues.size());

    // --- Populate Switch Cases ---
    if (pprBasis) {
        // --- Populate PPR Basis Switch Cases ---
        static StringRef xPauli[] = {"X"};
        static StringRef zPauli[] = {"Z"};

        // Case 0: "X" 8 (pi/8)
        Region &caseRegion0 = switchOp.getCaseRegions()[0];
        caseRegion0.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion0.front());
        mlir::Value qregCase0 = createPPROp(rewriter, xPauli, 8, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase0);

        // Case 1: "X" 4 (pi/4)
        Region &caseRegion1 = switchOp.getCaseRegions()[1];
        caseRegion1.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion1.front());
        mlir::Value qregCase1 = createPPROp(rewriter, xPauli, 4, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase1);

        // Case 2: "Z" 8 (pi/8)
        Region &caseRegion2 = switchOp.getCaseRegions()[2];
        caseRegion2.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion2.front());
        mlir::Value qregCase2 = createPPROp(rewriter, zPauli, 8, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase2);

        // Case 3: "Z" 4 (pi/4)
        Region &caseRegion3 = switchOp.getCaseRegions()[3];
        caseRegion3.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion3.front());
        mlir::Value qregCase3 = createPPROp(rewriter, zPauli, 4, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase3);
    }
    else {
        // --- Populate Clifford+T Switch Cases ---
        assert(caseConfigs.size() == caseValues.size() && "Mismatch in case config and case values");
        for (size_t i = 0; i < caseConfigs.size(); ++i) {
            const auto &config = caseConfigs[i];
            Region &caseRegion = switchOp.getCaseRegions()[i];
            caseRegion.push_back(new Block());
            rewriter.setInsertionPointToStart(&caseRegion.front());

            mlir::Value qregCase =
                createGateChain(rewriter, config.first, currentLoopReg, config.second);
            rewriter.create<mlir::scf::YieldOp>(loc, qregCase);
        }
    }

    // Populate Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    // Default to Identity
    static StringRef gatesDefault[] = {"Identity"};
    mlir::Value qregDefault = createGateChain(rewriter, gatesDefault, currentLoopReg);
    rewriter.create<mlir::scf::YieldOp>(loc, qregDefault);

    // Yield the result of the switch op from the for loop
    rewriter.setInsertionPointAfter(switchOp);
    rewriter.create<mlir::scf::YieldOp>(loc, switchOp.getResults());

    // --- Back in function body, after the loop ---
    rewriter.setInsertionPointAfter(forOp);
    mlir::Value finalReg = forOp.getResult(0);

    // Call free_memref
    rewriter.create<mlir::func::CallOp>(loc, freeMemrefFunc, mlir::ValueRange{rankedMemref});

    // Return the final register and the computed runtime phase
    rewriter.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{finalReg, runtimePhase});

    return func;
}

// --- Rewrite Pattern ---

struct DecomposeCustomOpPattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    double EPSILON;
    bool PPR_BASIS;

    DecomposeCustomOpPattern(mlir::MLIRContext *context, double epsilon, bool pprBasis)
        : mlir::OpRewritePattern<CustomOp>(context), EPSILON(epsilon), PPR_BASIS(pprBasis)
    {
    }

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        // Match RZ or PhaseShift
        StringRef gateName = op.getGateName();
        bool isRZ = gateName == "RZ";
        bool isPhaseShift = gateName == "PhaseShift";
        if (!isRZ && !isPhaseShift) {
            return rewriter.notifyMatchFailure(op, "op is not 'RZ' or 'PhaseShift'");
        }
        LLVM_DEBUG(llvm::dbgs() << "Matching op: " << op << "\n");

        // Find Qubit operand and trace it back to its ExtractOp
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
        // RZ or PhaseShift op has exactly one parameter
        if (paramOperands.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Op must have exactly one parameter");
        }
        mlir::Value angle = paramOperands[0];
        if (!mlir::isa<mlir::Float64Type>(angle.getType())) {
            return rewriter.notifyMatchFailure(op, "Op parameter is not f64");
        }

        if (!qbitOperand) {
            return rewriter.notifyMatchFailure(op, "Op has no qubit operand");
        }
        if (op->getNumResults() != 1 ||
            !mlir::isa<catalyst::quantum::QubitType>(op->getResult(0).getType())) {
            return rewriter.notifyMatchFailure(op, "Op does not return a single qubit");
        }

        // find the source ExtractOp for current Qubit
        catalyst::quantum::ExtractOp extractOp = findSourceExtract(qbitOperand);
        if (!extractOp) {
            return rewriter.notifyMatchFailure(op,
                                               "Qubit operand does not trace back to an ExtractOp");
        }

        // Get the register, index, and types
        mlir::Value qregOperand = extractOp.getQreg();
        mlir::Value qbitIndex = extractOp.getIdx();
        mlir::Type qregType = qregOperand.getType();
        mlir::Type qbitType = qbitOperand.getType();

        // Verify the qbitIndex is i64
        if (!mlir::isa<mlir::IntegerType>(qbitIndex.getType()) ||
            mlir::cast<mlir::IntegerType>(qbitIndex.getType()).getWidth() != 64) {
            return rewriter.notifyMatchFailure(
                op, "Traced ExtractOp index is not i64. This shouldn't happen.");
        }

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        mlir::Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        // Insert the input qubit back into the register
        mlir::Value regWithQubit =
            rewriter.create<InsertOp>(loc, qregType, qregOperand, qbitIndex, nullptr, qbitOperand);

        // Replace RZ/PhaseShift with decomposition subroutine
        // Get or declare the decomposition function, passing the basis flag
        mlir::func::FuncOp decompFunc =
            getOrDeclareDecompositionFunc(module, rewriter, EPSILON, PPR_BASIS);

        // Call the decomposition function
        auto callDecompOp = rewriter.create<mlir::func::CallOp>(
            loc, decompFunc, mlir::ValueRange{regWithQubit, qbitIndex, angle});
        mlir::Value finalReg = callDecompOp.getResult(0);
        mlir::Value runtimePhase = callDecompOp.getResult(1);

        // Extract the final qubit from the resulting register
        mlir::Value finalQbitResult =
            rewriter.create<ExtractOp>(loc, qbitType, finalReg, qbitIndex, /*idx_attr=*/nullptr);

        // Add GlobalPhaseOp
        rewriter.setInsertionPointAfter(finalQbitResult.getDefiningOp());

        mlir::Value finalPhase;
        if (isPhaseShift) {
            // For PhaseShift, add the original angle to the calculated phase
            finalPhase = rewriter.create<mlir::arith::AddFOp>(loc, runtimePhase, angle);
        }
        else {
            // For RZ, just use the calculated phase
            finalPhase = runtimePhase;
        }

        mlir::NamedAttrList gphaseAttrs;
        gphaseAttrs.append(
            rewriter.getNamedAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 0})));
        rewriter.create<GlobalPhaseOp>(loc, TypeRange{},        // No results
                                       ValueRange{finalPhase},  // Operands
                                       gphaseAttrs.getAttrs()); // Attributes

        // Clean up original op
        rewriter.replaceAllUsesWith(op->getResults(), finalQbitResult);
        rewriter.eraseOp(op);

        LLVM_DEBUG(llvm::dbgs() << "Replaced " << gateName
                                << " op with call to @" << decompFunc.getSymName() << "\n");
        return success();
    }

  private:
    /**
     * @brief Traces a qubit value backward to find its source ExtractOp.
     *
     * This function recursively traces a qubit value back through a chain of
     * unitary CustomOps until it finds the quantum.extract op that it
     * originated from. It assumes that for any CustomOp, the N-th qubit
     * operand maps to the N-th qubit result.
     */
    catalyst::quantum::ExtractOp findSourceExtract(mlir::Value qbit) const
    {
        if (!qbit) {
            return nullptr;
        }

        // Base case: Found the ExtractOp
        if (auto extractOp = qbit.getDefiningOp<ExtractOp>()) {
            LLVM_DEBUG(llvm::dbgs() << "Found source ExtractOp: " << extractOp << "\n");
            return extractOp;
        }

        // Get defining op
        auto definingOp = qbit.getDefiningOp();
        if (!definingOp) {
            return nullptr; // Reached top (e.g., block argument)
        }

        // Recursive case: Look through CustomOp (which includes CNOT, T, H, etc.)
        if (auto customOp = dyn_cast<CustomOp>(definingOp)) {
            // We need to find the index of our `qbit` in the results list.
            auto opResult = mlir::dyn_cast<mlir::OpResult>(qbit);
            if (!opResult) {
                // This should not happen if qbit is the result of an op.
                return nullptr;
            }

            // Get the index of the result we are tracing.
            unsigned resultIndex = opResult.getResultNumber();

            // Get all qubit operands and results.
            std::vector<mlir::Value> qubitOperands = customOp.getQubitOperands();
            std::vector<mlir::OpResult> qubitResults = customOp.getQubitResults();

            // For unitary ops, the N-th qubit input maps to the N-th qubit output.
            if (qubitOperands.size() != qubitResults.size()) {
                LLVM_DEBUG(llvm::dbgs() << "Op has mismatched qubit operands/results size: "
                                        << *definingOp << "\n");
                return nullptr; // Not a simple unitary mapping
            }

            if (resultIndex >= qubitOperands.size()) {
                // The result index is out of bounds of the qubit operands.
                LLVM_DEBUG(llvm::dbgs()
                           << "Result index is out of bounds for op: " << *definingOp << "\n");
                return nullptr;
            }

            // Get the corresponding qubit operand
            mlir::Value nextQubit = qubitOperands[resultIndex];

            if (nextQubit) {
                LLVM_DEBUG(llvm::dbgs() << "Looking through op: " << *definingOp << " (Result "
                                        << resultIndex << " -> Operand " << resultIndex << ")\n");
                // Recurse
                return findSourceExtract(nextQubit);
            }
        }

        // Default case: Hit an op we don't look through
        LLVM_DEBUG(llvm::dbgs() << "Stopping trace at op: " << *definingOp << "\n");
        return nullptr;
    }
};

void populateRSDecompositionPatterns(RewritePatternSet &patterns, double epsilon, bool pprBasis)
{
    patterns.add<DecomposeCustomOpPattern>(patterns.getContext(), epsilon, pprBasis);
}

struct RSDecompositionPass : impl::RSDecompositionPassBase<RSDecompositionPass> {
    using RSDecompositionPassBase::RSDecompositionPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "Running RSDecompositionPass\n");
        mlir::Operation *module = getOperation();
        mlir::MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);

        // Get options from the pass
        double epsilonVal = epsilon;
        bool pprBasisVal = pprBasis;

        populateRSDecompositionPatterns(patterns, epsilonVal, pprBasisVal);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
