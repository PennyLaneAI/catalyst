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

#define DEBUG_TYPE "rs-decomposition-patterns"

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

#include "Catalyst/IR/CatalystDialect.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h" // Import pattern declarations

using llvm::SmallVector;
using llvm::StringRef;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

// --- Helper Functions to Declare External Runtime Functions ---

/**
 * @brief Generic helper to find or insert a private function declaration.
 */
mlir::func::FuncOp getOrDeclareFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter,
                                    StringRef funcName, mlir::FunctionType funcType)
{
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `rs_decomposition_get_size`.
 */
mlir::func::FuncOp getOrDeclareGetSizeFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "rs_decomposition_get_size";
    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type();
    auto indexType = rewriter.getIndexType();
    auto funcType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {indexType});

    return getOrDeclareFunc(module, rewriter, funcName, funcType);
}

/**
 * @brief Gets or declares the external runtime function `rs_decomposition_get_gates`.
 */
mlir::func::FuncOp getOrDeclareGetGatesFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "rs_decomposition_get_gates";
    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type();
    auto rankedMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    // Signature: (memref<* x index>, f64, f64, i1) -> ()
    auto funcType = rewriter.getFunctionType({rankedMemRefType, f64Type, f64Type, i1Type}, {});

    return getOrDeclareFunc(module, rewriter, funcName, funcType);
}

/**
 * @brief Gets or declares the external runtime function `rs_decomposition_get_phase`.
 */
mlir::func::FuncOp getOrDeclareGetPhaseFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "rs_decomposition_get_phase";
    auto f64Type = rewriter.getF64Type();
    auto i1Type = rewriter.getI1Type();
    auto funcType = rewriter.getFunctionType({f64Type, f64Type, i1Type}, {f64Type});

    return getOrDeclareFunc(module, rewriter, funcName, funcType);
}

// --- Helper Functions to build switch cases ---

/**
 * @brief Populates the scf.index_switch op for the Clifford+T basis.
 */
void populateCliffordTSwitchCases(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                  mlir::scf::IndexSwitchOp switchOp, mlir::Value qregIn,
                                  mlir::Value qbitIndex)
{
    auto qregType = qregIn.getType();
    auto qbitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Helper lambda to create a chain of parameter-less single qubit CUSTOM gates.
    auto createGateChain = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> gateNames,
                               mlir::Value currentLoopReg, bool isAdjoint = false) -> mlir::Value {
        mlir::Value currentQbit = builder.create<ExtractOp>(loc, qbitType, currentLoopReg,
                                                            qbitIndex, /*idx_attr=*/nullptr);

        for (StringRef gateName : gateNames) {
            mlir::NamedAttrList newAttrs;
            newAttrs.append(builder.getNamedAttr("gate_name", builder.getStringAttr(gateName)));
            newAttrs.append(builder.getNamedAttr("operandSegmentSizes",
                                                 builder.getDenseI32ArrayAttr({0, 1, 0, 0})));
            if (isAdjoint) {
                newAttrs.append(builder.getNamedAttr("adjoint", builder.getUnitAttr()));
            }
            newAttrs.append(
                builder.getNamedAttr("resultSegmentSizes", builder.getDenseI32ArrayAttr({1, 0})));

            auto newOp = builder.create<CustomOp>(loc, qbitType, mlir::ValueRange{currentQbit},
                                                  newAttrs.getAttrs());
            currentQbit = newOp.getResult(0);
        }

        mlir::Value qregOut = builder.create<InsertOp>(loc, qregType, currentLoopReg, qbitIndex,
                                                       /*idx_attr=*/nullptr, currentQbit);
        return qregOut;
    };

    // --- Define cases ---
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

    // --- Populate Switch Cases ---
    assert(caseConfigs.size() == switchOp.getCases().size() &&
           "Mismatch in case config and case values");
    for (size_t i = 0; i < caseConfigs.size(); ++i) {
        const auto &config = caseConfigs[i];
        Region &caseRegion = switchOp.getCaseRegions()[i];
        caseRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion.front());

        mlir::Value qregCase = createGateChain(rewriter, config.first, qregIn, config.second);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase);
    }

    // Populate Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    static StringRef gatesDefault[] = {"Identity"};
    mlir::Value qregDefault = createGateChain(rewriter, gatesDefault, qregIn);
    rewriter.create<mlir::scf::YieldOp>(loc, qregDefault);
}

/**
 * @brief Populates the scf.index_switch op for the PPR-Basis.
 */
void populatePPRBasisSwitchCases(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                 mlir::scf::IndexSwitchOp switchOp, mlir::Value qregIn,
                                 mlir::Value qbitIndex)
{
    auto qregType = qregIn.getType();
    auto qbitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Helper lambda to create a single PPRotationOp.
    auto createPPROp = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> pauliWord,
                           uint16_t rotationKind, mlir::Value currentLoopReg) -> mlir::Value {
        mlir::Value currentQbit = builder.create<ExtractOp>(loc, qbitType, currentLoopReg,
                                                            qbitIndex, /*idx_attr=*/nullptr);

        auto pprOp = builder.create<catalyst::qec::PPRotationOp>(loc, pauliWord, rotationKind,
                                                                 mlir::ValueRange{currentQbit},
                                                                 /*condition=*/nullptr);

        mlir::Value resultQbit = pprOp.getResult(0);

        mlir::Value qregOut = builder.create<InsertOp>(loc, qregType, currentLoopReg, qbitIndex,
                                                       /*idx_attr=*/nullptr, resultQbit);
        return qregOut;
    };

    // --- Define cases ---
    static StringRef xPauli[] = {"X"};
    static StringRef zPauli[] = {"Z"};

    // Case 0: "X" 8 (pi/8)
    Region &caseRegion0 = switchOp.getCaseRegions()[0];
    caseRegion0.push_back(new Block());
    rewriter.setInsertionPointToStart(&caseRegion0.front());
    mlir::Value qregCase0 = createPPROp(rewriter, xPauli, 8, qregIn);
    rewriter.create<mlir::scf::YieldOp>(loc, qregCase0);

    // Case 1: "X" 4 (pi/4)
    Region &caseRegion1 = switchOp.getCaseRegions()[1];
    caseRegion1.push_back(new Block());
    rewriter.setInsertionPointToStart(&caseRegion1.front());
    mlir::Value qregCase1 = createPPROp(rewriter, xPauli, 4, qregIn);
    rewriter.create<mlir::scf::YieldOp>(loc, qregCase1);

    // Case 2: "Z" 8 (pi/8)
    Region &caseRegion2 = switchOp.getCaseRegions()[2];
    caseRegion2.push_back(new Block());
    rewriter.setInsertionPointToStart(&caseRegion2.front());
    mlir::Value qregCase2 = createPPROp(rewriter, zPauli, 8, qregIn);
    rewriter.create<mlir::scf::YieldOp>(loc, qregCase2);

    // Case 3: "Z" 4 (pi/4)
    Region &caseRegion3 = switchOp.getCaseRegions()[3];
    caseRegion3.push_back(new Block());
    rewriter.setInsertionPointToStart(&caseRegion3.front());
    mlir::Value qregCase3 = createPPROp(rewriter, zPauli, 4, qregIn);
    rewriter.create<mlir::scf::YieldOp>(loc, qregCase3);

    // Populate Default Case
    Region &defaultRegion = switchOp.getDefaultRegion();
    defaultRegion.push_back(new Block());
    rewriter.setInsertionPointToStart(&defaultRegion.front());
    // Default to Identity (do nothing)
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

    // Declare runtime functions
    mlir::func::FuncOp getSizeFunc = getOrDeclareGetSizeFunc(module, rewriter);
    mlir::func::FuncOp getGatesFunc = getOrDeclareGetGatesFunc(module, rewriter);
    mlir::func::FuncOp getPhaseFunc = getOrDeclareGetPhaseFunc(module, rewriter);

    mlir::Value epsilonVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(epsilon));
    mlir::Value pprBasisVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(pprBasis));

    // Call runtime functions
    // Get required memref size for returning result
    auto callGetSizeOp = rewriter.create<mlir::func::CallOp>(
        loc, getSizeFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});
    mlir::Value num_gates = callGetSizeOp.getResult(0);

    // Alloca memref
    auto gatesMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    mlir::Value gatesMemref =
        rewriter.create<mlir::memref::AllocaOp>(loc, gatesMemRefType, ValueRange{num_gates});

    // Compute gate sequence into the allocated memref
    rewriter.create<mlir::func::CallOp>(
        loc, getGatesFunc, mlir::ValueRange{gatesMemref, angle, epsilonVal, pprBasisVal});

    // Get phase
    auto callGetPhaseOp = rewriter.create<mlir::func::CallOp>(
        loc, getPhaseFunc, mlir::ValueRange{angle, epsilonVal, pprBasisVal});
    mlir::Value runtimePhase = callGetPhaseOp.getResult(0);

    // Create the scf.for loop over gate sequence indices
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value memrefSize = num_gates;
    auto forOp = rewriter.create<mlir::scf::ForOp>(loc, c0, memrefSize, c1, ValueRange{qregIn});

    mlir::OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    mlir::Value iv = forOp.getInductionVar();
    mlir::Value currentLoopReg = forOp.getRegionIterArg(0);

    // Load the gate index inside the loop
    mlir::Value currentGateIndex =
        rewriter.create<mlir::memref::LoadOp>(loc, gatesMemref, ValueRange{iv});

    // --- Define cases based on pprBasis ---
    SmallVector<int64_t> caseValues;
    if (pprBasis) {
        caseValues = {0, 1, 2, 3}; // PPR Basis cases
    }
    else {
        caseValues = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // Standard Clifford+T cases
    }

    // Create the switch operation inside the loop
    mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
    auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
        loc, mlir::TypeRange{qregType}, currentGateIndex, caseValuesAttr, caseValues.size());

    // --- Populate Switch Cases ---
    if (pprBasis) {
        populatePPRBasisSwitchCases(rewriter, loc, switchOp, currentLoopReg, qbitIndex);
    }
    else {
        populateCliffordTSwitchCases(rewriter, loc, switchOp, currentLoopReg, qbitIndex);
    }

    // Yield the result of the switch op from the for loop
    rewriter.setInsertionPointAfter(switchOp);
    rewriter.create<mlir::scf::YieldOp>(loc, switchOp.getResults());

    // --- Back in function body, after the loop ---
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
        // find the source ExtractOp for current Qubit
        catalyst::quantum::ExtractOp extractOp = findSourceExtract(qbitOperand);
        if (!extractOp) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Failed to find source ExtractOp for: " << qbitOperand << "\n");
            return rewriter.notifyMatchFailure(op,
                                               "Qubit operand does not trace back to an ExtractOp");
        }
        LLVM_DEBUG(llvm::dbgs() << "Found source ExtractOp: " << extractOp << "\n");

        // Get the register, index, and types
        mlir::Value qregOperand = extractOp.getQreg();
        if (!qregOperand) {
            return rewriter.notifyMatchFailure(op, "Source ExtractOp has a null qreg");
        }
        LLVM_DEBUG(llvm::dbgs() << "Found source qreg: " << qregOperand << "\n");

        mlir::Value qbitIndex; // This must be an i64 Value

        // We set the insertion point at the ExtractOp to insert a new
        // constant index if needed.
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(extractOp);

        if (extractOp.getIdx()) {
            // Case 1: Dynamic index (e.g., %idx)
            qbitIndex = extractOp.getIdx();
            LLVM_DEBUG(llvm::dbgs() << "Found dynamic index: " << qbitIndex << "\n");
        }
        else if (extractOp.getIdxAttr()) {
            // Case 2: Static index (e.g., [ 0])
            // We need to create a constant op for this index.
            int64_t staticIndex = *extractOp.getIdxAttr();
            LLVM_DEBUG(llvm::dbgs() << "Found static index attr: " << staticIndex << "\n");
            qbitIndex = rewriter.create<mlir::arith::ConstantOp>(
                extractOp.getLoc(), rewriter.getI64IntegerAttr(staticIndex));
            LLVM_DEBUG(llvm::dbgs() << "Created new constant index op: " << qbitIndex << "\n");
        }
        else {
            return rewriter.notifyMatchFailure(
                op, "Source ExtractOp has neither dynamic nor static index");
        }

        mlir::Type qregType = qregOperand.getType();
        mlir::Type qbitType = qbitOperand.getType();

        // Verify the qbitIndex is i64
        mlir::Type indexType = qbitIndex.getType();
        if (!indexType) {
            // This check is almost certainly redundant, but good to have
            return rewriter.notifyMatchFailure(op, "Source index has a null type");
        }
        if (!mlir::isa<mlir::IntegerType>(indexType) ||
            mlir::cast<mlir::IntegerType>(indexType).getWidth() != 64) {
            return rewriter.notifyMatchFailure(
                op, "Traced ExtractOp index is not i64. This shouldn't happen.");
        }
        LLVM_DEBUG(llvm::dbgs() << "Index type is i64, proceeding.\n");

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        if (!module) {
            return rewriter.notifyMatchFailure(op, "op is not contained within a ModuleOp");
        }
        mlir::Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        // Insert the input qubit back into the register
        mlir::Value regWithQubit =
            rewriter.create<InsertOp>(loc, qregType, qregOperand, qbitIndex, nullptr, qbitOperand);

        // Replace RZ/PhaseShift with decomposition subroutine
        // Get or declare the decomposition function, passing the basis flag
        mlir::func::FuncOp decompFunc =
            getOrCreateDecompositionFunc(module, rewriter, epsilon, pprBasis);

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

        LLVM_DEBUG(llvm::dbgs() << "Replaced " << gateName << " op with call to @"
                                << decompFunc.getSymName() << "\n");
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

        // Recursive case: Look through CustomOp
        if (auto customOp = dyn_cast<CustomOp>(definingOp)) {
            auto opResult = mlir::dyn_cast<mlir::OpResult>(qbit);
            if (!opResult) {
                return nullptr;
            }

            std::vector<mlir::Value> qubitOperands = customOp.getQubitOperands();
            std::vector<mlir::OpResult> qubitResults = customOp.getQubitResults();

            // For unitary ops, the N-th qubit input maps to the N-th qubit output.
            if (qubitOperands.size() != qubitResults.size()) {
                LLVM_DEBUG(llvm::dbgs() << "Op has mismatched qubit operands/results size: "
                                        << *definingOp << "\n");
                return nullptr; // Not a simple unitary mapping
            }

            // --- START: Corrected Logic ---
            // Find the index of our `qbit` in the *qubit results* list.
            int64_t qubitResultIndex = -1;
            for (size_t i = 0; i < qubitResults.size(); ++i) {
                if (qubitResults[i] == opResult) {
                    qubitResultIndex = i;
                    break;
                }
            }

            if (qubitResultIndex == -1) {
                // The value we are tracing is not a qubit result of this CustomOp.
                LLVM_DEBUG(llvm::dbgs() << "Stopping trace: Value is not a qubit result of op: "
                                        << *definingOp << "\n");
                return nullptr;
            }

            // Use this index to get the corresponding qubit operand
            mlir::Value nextQubit = qubitOperands[qubitResultIndex];
            // --- END: Corrected Logic ---

            if (nextQubit) {
                LLVM_DEBUG(llvm::dbgs() << "Looking through op: " << *definingOp
                                        << " (Qubit Result " << qubitResultIndex
                                        << " -> Qubit Operand " << qubitResultIndex << ")\n");
                // Recurse
                return findSourceExtract(nextQubit);
            }
        }

        // Default case: Hit an op we don't look through
        LLVM_DEBUG(llvm::dbgs() << "Stopping trace at op: " << *definingOp << "\n");
        return nullptr;
    }
};

} // anonymous namespace

namespace catalyst {
namespace quantum {

// This is the public function defined in Quantum/Transforms/Patterns.h
void populateRSDecompositionPatterns(RewritePatternSet &patterns, double epsilon, bool pprBasis)
{
    patterns.add<DecomposeCustomOpPattern>(patterns.getContext(), epsilon, pprBasis);
}

} // namespace quantum
} // namespace catalyst
