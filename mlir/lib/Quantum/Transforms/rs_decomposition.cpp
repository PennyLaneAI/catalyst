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
 * MLIR signature: func.func @rs_decomposition(f64, f64) -> memref<?xindex>
 */
mlir::func::FuncOp getOrDeclareGetGatesFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "rs_decomposition";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type(); // <-- Argument type
    auto rankedMemRefType =
        mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    // Runtime function takes two f64 inputs
    auto funcType = rewriter.getFunctionType({f64Type, f64Type}, {rankedMemRefType});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `rs_decomposition_get_phase`.
 * MLIR signature: func.func @rs_decomposition_get_phase(f64, f64) -> f64
 * Will be used for global phase
 */
mlir::func::FuncOp getOrDeclareGetPhaseFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "rs_decomposition_get_phase"; // Match C++ name
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type();
    // Runtime function takes two f64 inputs
    auto funcType = rewriter.getFunctionType({f64Type, f64Type}, {f64Type});

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

// --- Rewrite Pattern ---

struct DecomposeCustomOpPattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        // Match RZ
        if (op.getGateName() != "RZ") {
            return rewriter.notifyMatchFailure(op, "op is not 'RZ'");
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
        // RZ op has exactly one parameter
        if (paramOperands.size() != 1) {
            return rewriter.notifyMatchFailure(op, "RZ op must have exactly one parameter");
        }
        mlir::Value rzParam = paramOperands[0];
        // The parameter must be f64
        if (!mlir::isa<mlir::Float64Type>(rzParam.getType())) {
            return rewriter.notifyMatchFailure(op, "RZ parameter is not f64");
        }

        if (!qbitOperand) {
            return rewriter.notifyMatchFailure(op, "RZ op has no qubit operand");
        }
        if (op->getNumResults() != 1 ||
            !mlir::isa<catalyst::quantum::QubitType>(op->getResult(0).getType())) {
            return rewriter.notifyMatchFailure(op, "RZ op does not return a single qubit");
        }

        // Traceback loop to find the ExtractOp
        catalyst::quantum::ExtractOp extractOp;
        mlir::Value currentQbit = qbitOperand;

        while (true) {
            extractOp = currentQbit.getDefiningOp<ExtractOp>();
            if (extractOp) {
                LLVM_DEBUG(llvm::dbgs() << "Found source ExtractOp: " << extractOp << "\n");
                break;
            }
            auto definingOp = currentQbit.getDefiningOp();
            if (!definingOp) {
                return rewriter.notifyMatchFailure(
                    op, "Qubit operand does not trace back to an ExtractOp (reached top)");
            }
            if (auto customOp = dyn_cast<CustomOp>(definingOp)) {
                if (customOp->getNumResults() != 1 ||
                    !mlir::isa<catalyst::quantum::QubitType>(customOp->getResult(0).getType())) {
                    break;
                }
                mlir::Value nextQubit = nullptr;
                for (mlir::Value opnd : customOp.getOperands()) {
                    if (mlir::isa<catalyst::quantum::QubitType>(opnd.getType())) {
                        if (nextQubit) {
                            nextQubit = nullptr;
                            break;
                        }
                        nextQubit = opnd;
                    }
                }
                if (nextQubit) {
                    LLVM_DEBUG(llvm::dbgs() << "Looking through op: " << *definingOp << "\n");
                    currentQbit = nextQubit;
                    continue;
                }
            }
            break;
        }
        // --- End Trace-back Loop ---

        if (!extractOp) {
            return rewriter.notifyMatchFailure(op,
                                               "Qubit operand does not trace back to an ExtractOp");
        }

        // Get the register, index, and types
        mlir::Value qregOperand = extractOp.getQreg();
        mlir::Value qbitIndex = extractOp.getIdx();
        mlir::Type qregType = qregOperand.getType();
        mlir::Type qbitType = qbitOperand.getType();

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        mlir::Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        // Declare runtime functions
        mlir::func::FuncOp getGatesFunc = getOrDeclareGetGatesFunc(module, rewriter);
        mlir::func::FuncOp getPhaseFunc = getOrDeclareGetPhaseFunc(module, rewriter);
        mlir::func::FuncOp freeMemrefFunc = getOrDeclareFreeMemrefFunc(module, rewriter);

        // Call runtime functions, passing the RZ parameter for both arguments
        auto callGetGatesOp = rewriter.create<mlir::func::CallOp>(
            loc, getGatesFunc, mlir::ValueRange{rzParam, rzParam});
        auto callGetPhaseOp = rewriter.create<mlir::func::CallOp>(
            loc, getPhaseFunc, mlir::ValueRange{rzParam, rzParam});

        mlir::Value rankedMemref = callGetGatesOp.getResult(0);
        mlir::Value doubleVal1 = callGetPhaseOp.getResult(0);

        // Insert the qubit operand
        mlir::Value regBeforeLoop = rewriter.create<InsertOp>(loc, qregType, qregOperand, qbitIndex,
                                                              /*idx_attr=*/nullptr, qbitOperand);

        // Create the scf.for loop
        // Get size, and create loop bounds
        mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        mlir::Value memrefSize = rewriter.create<mlir::memref::DimOp>(loc, rankedMemref, c0);
        auto forOp =
            rewriter.create<mlir::scf::ForOp>(loc, c0, memrefSize, c1, ValueRange{regBeforeLoop});

        mlir::OpBuilder::InsertionGuard loopGuard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        mlir::Value iv = forOp.getInductionVar();
        mlir::Value currentLoopReg = forOp.getRegionIterArg(0);

        // Load the gate index inside the loop
        mlir::Value currentGateIndex =
            rewriter.create<mlir::memref::LoadOp>(loc, rankedMemref, ValueRange{iv});

        // Create the switch operation inside the loop
        SmallVector<int64_t> caseValues = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
        auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
            loc, mlir::TypeRange{qregType}, currentGateIndex, caseValuesAttr, caseValues.size());

        // Populate switch cases
        mlir::NamedAttrList copiedAttrs;
        for (const mlir::NamedAttribute &attr : op->getAttrs()) {
            if (attr.getName() != "gate_name" && attr.getName() != "operandSegmentSizes" &&
                attr.getName() != "adjoint") { // Need to fix adjoint case
                copiedAttrs.append(attr);
            }
        }

        // Helper lambda to create a chain of parameter-less gates.
        // It extracts, applies the gate chain, and inserts the result.
        // Returns the new qreg value.
        auto createGateChain = [&](mlir::OpBuilder &builder, mlir::ArrayRef<StringRef> gateNames,
                                   mlir::Value qregIn, bool isAdjoint = false) -> mlir::Value {
            // Extract the qubit
            mlir::Value currentQbit =
                builder.create<ExtractOp>(loc, qbitType, qregIn, qbitIndex, /*idx_attr=*/nullptr);

            // Apply the chain of gates
            for (StringRef gateName : gateNames) {
                mlir::NamedAttrList newAttrs = copiedAttrs;
                newAttrs.append(builder.getNamedAttr("gate_name", builder.getStringAttr(gateName)));

                // Format is [params, qubits, ctrl_wires, ctrl_values]
                auto segmentSizes = rewriter.getDenseI32ArrayAttr({0, 1, 0, 0});
                newAttrs.append(builder.getNamedAttr("operandSegmentSizes", segmentSizes));

                // Add adjoint flag if isAdjoint is true
                if (isAdjoint) {
                    newAttrs.append(builder.getNamedAttr("adjoint", builder.getUnitAttr()));
                }

                // Create the CustomOp.
                auto newOp = builder.create<CustomOp>(
                    loc, op->getResultTypes(), mlir::ValueRange{currentQbit}, newAttrs.getAttrs());
                currentQbit = newOp.getResult(0); // Chain the result
            }

            // Insert the final resulting qubit back into the register
            mlir::Value qregOut = builder.create<InsertOp>(loc, qregType, qregIn, qbitIndex,
                                                           /*idx_attr=*/nullptr, currentQbit);

            return qregOut;
        };

        StringRef gates0[] = {"T"};
        StringRef gates1[] = {"Hadamard", "T"};
        StringRef gates2[] = {"S", "Hadamard", "T"};
        StringRef gates3[] = {"Identity"};
        StringRef gates4[] = {"PauliX"};
        StringRef gates5[] = {"PauliY"};
        StringRef gates6[] = {"PauliZ"};
        StringRef gates7[] = {"Hadamard"};
        StringRef gates8[] = {"S"}; // Used for both Case 8 and 9

        // Map case indices to their gate chain config (gates, isAdjoint)
        std::vector<std::pair<llvm::ArrayRef<StringRef>, bool>> caseConfigs = {
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

        // Ensure our config vector matches the number of cases we told the switch op
        assert(caseConfigs.size() == caseValues.size() &&
               "Mismatch in case config and case values");

        // --- Populate Switch Cases ---
        for (size_t i = 0; i < caseConfigs.size(); ++i) {
            const auto &config = caseConfigs[i];
            Region &caseRegion = switchOp.getCaseRegions()[i];
            caseRegion.push_back(new Block());
            rewriter.setInsertionPointToStart(&caseRegion.front());

            mlir::Value qregCase =
                createGateChain(rewriter, config.first, currentLoopReg, config.second);
            rewriter.create<mlir::scf::YieldOp>(loc, qregCase);
        }

        // Populate Default Case : Might want to update
        Region &defaultRegion = switchOp.getDefaultRegion();
        defaultRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&defaultRegion.front());

        // Default to Identity
        mlir::Value qregDefault = createGateChain(rewriter, {"Identity"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregDefault);

        // Yield the result of the switch op from the for loop
        rewriter.setInsertionPointAfter(switchOp);
        rewriter.create<mlir::scf::YieldOp>(loc, switchOp.getResults());

        // Call free_memref after the loop
        rewriter.setInsertionPointAfter(forOp);
        auto freeCall = rewriter.create<mlir::func::CallOp>(loc, freeMemrefFunc,
                                                            mlir::ValueRange{rankedMemref});

        // Extract final qbit from loop's result qreg
        rewriter.setInsertionPointAfter(freeCall);
        mlir::Value finalReg = forOp.getResult(0);
        mlir::Value finalQbitResult =
            rewriter.create<ExtractOp>(loc, qbitType, finalReg, qbitIndex, /*idx_attr=*/nullptr);

        // Add the GlobalPhaseOp after the final extract
        rewriter.setInsertionPointAfter(finalQbitResult.getDefiningOp());

        mlir::NamedAttrList gphaseAttrs;
        // Set operandSegmentSizes: [params, in_ctrl_qubits, in_ctrl_values]
        gphaseAttrs.append(
            rewriter.getNamedAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({1, 0, 0})));

        rewriter.create<GlobalPhaseOp>(loc, TypeRange{},        // No results
                                       ValueRange{doubleVal1},  // Operands
                                       gphaseAttrs.getAttrs()); // Attributes

        // Clean up original op
        rewriter.replaceAllUsesWith(op->getResults(), finalQbitResult);
        rewriter.eraseOp(op);

        LLVM_DEBUG(llvm::dbgs() << "Replaced op with scf.for loop and added free_memref call.\n");
        return success();
    }
};

void populateRSDecompositionPatterns(RewritePatternSet &patterns)
{
    patterns.add<DecomposeCustomOpPattern>(patterns.getContext());
}

struct RSDecompositionPass : impl::RSDecompositionPassBase<RSDecompositionPass> {
    using RSDecompositionPassBase::RSDecompositionPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "Running RSDecompositionPass\n");
        mlir::Operation *module = getOperation();
        mlir::MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        populateRSDecompositionPatterns(patterns);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
