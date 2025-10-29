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
#include "Quantum/Transforms/Passes.h"

using llvm::SmallVector;
using llvm::StringRef;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RSDECOMPOSITIONPASS
#define GEN_PASS_DECL_RSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

// --- Helper Functions to Declare External Runtime Functions ---

/**
 * @brief Gets or declares the external runtime function `some_func_0_get_gates`.
 * MLIR signature: func.func @rs_decomposition_0() -> memref<?xindex>
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
    auto funcType = rewriter.getFunctionType({f64Type}, {rankedMemRefType});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

// /**
//  * @brief Gets or declares the external runtime function `some_func_0_get_val1`.
//  * MLIR signature: func.func @some_func_0_get_val1() -> f64
//  * Will be used for global phase
//  */
// mlir::func::FuncOp getOrDeclareGetVal1Func(mlir::ModuleOp module, mlir::PatternRewriter
// &rewriter)
// {
//     const char *funcName = "some_func_get_val1"; // Match C++ name
//     auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
//     if (func) {
//         return func;
//     }

//     auto f64Type = rewriter.getF64Type();
//     auto funcType = rewriter.getFunctionType({}, {f64Type});

//     mlir::OpBuilder::InsertionGuard guard(rewriter);
//     rewriter.setInsertionPointToStart(module.getBody());
//     func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
//     func.setPrivate();

//     return func;
// }

/**
 * @brief Gets or declares the external runtime function `free_memref_0`.
 * MLIR signature: func.func @free_memref_0(memref<?xindex>) -> ()
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
        if (paramOperands.size() != 1) {
            return rewriter.notifyMatchFailure(op, "RZ op must have exactly one parameter");
        }
        mlir::Value rzParam = paramOperands[0];
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
        // mlir::func::FuncOp getVal1Func = getOrDeclareGetVal1Func(module, rewriter);
        mlir::func::FuncOp freeMemrefFunc = getOrDeclareFreeMemrefFunc(module, rewriter);

        // Call runtime functions
        auto callGetGatesOp = rewriter.create<mlir::func::CallOp>(
            loc, getGatesFunc, mlir::ValueRange{rzParam}); // <-- Pass RZ parameter
        // auto callGetVal1Op =
        //     rewriter.create<mlir::func::CallOp>(loc, getVal1Func, mlir::ValueRange{});

        mlir::Value rankedMemref = callGetGatesOp.getResult(0);
        // mlir::Value doubleVal1 = callGetVal1Op.getResult(0);
        // (void)doubleVal1; // Explicitly mark as unused

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

        // Case 0: "T"
        Region &caseRegion0 = switchOp.getCaseRegions()[0];
        caseRegion0.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion0.front());
        mlir::Value qregCase0 = createGateChain(rewriter, {"T"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase0);

        // Case 1: "Hadamard" + "T"
        Region &caseRegion1 = switchOp.getCaseRegions()[1];
        caseRegion1.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion1.front());
        mlir::Value qregCase1 = createGateChain(rewriter, {"Hadamard", "T"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase1);

        // Case 2: "S" + "Hadamard" + "T"
        Region &caseRegion2 = switchOp.getCaseRegions()[2];
        caseRegion2.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion2.front());
        mlir::Value qregCase2 = createGateChain(rewriter, {"S", "Hadamard", "T"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase2);

        // Case 3: "Identity"
        Region &caseRegion3 = switchOp.getCaseRegions()[3];
        caseRegion3.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion3.front());
        mlir::Value qregCase3 = createGateChain(rewriter, {"Identity"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase3);

        // Case 4: "PauliX"
        Region &caseRegion4 = switchOp.getCaseRegions()[4];
        caseRegion4.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion4.front());
        mlir::Value qregCase4 = createGateChain(rewriter, {"PauliX"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase4);

        // Case 5: "PauliY"
        Region &caseRegion5 = switchOp.getCaseRegions()[5];
        caseRegion5.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion5.front());
        mlir::Value qregCase5 = createGateChain(rewriter, {"PauliY"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase5);

        // Case 6: "PauliZ"
        Region &caseRegion6 = switchOp.getCaseRegions()[6];
        caseRegion6.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion6.front());
        mlir::Value qregCase6 = createGateChain(rewriter, {"PauliZ"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase6);

        // Case 7: "Hadamard"
        Region &caseRegion7 = switchOp.getCaseRegions()[7];
        caseRegion7.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion7.front());
        mlir::Value qregCase7 = createGateChain(rewriter, {"Hadamard"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase7);

        // Case 8: "S"
        Region &caseRegion8 = switchOp.getCaseRegions()[8];
        caseRegion8.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion8.front());
        mlir::Value qregCase8 = createGateChain(rewriter, {"S"}, currentLoopReg);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase8);

        // Case 9: Adjoint<"S">
        Region &caseRegion9 = switchOp.getCaseRegions()[9];
        caseRegion9.push_back(new Block());
        rewriter.setInsertionPointToStart(&caseRegion9.front());
        mlir::Value qregCase9 =
            createGateChain(rewriter, {"S"}, currentLoopReg, /*isAdjoint=*/true);
        rewriter.create<mlir::scf::YieldOp>(loc, qregCase9);

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

        // Call free_memref_0 after the loop
        rewriter.setInsertionPointAfter(forOp);
        auto freeCall = rewriter.create<mlir::func::CallOp>(loc, freeMemrefFunc,
                                                            mlir::ValueRange{rankedMemref});

        // Extract final qbit from loop's result qreg
        rewriter.setInsertionPointAfter(freeCall);
        mlir::Value finalReg = forOp.getResult(0);
        mlir::Value finalQbitResult =
            rewriter.create<ExtractOp>(loc, qbitType, finalReg, qbitIndex, /*idx_attr=*/nullptr);

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

std::unique_ptr<mlir::Pass> createRSDecompositionPass()
{
    return std::make_unique<quantum::RSDecompositionPass>();
}

} // namespace catalyst
