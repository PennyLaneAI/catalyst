// Copyright 2025 Xanadu Quantum Technologies Inc.

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
#include "Quantum/Transforms/Passes.h" // For pass generation macros

using llvm::SmallVector;
using llvm::StringRef;
using namespace mlir;
using catalyst::quantum::CustomOp;
using catalyst::quantum::ExtractOp;
using catalyst::quantum::InsertOp;
using catalyst::quantum::QubitType;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RSDECOMPOSITIONPASS
#define GEN_PASS_DECL_RSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

// --- Helper Functions to Declare External Runtime Functions ---

/**
 * @brief Gets or declares the external runtime function `some_func_0_get_gates`.
 * MLIR signature: func.func @some_func_0_get_gates() -> memref<?xindex>
 */
mlir::func::FuncOp getOrDeclareGetGatesFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "some_func_get_gates"; // Match C++ name
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

/**
 * @brief Gets or declares the external runtime function `some_func_0_get_val1`.
 * MLIR signature: func.func @some_func_0_get_val1() -> f64
 */
mlir::func::FuncOp getOrDeclareGetVal1Func(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "some_func_get_val1"; // Match C++ name
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type();
    auto funcType = rewriter.getFunctionType({}, {f64Type});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `some_func_0_get_val2`.
 * MLIR signature: func.func @some_func_0_get_val2() -> f64
 */
mlir::func::FuncOp getOrDeclareGetVal2Func(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "some_func_get_val2"; // Match C++ name
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto f64Type = rewriter.getF64Type();
    auto funcType = rewriter.getFunctionType({}, {f64Type});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `free_memref_0`.
 * MLIR signature: func.func @free_memref_0(memref<?xindex>) -> ()
 * (This declaration is correct, MLIR will unbundle the memref argument)
 */
mlir::func::FuncOp getOrDeclareFreeMemrefFunc(mlir::ModuleOp module,
                                              mlir::PatternRewriter &rewriter)
{
    const char *funcName = "free_memref"; // Match C++ name
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
        // 1. Match RZ
        if (op.getGateName() != "RZ") {
            return rewriter.notifyMatchFailure(op, "op is not 'RZ'");
        }
        LLVM_DEBUG(llvm::dbgs() << "Matching op: " << op << "\n");

        // *** Find Qubit operand and trace it back to its ExtractOp ***
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

        // --- Start Trace-back Loop ---
        catalyst::quantum::ExtractOp extractOp;
        mlir::Value currentQubit = qbitOperand;

        while (true) {
            extractOp = currentQubit.getDefiningOp<ExtractOp>();
            if (extractOp) {
                LLVM_DEBUG(llvm::dbgs() << "Found source ExtractOp: " << extractOp << "\n");
                break;
            }
            auto definingOp = currentQubit.getDefiningOp();
            if (!definingOp) {
                return rewriter.notifyMatchFailure(
                    op, "Qubit operand does not trace back to an ExtractOp (reached top)");
            }
            if (auto customOp = dyn_cast<CustomOp>(definingOp)) {
                if (customOp->getNumResults() != 1 ||
                    !mlir::isa<catalyst::quantum::QubitType>(customOp.getResult(0).getType())) {
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
                    currentQubit = nextQubit;
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

        // Get the register, index, and types we'll need
        mlir::Value qregOperand = extractOp.getQreg();
        mlir::Value qbitIndex = extractOp.getIdx();
        mlir::Type qregType = qregOperand.getType();
        mlir::Type qbitType = qbitOperand.getType();

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        mlir::Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        // 2. Declare runtime functions
        mlir::func::FuncOp getGatesFunc = getOrDeclareGetGatesFunc(module, rewriter);
        mlir::func::FuncOp getVal1Func = getOrDeclareGetVal1Func(module, rewriter);
        mlir::func::FuncOp getVal2Func = getOrDeclareGetVal2Func(module, rewriter);
        mlir::func::FuncOp freeMemrefFunc = getOrDeclareFreeMemrefFunc(module, rewriter);

        // 3. Call runtime functions
        auto callGetGatesOp = rewriter.create<mlir::func::CallOp>(
            loc, getGatesFunc, mlir::ValueRange{rzParam}); // <-- CHANGED
        auto callGetVal1Op =
            rewriter.create<mlir::func::CallOp>(loc, getVal1Func, mlir::ValueRange{});
        auto callGetVal2Op =
            rewriter.create<mlir::func::CallOp>(loc, getVal2Func, mlir::ValueRange{});

        mlir::Value rankedMemref = callGetGatesOp.getResult(0);
        mlir::Value doubleVal1 = callGetVal1Op.getResult(0);
        mlir::Value doubleVal2 = callGetVal2Op.getResult(0);
        (void)doubleVal1; // Explicitly mark as unused
        (void)doubleVal2; // Explicitly mark as unused

        // 4. Get size, and create loop bounds
        mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        mlir::Value memrefSize = rewriter.create<mlir::memref::DimOp>(loc, rankedMemref, c0);

        // 5. Insert the qubit operand
        mlir::Value regBeforeLoop = rewriter.create<InsertOp>(loc, qregType, qregOperand, qbitIndex,
                                                              /*idx_attr=*/nullptr, qbitOperand);

        // 6. Create the scf.for loop
        auto forOp =
            rewriter.create<mlir::scf::ForOp>(loc, c0, memrefSize, c1, ValueRange{regBeforeLoop});

        mlir::OpBuilder::InsertionGuard loopGuard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        mlir::Value iv = forOp.getInductionVar();
        mlir::Value currentLoopReg = forOp.getRegionIterArg(0);

        // 7. Load the gate index *inside* the loop
        mlir::Value currentGateIndex =
            rewriter.create<mlir::memref::LoadOp>(loc, rankedMemref, ValueRange{iv});

        // 8. Create the switch operation *inside* the loop
        SmallVector<int64_t> caseValues = {0, 1, 2, 3};
        mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
        auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
            loc, mlir::TypeRange{qregType}, currentGateIndex, caseValuesAttr, caseValues.size());

        // 9. Populate switch cases
        SmallVector<StringRef> gateNames = {"Hadamard", "PauliX", "PauliY", "PauliZ"};
        mlir::NamedAttrList copiedAttrs;
        for (const mlir::NamedAttribute &attr : op->getAttrs()) {
            if (attr.getName() != "gate_name") {
                copiedAttrs.append(attr);
            }
        }

        for (unsigned i = 0; i < gateNames.size(); ++i) {
            Region &caseRegion = switchOp.getCaseRegions()[i];
            caseRegion.push_back(new Block());
            rewriter.setInsertionPointToStart(&caseRegion.front());

            mlir::Value qbitToOperateOn = rewriter.create<ExtractOp>(
                loc, qbitType, currentLoopReg, qbitIndex, /*idx_attr=*/nullptr);

            SmallVector<mlir::Value> newOperands = paramOperands;
            newOperands.push_back(qbitToOperateOn);

            mlir::NamedAttrList newAttrs = copiedAttrs;
            newAttrs.append(
                rewriter.getNamedAttr("gate_name", rewriter.getStringAttr(gateNames[i])));
            auto newOp = rewriter.create<CustomOp>(loc, op->getResultTypes(), newOperands,
                                                   newAttrs.getAttrs());
            mlir::Value resultQbit = newOp.getResult(0);

            mlir::Value resultQreg = rewriter.create<InsertOp>(
                loc, qregType, currentLoopReg, qbitIndex, /*idx_attr=*/nullptr, resultQbit);

            rewriter.create<mlir::scf::YieldOp>(loc, resultQreg);
        }

        // 10. Populate Default Case
        Region &defaultRegion = switchOp.getDefaultRegion();
        defaultRegion.push_back(new Block());
        rewriter.setInsertionPointToStart(&defaultRegion.front());

        mlir::Value defaultQbitToOperateOn = rewriter.create<ExtractOp>(
            loc, qbitType, currentLoopReg, qbitIndex, /*idx_attr=*/nullptr);

        SmallVector<mlir::Value> defaultOperands = paramOperands;
        defaultOperands.push_back(defaultQbitToOperateOn);

        mlir::NamedAttrList defaultAttrs = copiedAttrs;
        defaultAttrs.append(rewriter.getNamedAttr("gate_name", rewriter.getStringAttr("Identity")));
        auto defaultOp = rewriter.create<CustomOp>(loc, op->getResultTypes(), defaultOperands,
                                                   defaultAttrs.getAttrs());
        mlir::Value defaultResultQbit = defaultOp.getResult(0);

        mlir::Value defaultResultQreg = rewriter.create<InsertOp>(
            loc, qregType, currentLoopReg, qbitIndex, /*idx_attr=*/nullptr, defaultResultQbit);

        rewriter.create<mlir::scf::YieldOp>(loc, defaultResultQreg);

        // 11. Yield the result of the switch op from the for loop
        rewriter.setInsertionPointAfter(switchOp);
        rewriter.create<mlir::scf::YieldOp>(loc, switchOp.getResults());

        // 12. Call free_memref_0 *after* the loop
        rewriter.setInsertionPointAfter(forOp);
        auto freeCall = rewriter.create<mlir::func::CallOp>(loc, freeMemrefFunc,
                                                            mlir::ValueRange{rankedMemref});

        // 13. Extract final qbit from loop's result qreg
        rewriter.setInsertionPointAfter(freeCall);
        mlir::Value finalReg = forOp.getResult(0);
        mlir::Value finalQbitResult =
            rewriter.create<ExtractOp>(loc, qbitType, finalReg, qbitIndex, /*idx_attr=*/nullptr);

        // 14. Clean up original op
        rewriter.replaceAllUsesWith(op->getResults(), finalQbitResult);
        rewriter.eraseOp(op);

        LLVM_DEBUG(llvm::dbgs() << "Replaced op with scf.for loop and added free_memref call.\n");
        return success();
    }
};

// --- Pass Definition ---

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
