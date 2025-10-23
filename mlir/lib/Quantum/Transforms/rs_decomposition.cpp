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

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RSDECOMPOSITIONPASS
#define GEN_PASS_DECL_RSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

// --- Helper Functions to Declare External Runtime Functions ---

/**
 * @brief Gets or declares the external runtime function `some_func`.
 * MLIR signature: func.func @some_func() -> memref<*xindex>
 */
mlir::func::FuncOp getOrDeclareGetGatesFunc(mlir::ModuleOp module, mlir::PatternRewriter &rewriter)
{
    const char *funcName = "some_func";
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    auto unrankedMemRefType =
        mlir::UnrankedMemRefType::get(rewriter.getIndexType(), 0);
    auto funcType = rewriter.getFunctionType({}, {unrankedMemRefType});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();

    return func;
}

/**
 * @brief Gets or declares the external runtime function `free_memref`.
 * MLIR signature: func.func @free_memref(memref<*xindex>) -> ()
 */
mlir::func::FuncOp getOrDeclareFreeMemrefFunc(mlir::ModuleOp module,
                                              mlir::PatternRewriter &rewriter)
{
    const char *funcName = "free_memref"; // Name of the deallocation function
    auto func = module.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (func) {
        return func;
    }

    // Takes one argument: the unranked memref we got from some_func
    auto unrankedMemRefType =
        mlir::UnrankedMemRefType::get(rewriter.getIndexType(), /*memorySpace=*/0);
    // Returns nothing
    auto funcType = rewriter.getFunctionType({unrankedMemRefType}, {});

    // Insert declaration at the start of the module
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<mlir::func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate(); // Mark as external declaration

    return func;
}

// --- Rewrite Pattern ---

struct DecomposeCustomOpPattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        // 1. Match
        if (op.getGateName() !=
            "RZ") { // *** Make sure this is the correct name you want to match ***
            return rewriter.notifyMatchFailure(op, "op is not 'RZ'");
        }
        LLVM_DEBUG(llvm::dbgs() << "Matching op: " << op << "\n");

        mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
        mlir::Location loc = op.getLoc();
        // Set insertion point *before* the op we are replacing
        rewriter.setInsertionPoint(op);

        // 2. Declare runtime functions
        mlir::func::FuncOp getGatesFunc = getOrDeclareGetGatesFunc(module, rewriter);
        mlir::func::FuncOp freeMemrefFunc =
            getOrDeclareFreeMemrefFunc(module, rewriter); // *** Declare free func ***

        // 3. Call some_func to get memref
        auto callGetGatesOp =
            rewriter.create<mlir::func::CallOp>(loc, getGatesFunc, mlir::ValueRange{});
        mlir::Value unrankedMemref =
            callGetGatesOp.getResult(0); // This is the value holding the allocated memory info

        // 4. Cast and load first element
        mlir::Type indexType = rewriter.getIndexType();
        mlir::Type rankedType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, indexType);
        mlir::Value rankedMemref =
            rewriter.create<mlir::memref::CastOp>(loc, rankedType, unrankedMemref);
        mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value firstGateIndex =
            rewriter.create<mlir::memref::LoadOp>(loc, rankedMemref, mlir::ValueRange{c0});

        // 5. Create the switch operation
        SmallVector<int64_t> caseValues = {0, 1, 2, 3};
        mlir::DenseI64ArrayAttr caseValuesAttr = rewriter.getDenseI64ArrayAttr(caseValues);
        auto switchOp = rewriter.create<mlir::scf::IndexSwitchOp>(
            loc, op->getResultTypes(), firstGateIndex, caseValuesAttr, caseValues.size());

        // 6. Populate switch cases
        SmallVector<StringRef> gateNames = {"Hadamard", "PauliX", "PauliY", "PauliZ"};
        assert(switchOp.getNumRegions() == gateNames.size() + 1 && "Mismatch regions");

        mlir::NamedAttrList copiedAttrs;
        for (const mlir::NamedAttribute &attr : op->getAttrs()) {
            if (attr.getName() != "gate_name") {
                copiedAttrs.append(attr);
            }
        }

        for (unsigned i = 0; i < gateNames.size(); ++i) {
            Region &caseRegion = switchOp.getCaseRegions()[i];
            if (caseRegion.empty()) {
                caseRegion.push_back(new Block());
            }
            rewriter.setInsertionPointToStart(&caseRegion.front());

            mlir::NamedAttrList newAttrs = copiedAttrs;
            newAttrs.append(
                rewriter.getNamedAttr("gate_name", rewriter.getStringAttr(gateNames[i])));
            auto newOp = rewriter.create<CustomOp>(loc, op->getResultTypes(), op->getOperands(),
                                                   newAttrs.getAttrs());
            rewriter.create<mlir::scf::YieldOp>(loc, newOp->getResults());
        }

        // 7. Populate Default Case
        Region &defaultRegion = switchOp.getDefaultRegion();
        if (defaultRegion.empty()) {
            defaultRegion.push_back(new Block());
        }
        rewriter.setInsertionPointToStart(&defaultRegion.front());
        if (!op->getResultTypes().empty()) {
            mlir::NamedAttrList defaultAttrs = copiedAttrs;
            defaultAttrs.append(
                rewriter.getNamedAttr("gate_name", rewriter.getStringAttr("Identity")));
            auto defaultOp = rewriter.create<CustomOp>(loc, op->getResultTypes(), op->getOperands(),
                                                       defaultAttrs.getAttrs());
            rewriter.create<mlir::scf::YieldOp>(loc, defaultOp->getResults());
        }
        else {
            rewriter.create<mlir::scf::YieldOp>(loc);
        }

        // 8. *** Call free_memref ***
        // Place the call *after* the switch, as the memref value ('unrankedMemref')
        // itself (or derived values like 'rankedMemref') is no longer needed after the load.
        rewriter.setInsertionPointAfter(switchOp); // Ensure insertion point is correct
        rewriter.create<mlir::func::CallOp>(
            loc, freeMemrefFunc,
            mlir::ValueRange{unrankedMemref}); // Pass the value from callGetGatesOp

        // 9. Clean up original op
        rewriter.replaceAllUsesWith(op->getResults(), switchOp.getResults());
        rewriter.eraseOp(op);

        LLVM_DEBUG(
            llvm::dbgs() << "Replaced op with switch structure and added free_memref call.\n");
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
