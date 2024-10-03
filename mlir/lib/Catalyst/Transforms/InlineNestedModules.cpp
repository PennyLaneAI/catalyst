// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * High level overview:
 * The pass is split into the following phases:
 *
 *  1. For functions inside nested modules, annotate them
 *     with a discardable attribute catalyst.fully_qualified_name
 *     that corresponds to the name of the function as it may be reached
 *     from the module in which the pass was scheduled. E.g.,
 *
 *     quantum-opt --pass-pipeline=builtin.module(inline-nested-module{stop-after-step=1})
 *
 *     ```mlir
 *     module @foo {
 *       module @bar {
 *         func.func @baz()
 *       }
 *     }
 *     ```
 *
 *     results in
 *
 *     ```mlir
 *     module @foo {
 *       module @bar {
 *         func.func @baz() attributes { catalyst.fully_qualified_name = @bar::@baz }
 *       }
 *     }
 *     ```
 *
 * 2. Functions are renamed to be unique across all nested modules and the root module.
 * 3. Nested modules are inlined.
 * 4. Calls to functions that were inlined are replaced with the new inlined functions.
 *
 *     ```mlir
 *     module @foo {
 *       func.func @baz() attributes { catalyst.fully_qualified_name = @bar::@baz }
 *       catalyst.launch_kernel @bar::baz() : () : ()
 *     }
 *     ```
 *     results in
 *
 *     ```mlir
 *     module @foo {
 *       func.func @baz() attributes { catalyst.fully_qualified_name = @bar::@baz }
 *       func.call @baz()
 *     }
 *     ```
 * 5. Cleanup: remove the catalyst.fully_qualified_name attribute
 *
 */

#include <deque>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"
#include "Gradient/IR/GradientInterfaces.h"
#include "Mitigation/IR/MitigationOps.h"

using namespace mlir;

namespace {

/* Will recursively traverse parents until parent == op.
 * Upon execution, it will store the parent name in the first
 * position of the hierarchy deque.
 *
 * This means that the return value is hierarchy
 * where each element of hierarchy is a flat symbol reference
 * corresponding to parent operations not including op itself.
 *
 * E.g.,
 *
 * ```mlir
 * module @foo {
 *   module @bar {
 *     func @baz()
 *   }
 * }
 * ```
 *
 * calling getFullyQualifiedName(@baz, @foo, output)
 * will finish executing when:
 * output = { @bar, @baz }
 */
void getFullyQualifiedName(SymbolOpInterface symbol, const Operation *op,
                           std::deque<FlatSymbolRefAttr> &hierarchy)
{
    auto stringRef = symbol.getNameAttr();
    auto flatSymbolRef = SymbolRefAttr::get(stringRef);
    hierarchy.push_front(flatSymbolRef);

    auto parent = symbol->getParentOp();
    auto parentSymbol = dyn_cast<SymbolOpInterface>(parent);
    auto parentIsLimit = parent == op;
    auto isValidParent = parent && parentSymbol && !parentIsLimit;
    if (!isValidParent)
        return;

    getFullyQualifiedName(parentSymbol, op, hierarchy);
}

/* Obtains the fully qualified name for a symbol up to but not including op.
 *
 * E.g.,
 *
 * ```mlir
 * module @foo {
 *   module @bar {
 *     func @baz()
 *   }
 * }
 * ```
 * getFullyQualifiedNameUntil(@baz, @foo) = @bar::@baz
 */
SymbolRefAttr getFullyQualifiedNameUntil(SymbolOpInterface symbol, const Operation *op)
{
    auto symbolTable = symbol->getParentOp();
    assert(symbolTable->hasTrait<OpTrait::SymbolTable>() &&
           "symbolTable must have OpTrait::SymbolTable");

    std::deque<FlatSymbolRefAttr> hierarchy;
    getFullyQualifiedName(symbol, op, hierarchy);
    assert(hierarchy.size() > 0 && "At least one symbol is expected here.");
    bool inNamelessModule = hierarchy.size() == 1;
    if (inNamelessModule) {
        return SymbolRefAttr::get(symbol);
    }

    FlatSymbolRefAttr root = hierarchy.front();
    hierarchy.pop_front();
    SmallVector<FlatSymbolRefAttr> hierarchy_vector(hierarchy.begin(), hierarchy.end());
    auto ctx = symbol.getContext();
    return SymbolRefAttr::get(ctx, root.getValue(), hierarchy_vector);
}

static constexpr llvm::StringRef fullyQualifiedNameAttr = "catalyst.fully_qualified_name";

struct AnnotateWithFullyQualifiedName : public OpInterfaceRewritePattern<SymbolOpInterface> {
    using OpInterfaceRewritePattern<SymbolOpInterface>::OpInterfaceRewritePattern;

    /// This overload constructs a pattern that matches any operation type.
    AnnotateWithFullyQualifiedName(MLIRContext *context, Operation *root)
        : OpInterfaceRewritePattern<SymbolOpInterface>::OpInterfaceRewritePattern(context),
          _root(root)
    {
    }

    LogicalResult matchAndRewrite(SymbolOpInterface symbol,
                                  PatternRewriter &rewriter) const override
    {
        auto hasQualifiedName = symbol->hasAttr(fullyQualifiedNameAttr);
        if (hasQualifiedName) {
            return failure();
        }

        auto fullyQualifiedName = getFullyQualifiedNameUntil(symbol, _root);
        rewriter.modifyOpInPlace(
            symbol, [&] { symbol->setAttr(fullyQualifiedNameAttr, fullyQualifiedName); });
        return success();
    }

    const Operation *_root;
};

struct RenameFunctionsPattern : public RewritePattern {
    /// This overload constructs a pattern that matches any operation type.
    RenameFunctionsPattern(MLIRContext *context, SmallVector<Operation *> *symbolTables)
        : RewritePattern(MatchAnyOpTypeTag(), 1, context), _symbolTables(symbolTables)
    {
    }

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override;

    SmallVector<Operation *> *_symbolTables;
};

static constexpr llvm::StringRef hasBeenRenamedAttrName = "catalyst.unique_names";

LogicalResult RenameFunctionsPattern::matchAndRewrite(Operation *child,
                                                      PatternRewriter &rewriter) const
{
    bool isSymbolTable = child->hasTrait<OpTrait::SymbolTable>();
    bool hasBeenRenamed = child->hasAttr(hasBeenRenamedAttrName);
    // TODO: isQnode
    bool mustRename = isSymbolTable && !hasBeenRenamed;
    if (!mustRename) {
        return failure();
    }

    assert(child->hasTrait<OpTrait::SymbolTable>() && "child must be symbol table");
    auto parent = child->getParentOp();
    assert(parent && "parent must exist");
    assert(parent->hasTrait<OpTrait::SymbolTable>() && "parent must be a symbol table");

    SymbolTable childSymTab(child);
    SmallVector<std::shared_ptr<SymbolTable>> tables;
    SmallVector<SymbolTable *> raw_tables;
    for (auto operation : *_symbolTables) {
        std::shared_ptr<SymbolTable> sym(new SymbolTable(operation));
        tables.push_back(sym);
        raw_tables.push_back(sym.get());
    }

    // Yes, this is a triple nested loop, but it is faster
    // than doing a post-order walk because we are not interested in
    // nested regions nor blocks. Also, since child is expected to be a module
    // then the number of regions is guaranteed to be one and the number of blocks
    // is also guaranteed to be one.
    for (auto &region : child->getRegions()) {
        for (auto &block : region.getBlocks()) {
            for (auto &op : block) {
                if (!isa<SymbolOpInterface>(op))
                    continue;

                if (failed(childSymTab.renameToUnique(&op, raw_tables))) {
                    // TODO: Check for error in one of the tests.
                    // The only case I've seen this is fail is when an operation which is not
                    // a symbol is passed to the function renameToUnique.
                    op.emitError() << "Cannot rename operation";
                    return failure();
                }
            }
        }
    }

    rewriter.modifyOpInPlace(
        child, [&] { child->setAttr(hasBeenRenamedAttrName, rewriter.getUnitAttr()); });
    return success();
}

struct InlineNestedModule : public RewritePattern {
    /// This overload constructs a pattern that matches any operation type.
    InlineNestedModule(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        bool isSymbolTable = op->hasTrait<OpTrait::SymbolTable>();
        // TODO: isQnode
        bool mustInline = isSymbolTable;
        if (!mustInline) {
            return failure();
        }

        auto parent = op->getParentOp();
        // Can't generalize getting a region other than the zero-th one.
        rewriter.inlineRegionBefore(op->getRegion(0), &parent->getRegion(0).back());
        Block *inlinedBlock = &parent->getRegion(0).front();
        Block *originalBlock = &parent->getRegion(0).back();
        Block *after = rewriter.splitBlock(originalBlock, rewriter.getInsertionPoint());
        rewriter.mergeBlocks(inlinedBlock, originalBlock);
        rewriter.mergeBlocks(after, originalBlock);
        rewriter.eraseOp(op);

        return success();
    }
};

struct SymbolReplacerPattern
    : public OpInterfaceRewritePattern<catalyst::gradient::GradientOpInterface> {
    using OpInterfaceRewritePattern<
        catalyst::gradient::GradientOpInterface>::OpInterfaceRewritePattern;

    SymbolReplacerPattern(MLIRContext *context, const DenseMap<SymbolRefAttr, SymbolRefAttr> *map)
        : OpInterfaceRewritePattern<
              catalyst::gradient::GradientOpInterface>::OpInterfaceRewritePattern(context),
          _map(map)
    {
    }

    LogicalResult matchAndRewrite(catalyst::gradient::GradientOpInterface user,
                                  PatternRewriter &rewriter) const override
    {
        auto found = _map->find(user.getCallee()) != _map->end();
        if (!found) {
            return failure();
        }
        auto newSymbolRefAttr = _map->find(user.getCallee())->getSecond();
        rewriter.modifyOpInPlace(user, [&] { user->setAttr("callee", newSymbolRefAttr); });
        return success();
    }

    const DenseMap<SymbolRefAttr, SymbolRefAttr> *_map;
};

struct ZNEReplacerPattern : public OpRewritePattern<catalyst::mitigation::ZneOp> {
    using OpRewritePattern<catalyst::mitigation::ZneOp>::OpRewritePattern;

    ZNEReplacerPattern(MLIRContext *context, const DenseMap<SymbolRefAttr, SymbolRefAttr> *map)
        : OpRewritePattern<catalyst::mitigation::ZneOp>::OpRewritePattern(context), _map(map)
    {
    }

    LogicalResult matchAndRewrite(catalyst::mitigation::ZneOp op,
                                  PatternRewriter &rewriter) const override
    {
        auto found = _map->find(op.getCallee()) != _map->end();
        if (!found) {
            return failure();
        }
        auto newSymbolRefAttr = _map->find(op.getCallee())->getSecond();
        rewriter.modifyOpInPlace(op, [&] { op->setAttr("callee", newSymbolRefAttr); });
        return success();
    }

    const DenseMap<SymbolRefAttr, SymbolRefAttr> *_map;
};

struct NestedToFlatCallPattern : public OpRewritePattern<catalyst::LaunchKernelOp> {
    using OpRewritePattern<catalyst::LaunchKernelOp>::OpRewritePattern;
    /// This overload constructs a pattern that matches any operation type.
    NestedToFlatCallPattern(MLIRContext *context, const DenseMap<SymbolRefAttr, SymbolRefAttr> *map)
        : OpRewritePattern<catalyst::LaunchKernelOp>::OpRewritePattern(context), _map(map)
    {
    }

    LogicalResult matchAndRewrite(catalyst::LaunchKernelOp op,
                                  PatternRewriter &rewriter) const override
    {
        auto found = _map->find(op.getCallee()) != _map->end();
        if (!found) {
            return failure();
        }

        auto newSymbolRefAttr = _map->find(op.getCallee())->getSecond();
        rewriter.replaceOpWithNewOp<func::CallOp>(op, newSymbolRefAttr, op.getResultTypes(),
                                                  op.getOperands());
        return success();
    }

    const DenseMap<SymbolRefAttr, SymbolRefAttr> *_map;
};

struct CleanupPattern : public RewritePattern {
    /// This overload constructs a pattern that matches any operation type.
    CleanupPattern(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        auto hasQualifiedName = op->hasAttr(fullyQualifiedNameAttr);
        if (!hasQualifiedName) {
            return failure();
        }
        rewriter.modifyOpInPlace(op, [&] { op->removeAttr(fullyQualifiedNameAttr); });
        return success();
    }
};

} // namespace

namespace catalyst {

struct AnnotateWithFullyQualifiedNamePass
    : PassWrapper<AnnotateWithFullyQualifiedNamePass, OperationPass<>> {

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasTrait<OpTrait::SymbolTable>();
    }

    void runOnOperation() override
    {
        MLIRContext *context = &getContext();

        // Do not fold to save in compile time.
        GreedyRewriteConfig config;
        config.strictMode = GreedyRewriteStrictness::ExistingOps;
        config.enableRegionSimplification = false;

        RewritePatternSet annotate(context);
        auto root = getOperation();
        auto parent = root->getParentOp();
        annotate.add<AnnotateWithFullyQualifiedName>(context, parent);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(annotate), config))) {
            signalPassFailure();
        }
    }
};

struct InlineNestedSymbolTablePass : PassWrapper<InlineNestedSymbolTablePass, OperationPass<>> {
    int _stopAfterStep;
    InlineNestedSymbolTablePass(int stopAfter) : _stopAfterStep(stopAfter) {}

    void runOnOperation() override
    {
        // Here we are in a root module/symbol table
        // that contains other nested modules/symbol tables.

        MLIRContext *context = &getContext();

        GreedyRewriteConfig config;
        config.strictMode = GreedyRewriteStrictness::ExistingOps;
        config.enableRegionSimplification = false;

        RewritePatternSet renameFunctions(context);

        // Get all symbol tables in current symbol table. Will be useful for making sure that
        // the symbol being rewritten is unique among all symbol tables.
        auto symbolTable = getOperation();
        assert(symbolTable->hasTrait<OpTrait::SymbolTable>() && "operation must be a symbol table");

        // TODO: upstream change to renameToUnique API to take a set
        SmallVector<Operation *> symbolTables;

        symbolTable->walk([&](Operation *op) {
            if (!op->hasTrait<OpTrait::SymbolTable>()) {
                return WalkResult::skip();
            }
            symbolTables.push_back(op);
            return WalkResult::skip();
        });

        renameFunctions.add<RenameFunctionsPattern>(context, &symbolTables);

        bool run = _stopAfterStep >= 2 || _stopAfterStep == 0;
        if (run &&
            failed(applyPatternsAndFoldGreedily(symbolTable, std::move(renameFunctions), config))) {
            signalPassFailure();
        }

        RewritePatternSet inlineNested(context);
        inlineNested.add<InlineNestedModule>(context);
        run = _stopAfterStep >= 3 || _stopAfterStep == 0;
        if (run &&
            failed(applyPatternsAndFoldGreedily(symbolTable, std::move(inlineNested), config))) {
            signalPassFailure();
        }

        mlir::DenseMap<SymbolRefAttr, SymbolRefAttr> old_to_new;
        for (auto &region : symbolTable->getRegions()) {
            for (auto &block : region.getBlocks()) {
                for (auto &op : block) {
                    if (!isa<SymbolOpInterface>(op))
                        continue;
                    auto hasQualifiedName = op.hasAttr(fullyQualifiedNameAttr);
                    if (!hasQualifiedName)
                        continue;

                    SymbolRefAttr old = op.getAttrOfType<SymbolRefAttr>(fullyQualifiedNameAttr);
                    auto symbol = cast<SymbolOpInterface>(op);
                    SymbolRefAttr _new = SymbolRefAttr::get(symbol);
                    old_to_new.insert({old, _new});
                }
            }
        }

        RewritePatternSet nestedToFlat(context);
        nestedToFlat.add<NestedToFlatCallPattern, SymbolReplacerPattern, ZNEReplacerPattern>(
            context, &old_to_new);
        run = _stopAfterStep >= 4 || _stopAfterStep == 0;
        if (run &&
            failed(applyPatternsAndFoldGreedily(symbolTable, std::move(nestedToFlat), config))) {
            signalPassFailure();
        }

        RewritePatternSet cleanup(context);
        cleanup.add<CleanupPattern>(context);
        run = _stopAfterStep >= 5 || _stopAfterStep == 0;
        if (run && failed(applyPatternsAndFoldGreedily(symbolTable, std::move(cleanup), config))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_INLINENESTEDMODULEPASS
#define GEN_PASS_DECL_INLINENESTEDMODULEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct InlineNestedModulePass : impl::InlineNestedModulePassBase<InlineNestedModulePass> {
    using InlineNestedModulePassBase::InlineNestedModulePassBase;

    void runOnOperation() final
    {
        MLIRContext *ctx = &getContext();
        auto op = getOperation();
        auto pm = PassManager::on<ModuleOp>(ctx);
        if (stopAfterStep >= 1 || stopAfterStep == 0) {
            OpPassManager &nestedModulePM = pm.nestAny();
            nestedModulePM.addPass(std::make_unique<AnnotateWithFullyQualifiedNamePass>());
        }

        if (stopAfterStep >= 2 || stopAfterStep == 0) {
            pm.addPass(std::make_unique<InlineNestedSymbolTablePass>(stopAfterStep));
        }

        if (failed(pm.run(op))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createInlineNestedModulePass()
{
    return std::make_unique<InlineNestedModulePass>();
}

} // namespace catalyst
