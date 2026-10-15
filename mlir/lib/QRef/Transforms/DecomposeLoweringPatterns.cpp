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

#include <cassert>
#include <cstddef>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefTypes.h"
#include "QRef/Transforms/Patterns.h"

#include "DecompUtils.hpp"
#include "DecomposeLoweringImpl.hpp"

#define DEBUG_TYPE "decompose-lowering"

using namespace mlir;

namespace catalyst {
namespace qref {

/**
 * @brief
 * Inline the body of `rule` at `rewriter`'s current insertion point, using `operands` to
 * replace the parameters of `rule` and returning the results. `rewriter`'s insertion point will be
 * moved to the end of the inlined function body.
 */
void inlineRule(PatternRewriter &rewriter, func::FuncOp rule, ValueRange operands) {
    assert(rule.getBlocks().size() == 1);
    Block &body = rule.front();

    IRMapping mapping;
    mapping.map(body.getArguments(), operands);

    for (Operation &op : body.without_terminator()) {
        rewriter.clone(op, mapping);
    }
}

// Clone a rule function and create a call to the clone
func::CallOp cloneAndCallRule(PatternRewriter &rewriter, func::FuncOp originalRule,
                              ValueRange operands, SymbolTable &moduleSymbolTable,
                              llvm::DenseMap<func::FuncOp, func::FuncOp> &rulesToClonedFuncs) {
    Location loc = originalRule.getLoc();

    // Already cloned before, just create a call
    if (rulesToClonedFuncs.contains(originalRule)) {
        return func::CallOp::create(rewriter, loc, rulesToClonedFuncs[originalRule], operands);
    }

    // First encounter, need to clone and call
    func::FuncOp clonedFunc;
    {
        // Only the function cloning needs to change insertion point:
        // the call to the clone needs to happen at where the gate op originally was
        OpBuilder::InsertionGuard guard(rewriter);
        auto module = cast<ModuleOp>(moduleSymbolTable.getOp());
        rewriter.setInsertionPointToEnd(module.getBody());

        // IR modifications in a rewrite pattern must go through the rewriter, so it can
        // record the changes and rewrite until fixed point
        // Then, insert into the symbol table to resolve naming collisions
        clonedFunc = cast<func::FuncOp>(rewriter.clone(*originalRule));
    }

    // The clones are not rules in the decomp graph: they are just functions to be called
    // by the main circuit, and must not interfere with potential future graph solutions
    clonedFunc.setVisibility(SymbolTable::Visibility::Private);
    clonedFunc->removeAttr("frontend_name");
    clonedFunc->removeAttr("target_gate");
    clonedFunc->removeAttr("resources");
    moduleSymbolTable.insert(clonedFunc);

    rulesToClonedFuncs[originalRule] = clonedFunc;
    return func::CallOp::create(rewriter, loc, clonedFunc, operands);
}

struct DecomposableGatePattern final : public OpInterfaceRewritePattern<DecomposableGate> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    bool inlineRuleBody;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;
    SymbolTable &moduleSymbolTable;
    mutable llvm::DenseMap<func::FuncOp, func::FuncOp> rulesToClonedFuncs;

  public:
    DecomposableGatePattern(MLIRContext *context, const llvm::StringMap<func::FuncOp> &registry,
                            bool inlineRule, const llvm::StringSet<llvm::MallocAllocator> &gateSet,
                            SymbolTable &symbolTable)
        : OpInterfaceRewritePattern<DecomposableGate>(context), decompositionRegistry(registry),
          inlineRuleBody(inlineRule), targetGateSet(gateSet), moduleSymbolTable(symbolTable) {};

    LogicalResult matchAndRewrite(DecomposableGate op, PatternRewriter &rewriter) const override {
        std::string gateName = op.getOperatorName();

        // A modified op (adjoint and/or controlled) is a distinct operator from its base gate.
        bool isModified =
            op.getOperation()->hasAttr("adjoint") || !op.getCtrlQubitOperands().empty();

        // Only decompose the op if it is not in the target gate set. A modified op is never treated
        // as a native gate-set member by its base name: `Adjoint(Op)`/`C(Op)` are distinct gates
        // that must reach the gate_set through their own rules.
        if (!isModified && targetGateSet.contains(gateName)) {
            return failure();
        }

        // do not nest decomposition rules, they're applied greedily and this can lead to
        // cycles/identity rules
        if (DecompUtils::isInDecompRule(op)) {
            return failure();
        }

        // Find the corresponding decomposition rule for the op
        // TODO: migration to use the DecomposableGate interface gateID for all decomp rules is not
        // yet complete. Some rules' target_gate is already gateID, but some other gates' are still
        // the simple gate class name.
        // To maintain legacy compatibility, we fallback to the old pattern, where only the
        // gate class name is used (i.e. without distinguishing different static data on the same
        // gate class)
        // When the migration is complete, all rules need to be identified through gate ID.
        std::string gateID = op.getGraphOpId();
        func::FuncOp rule;
        auto it_gateID = decompositionRegistry.find(gateID);
        if (it_gateID != decompositionRegistry.end()) {
            // Found a rule with the wanted ID, highest priority rule, just use this one
            rule = it_gateID->second;
        } else if (!isModified) {
            // Didn't find ID match, try matching gate name. Unmodified ops only: a modified op
            // (`Adjoint(Op)`/`C(Op)`) must match its own rule by id and never fall back to a plain
            // base-name rule (that would apply the unmodified decomposition to the modified op).
            // TODO: remove multirz's special name editing
            if (isa<qref::MultiRZOp>(op)) {
                gateName = gateName + "_" + std::to_string(op.getWireLens()["wires"]);
            }
            auto it_gateName = decompositionRegistry.find(gateName);
            if (it_gateName != decompositionRegistry.end()) {
                rule = it_gateName->second;
            } else {
                // Didn't find any rule
                return failure();
            }
        } else {
            // Modified op with no id-matched rule: do not fall back to the base-name rule.
            return failure();
        }

        // For null decomp rules, the signature will not have any quantum values
        // This is a deviation from the standard decomp func signature, so we deal with it
        // separately
        if (!llvm::any_of(
                llvm::concat<const Type>(rule.getFunctionType().getInputs(),
                                         rule.getFunctionType().getResults()),
                [](const mlir::Type t) { return isa<qref::QuregType, qref::QubitType>(t); })) {
            rewriter.eraseOp(op);
            return success();
        }

        // Here is the assumption that the decomposition rule must have at least one input
        assert(rule.getFunctionType().getNumInputs() > 0 &&
               "Decomposition function must have at least one input");

        rewriter.setInsertionPointAfter(op);

        auto enableQreg = llvm::any_of(rule.getFunctionType().getInputs(),
                                       [](mlir::Type t) { return isa<qref::QuregType>(t); });
        auto analyzer = DecomposableGateSignatureAnalyzer(op, enableQreg);
        if (!analyzer) {
            return failure();
        }

        auto operands = analyzer.prepareOperands(rule, rewriter, op.getLoc());
        // prepareOperands flags the analyzer invalid (and emits a diagnostic) when the rule's
        // signature cannot be reconciled with the operator, rather than building a malformed op.
        if (!analyzer) {
            return failure();
        }

        if (inlineRuleBody) {
            inlineRule(rewriter, rule, operands);
            rewriter.eraseOp(op);
        } else {
            func::CallOp callOp =
                cloneAndCallRule(rewriter, rule, operands, moduleSymbolTable, rulesToClonedFuncs);

            // DL is in reference semantics, so only classical results will be returned from the
            // rules
            rewriter.replaceOp(op, callOp);
        }
        return success();
    }
};

void populateDecomposeLoweringPatterns(RewritePatternSet &patterns,
                                       const llvm::StringMap<func::FuncOp> &decompositionRegistry,
                                       bool inlineRuleBody,
                                       const llvm::StringSet<llvm::MallocAllocator> &targetGateSet,
                                       SymbolTable &moduleSymbolTable) {
    patterns.add<DecomposableGatePattern>(patterns.getContext(), decompositionRegistry,
                                          inlineRuleBody, targetGateSet, moduleSymbolTable);
}

} // namespace qref
} // namespace catalyst
