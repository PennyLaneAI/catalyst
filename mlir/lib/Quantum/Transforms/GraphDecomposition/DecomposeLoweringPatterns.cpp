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
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"

#include "DecompUtils.hpp"
#include "DecomposeLoweringImpl.hpp"

#include <llvm/Support/LogicalResult.h>

#define DEBUG_TYPE "decompose-lowering"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

/**
 * @brief
 * Inline the body of `rule` at `rewriter`'s current insertion point, using `operands` to
 * replace the parameters of `rule` and returning the results. `rewriter`'s insertion point will be
 * moved to the end of the inlined function body.
 */
static SmallVector<Value> inlineRuleBody(PatternRewriter &rewriter, func::FuncOp rule,
                                         ValueRange operands)
{
    assert(rule.getBlocks().size() == 1);
    Block &body = rule.front();
    auto returnOp = cast<func::ReturnOp>(body.getTerminator());

    IRMapping mapping;
    mapping.map(body.getArguments(), operands);

    for (Operation &op : body.without_terminator()) {
        rewriter.clone(op, mapping);
    }

    SmallVector<Value> results;
    for (Value operand : returnOp.getOperands()) {
        results.push_back(mapping.lookupOrDefault(operand));
    }
    return results;
}

struct DecomposableGatePattern final : public OpInterfaceRewritePattern<DecomposableGate> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;

  public:
    DecomposableGatePattern(MLIRContext *context, const llvm::StringMap<func::FuncOp> &registry,
                            const llvm::StringSet<llvm::MallocAllocator> &gateSet)
        : OpInterfaceRewritePattern<DecomposableGate>(context), decompositionRegistry(registry),
          targetGateSet(gateSet)
    {
    }

    LogicalResult matchAndRewrite(DecomposableGate op, PatternRewriter &rewriter) const override
    {
        std::string gateName = op.getOperatorName();
        llvm::errs() << "visiting " << gateName << "\n";

        // Only decompose the op if it is not in the target gate set
        if (targetGateSet.contains(gateName)) {
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
        }
        else {
            // Didn't find ID match, try matching gate name
            auto it_gateName = decompositionRegistry.find(gateName);
            if (it_gateName != decompositionRegistry.end()) {
                rule = it_gateName->second;
            }
            else {
                // Didn't find any rule
                return failure();
            }
        }

        // For null decomp rules, the signature will not have any quantum values
        // This is a deviation from the standard decomp func signature, so we deal with it
        // separately
        if (!llvm::any_of(llvm::concat<const Type>(rule.getFunctionType().getInputs(),
                                                   rule.getFunctionType().getResults()),
                          [](const mlir::Type t) {
                              return isa<quantum::QuregType, quantum::QubitType>(t);
                          })) {
            for (auto [inQubit, outQubit] :
                 llvm::zip_equal(op.getQubitOperands(), op.getQubitResults())) {
                rewriter.replaceAllUsesWith(outQubit, inQubit);
            }
            return success();
        }

        // Here is the assumption that the decomposition rule must have at least one input and
        // one result
        assert(rule.getFunctionType().getNumInputs() > 0 &&
               "Decomposition function must have at least one input");
        assert(rule.getFunctionType().getNumResults() >= 1 &&
               "Decomposition function must have at least one result");

        rewriter.setInsertionPointAfter(op);

        auto enableQreg = llvm::any_of(rule.getFunctionType().getInputs(),
                                       [](mlir::Type t) { return isa<quantum::QuregType>(t); });
        auto analyzer = DecomposableGateSignatureAnalyzer(op, enableQreg);
        assert(analyzer && "Analyzer should be valid");

        auto operands = analyzer.prepareOperands(rule, rewriter, op.getLoc());
        SmallVector<Value> inlinedFunctionResults = inlineRuleBody(rewriter, rule, operands);

        // Replace the op with the inlined function and adjust the insert ops for the qreg mode
        if (inlinedFunctionResults.size() == 1 &&
            isa<quantum::QuregType>(inlinedFunctionResults.front().getType())) {
            auto results = analyzer.prepareResultsForQreg(inlinedFunctionResults.front(),
                                                          op.getLoc(), rewriter);
            rewriter.replaceOp(op, results);
        }
        else {
            rewriter.replaceOp(op, inlinedFunctionResults);
        }

        return success();
    }
};

void populateDecomposeLoweringPatterns(RewritePatternSet &patterns,
                                       const llvm::StringMap<func::FuncOp> &decompositionRegistry,
                                       const llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
{
    patterns.add<DecomposableGatePattern>(patterns.getContext(), decompositionRegistry,
                                          targetGateSet);
}

} // namespace quantum
} // namespace catalyst
