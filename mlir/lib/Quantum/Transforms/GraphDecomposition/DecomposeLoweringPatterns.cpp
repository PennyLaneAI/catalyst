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
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "Quantum/Transforms/Patterns.h"

#include "DecompUtils.hpp"
#include "DecomposeLoweringImpl.hpp"

#define DEBUG_TYPE "decompose-lowering"

using namespace mlir;

namespace catalyst::quantum {

static void inlineRuleBody(PatternRewriter &rewriter, func::FuncOp rule, ValueRange operands) {
    assert(rule.getBlocks().size() == 1 && "decomposition rules must contain one block");
    Block &body = rule.front();
    IRMapping mapping;
    mapping.map(body.getArguments(), operands);
    for (Operation &operation : body.without_terminator()) {
        rewriter.clone(operation, mapping);
    }
}

struct DecomposableGatePattern final
    : public OpInterfaceRewritePattern<catalyst::qref::DecomposableGate> {
    DecomposableGatePattern(MLIRContext *context,
                            const llvm::StringMap<func::FuncOp> &decompositionRegistry,
                            const llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
        : OpInterfaceRewritePattern<catalyst::qref::DecomposableGate>(context),
          decompositionRegistry(decompositionRegistry), targetGateSet(targetGateSet) {}

    LogicalResult matchAndRewrite(catalyst::qref::DecomposableGate gate,
                                  PatternRewriter &rewriter) const override {
        if (DecompUtils::isInDecompRule(gate)) {
            return failure();
        }

        std::string gateName = gate.getOperatorName();
        const bool isModified = gate.getAdjointFlag() || !gate.getCtrlQubitOperands().empty();
        if (!isModified && targetGateSet.contains(gateName)) {
            return failure();
        }

        func::FuncOp rule;
        if (auto iter = decompositionRegistry.find(gate.getGraphOpId());
            iter != decompositionRegistry.end()) {
            rule = iter->second;
        } else if (!isModified) {
            if (auto multiRZ = dyn_cast<catalyst::qref::MultiRZOp>(gate.getOperation())) {
                gateName += "_" + std::to_string(multiRZ.getQubits().size());
            }
            if (auto iter = decompositionRegistry.find(gateName);
                iter != decompositionRegistry.end()) {
                rule = iter->second;
            }
        }
        if (!rule) {
            return failure();
        }

        // Reference-semantics rules have no quantum results. A rule with no quantum arguments is
        // the null decomposition, so deleting the target gate is the complete rewrite.
        bool hasQuantumArgument = llvm::any_of(rule.getArgumentTypes(), [](Type type) {
            return isa<catalyst::qref::QubitType, catalyst::qref::QuregType>(type);
        });
        if (!hasQuantumArgument) {
            rewriter.eraseOp(gate);
            return success();
        }

        rewriter.setInsertionPoint(gate);
        ReferenceSignatureAnalyzer analyzer(gate);
        auto operands = analyzer.prepareOperands(rule, rewriter);
        if (failed(operands)) {
            return failure();
        }

        inlineRuleBody(rewriter, rule, *operands);
        rewriter.eraseOp(gate);
        return success();
    }

  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;
};

void populateDecomposeLoweringPatterns(
    RewritePatternSet &patterns, const llvm::StringMap<func::FuncOp> &decompositionRegistry,
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet) {
    patterns.add<DecomposableGatePattern>(patterns.getContext(), decompositionRegistry,
                                          targetGateSet);
}

} // namespace catalyst::quantum
