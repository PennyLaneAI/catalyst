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

#define DEBUG_TYPE "loop-boundary"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "VerifyParentGateAnalysis.hpp"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

static const mlir::StringSet<> hamiltonianSet = {"H", "X", "Y", "Z"};

namespace {

// TODO: Reduce the complexity of the function
// TODO: Support multi-qubit gates
template <typename OpType>
std::map<OpType, std::vector<mlir::Value>> traceOperationQubit(mlir::Block *block)
{
    std::map<OpType, std::vector<mlir::Value>> opMap;
    block->walk([&](OpType op) {
        mlir::Value operand = op.getInQubits()[0];

        while (auto definingOp = dyn_cast_or_null<CustomOp>(operand.getDefiningOp())) {
            operand = definingOp.getInQubits()[0];
        }

        opMap[op].push_back(operand);
    });
    return opMap;
}

bool verifyEdgeQubitOperands(std::map<quantum::CustomOp, std::vector<mlir::Value>> opMap)
{
    quantum::CustomOp topEdgeOp = opMap.begin()->first;
    quantum::CustomOp bottomEdgeOp = opMap.rbegin()->first;

    if (topEdgeOp == bottomEdgeOp) {
        return false;
    }

    if (topEdgeOp.getGateName() != bottomEdgeOp.getGateName()) {
        return false;
    }

    for (auto [idx, rootTopEdgeQubit] : llvm::enumerate(opMap[topEdgeOp])) {
        if (rootTopEdgeQubit != opMap[bottomEdgeOp][idx]) {
            return false;
        }
    }

    return true;
}

struct LoopBoundaryForLoopRewritePattern : public mlir::OpRewritePattern<scf::ForOp> {
    using mlir::OpRewritePattern<scf::ForOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(scf::ForOp forOp,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << forOp << "\n");

        // if number of operations in the loop is less than 2, return
        if (forOp.getBody()->getOperations().size() < 2) {
            return mlir::failure();
        }

        auto opMap = traceOperationQubit<quantum::CustomOp>(forOp.getBody());

        quantum::CustomOp topEdgeOp = opMap.begin()->first;
        quantum::CustomOp bottomEdgeOp = opMap.rbegin()->first;

        if (verifyEdgeQubitOperands(opMap)) {

            ValueRange parentTopInQubits = topEdgeOp.getInQubits();

            // iter_args(%q_0 = %q)
            // - %q_0 -> getRegionIterArg(0)
            // - %q -> getInitArgs()[0]

            rewriter.moveOpBefore(topEdgeOp, forOp);
            topEdgeOp.getOutQubits().replaceAllUsesWith(
                topEdgeOp.getInQubits()); // config successor

            for (auto [idx, arg] : llvm::enumerate(forOp.getInitArgs())) {
                for (auto regionArg : forOp.getRegionIterArgs()) {
                    if (regionArg == topEdgeOp.getInQubits()[idx]) {
                        arg.replaceAllUsesWith(topEdgeOp.getOutQubits()[idx]);
                        topEdgeOp.setOperand(idx, arg);
                    }
                }
            }

            bottomEdgeOp.getOutQubits().replaceAllUsesWith(
                bottomEdgeOp.getInQubits()); // config the successor
            forOp.getResults().replaceAllUsesWith(bottomEdgeOp);
            bottomEdgeOp.setQubitOperands(forOp.getResults()); // config predecessor
            rewriter.moveOpAfter(bottomEdgeOp, forOp);

            return mlir::success();
        }

        return mlir::failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateLoopBoundaryPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<LoopBoundaryForLoopRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
