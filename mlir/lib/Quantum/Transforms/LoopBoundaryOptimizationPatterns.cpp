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

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "VerifyParentGateAnalysis.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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

        quantum::CustomOp firstGateOp = opMap.begin()->first;
        quantum::CustomOp secondGateOp = opMap.rbegin()->first;

        if (opMap.begin()->first == opMap.rbegin()->first) {
            return mlir::failure();
        }

        if (opMap[firstGateOp][0] == opMap[secondGateOp][0] &&
            firstGateOp.getGateName() == secondGateOp.getGateName()) {

            // create new top-edge gate
            auto firstOp = firstGateOp.clone();
            firstOp->setOperands(forOp.getInitArgs());
            rewriter.setInsertionPoint(forOp);
            rewriter.insert(firstOp);

            // config the operand of for-loop to be the result of the first gate
            forOp.setOperand(3, firstOp.getResult(0));

            // config the successor of the first gate to be the for-loop
            firstGateOp.getOutQubits().replaceAllUsesWith(firstGateOp.getInQubits());

            // erase the first gate
            rewriter.eraseOp(firstGateOp);

            // create new bottom-edge gate
            auto secondOp = secondGateOp.clone();

            // replace successor of for-loop with the second gate
            forOp.getResults().replaceAllUsesWith(secondOp);

            // set the operands of the 
            secondOp->setOperands(forOp.getResults());
            rewriter.setInsertionPointAfter(forOp);
            rewriter.insert(secondOp);

            // config the successor of the second gate to be the successor of the for-loop
            secondGateOp.getOutQubits().replaceAllUsesWith(secondGateOp.getInQubits());
            rewriter.eraseOp(secondGateOp);
            
            return mlir::success();
        }

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