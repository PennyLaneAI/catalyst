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

template <typename OpType> using OperationToQubits = std::map<OpType, std::vector<mlir::Value>>;

template <typename OpType>
void traceOperandQubit(OpType &op, std::map<OpType, std::vector<mlir::Value>> &opToQubits)
{
    if (opToQubits.find(op) != opToQubits.end()) {
        return;
    }

    mlir::Value rootInQubit;

    for (auto qubit : op.getInQubits()) {
        rootInQubit = qubit;
        while (auto definingOp = dyn_cast_or_null<OpType>(rootInQubit.getDefiningOp())) {
            for (auto [idx, outQubit] : llvm::enumerate(definingOp.getOutQubits())) {
                if (outQubit == rootInQubit) {
                    rootInQubit = definingOp.getInQubits()[idx];
                }
            }
        }
        opToQubits[op].push_back(rootInQubit);
    }
}

template <typename OpType> 
OperationToQubits<OpType> traceTopEdgeOperations(scf::ForOp forOp)
{
    OperationToQubits<OpType> opToQubits;
    for (auto arg : forOp.getRegionIterArgs()) {
        for (mlir::Operation *user : arg.getUsers()) {
            if (auto operation = dyn_cast_or_null<OpType>(user)) {
                traceOperandQubit(operation, opToQubits);
            }
        }
    }
    return opToQubits;
}

template <typename OpType> 
OperationToQubits<OpType> traceBottomEdgeOperations(scf::ForOp forOp)
{
    OperationToQubits<OpType> opToQubits;
    Operation *terminator = forOp.getBody()->getTerminator();
    for (auto operand : terminator->getOperands()) {

        // check if operation is not quantum
        auto operation = dyn_cast_or_null<OpType>(operand.getDefiningOp());
        if (!operation) {
            continue;
        }
        traceOperandQubit(operation, opToQubits);
    }

    return opToQubits;
}

template <typename OpType>
std::vector<std::pair<OpType, OpType>>
getVerifyEdgeOperationSet(OperationToQubits<OpType> bottomEdgeOperationSet,
                          OperationToQubits<OpType> topEdgeOperationSet)
{

    std::vector<std::pair<OpType, OpType>> edgeOperationSet;

    for (auto [bottomOp, bottomQubits] : bottomEdgeOperationSet) {
        for (auto [topOp, topQubits] : topEdgeOperationSet) {
            // convert const BottomOp to non-const
            OpType bottomOpNonConst = bottomOp;
            OpType topOpNonConst = topOp;

            // check if the operation types are the same
            if (bottomOpNonConst.getGateName() != topOpNonConst.getGateName()) {
                continue;
            }

            // operations should not be the same
            if (topOp == bottomOp) {
                continue;
            }

            // check if the qubits are the same
            bool verifyEdgeOperationSet = true;
            for (auto [idx, rootTopEdgeQubit] : llvm::enumerate(topQubits)) {
                if (rootTopEdgeQubit != bottomQubits[idx]) {
                    verifyEdgeOperationSet = false;
                    break;
                }
            }

            // check if the predecessor of the top op is not quantumCustom
            for (auto operand : topOpNonConst.getInQubits()) {
                if (operand.getDefiningOp()) {
                    verifyEdgeOperationSet = false;
                    break;
                }
            }

            // check if the successor of the bottom op is not quantumCustom
            // TODO: verify if it is not any quantum operation
            for (auto op : bottomOpNonConst->getUsers()) {
                if (isa<CustomOp>(op)) {
                    verifyEdgeOperationSet = false;
                    break;
                }
            }

            // check if the operation is a rotation
            if (!verifyEdgeOperationSet) {
                continue;
            }

            // check if the operation has previous operations
            edgeOperationSet.push_back(std::make_pair(topOp, bottomOp));
        }
    }
    return edgeOperationSet;
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

        auto topEdgeOperationSet = traceTopEdgeOperations<CustomOp>(forOp);

        auto bottomEdgeOperationSet = traceBottomEdgeOperations<CustomOp>(forOp);

        auto edgeOperationSet =
            getVerifyEdgeOperationSet<CustomOp>(bottomEdgeOperationSet, topEdgeOperationSet);

        for (auto [topEdgeOp, bottomEdgeOp] : edgeOperationSet) {

            auto topEdgeParams = topEdgeOp.getParams();
            auto bottomEdgeParams = bottomEdgeOp.getParams();

            // Hoist the top edge operation to the top of the loop
            rewriter.moveOpBefore(topEdgeOp, forOp);
            topEdgeOp.getOutQubits().replaceAllUsesWith(
                topEdgeOp.getInQubits()); // config successor

            for (auto [arg, regionArg] :
                 llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
                for (auto [idx, qubit] : llvm::enumerate(topEdgeOp.getInQubits())) {
                    if (qubit == regionArg) {
                        arg.replaceAllUsesWith(topEdgeOp.getOutQubits()[idx]);
                        unsigned qubitIndx = topEdgeParams.size() + idx;
                        topEdgeOp.setOperand(qubitIndx, arg);
                    }
                }
            }

            if (topEdgeParams.size() > 0) {
                auto cloneTopOp = topEdgeOp.clone();

                for (auto [idx, param] : llvm::enumerate(topEdgeParams)) {
                    cloneTopOp.setOperand(idx, param);
                }
                cloneTopOp.setQubitOperands(bottomEdgeOp.getInQubits());

                rewriter.setInsertionPointAfter(bottomEdgeOp);
                rewriter.insert(cloneTopOp);

                auto cloneBottomOp = bottomEdgeOp.clone();

                for (auto [idx, param] : llvm::enumerate(bottomEdgeParams)) {
                    cloneBottomOp.setOperand(idx, param);
                }
                cloneBottomOp.setQubitOperands(cloneTopOp.getOutQubits());
                bottomEdgeOp.setQubitOperands(cloneBottomOp.getOutQubits());

                rewriter.setInsertionPointAfter(cloneTopOp);
                rewriter.insert(cloneBottomOp);

                // change the param of topEdgeOp to negative value
                for (auto [idx, param] : llvm::enumerate(topEdgeParams)) {
                    mlir::Value negParam =
                        rewriter.create<arith::NegFOp>(bottomEdgeOp.getLoc(), param).getResult();
                    bottomEdgeOp.setOperand(idx, negParam);
                }
            }

            // Hoist the bottom edge operation to the bottom of the loop
            bottomEdgeOp.getOutQubits().replaceAllUsesWith(
                bottomEdgeOp.getInQubits()); // config the successor
            for (auto [arg, regionArg] : llvm::zip(forOp.getResults(), forOp.getRegionIterArgs())) {
                for (auto [idx, qubit] : llvm::enumerate(bottomEdgeOperationSet[bottomEdgeOp])) {
                    if (qubit == regionArg) {
                        arg.replaceAllUsesWith(bottomEdgeOp.getOutQubits()[idx]);
                        unsigned qubitIndx = topEdgeParams.size() + idx;
                        bottomEdgeOp.setOperand(qubitIndx, arg);
                    }
                }
            }
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
