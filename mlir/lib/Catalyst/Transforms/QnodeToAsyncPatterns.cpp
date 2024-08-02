// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "scatter"

#include <algorithm>
#include <vector>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace catalyst {

struct CallOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

    void insertDropRefOps(func::CallOp op, async::ExecuteOp executeOp,
                          PatternRewriter &rewriter) const
    {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        // Let's first place the awaits just before the terminator of this block.
        // If there is no terminator in this block, then we will place the await
        // at the end of the block.
        Block *block = op->getBlock();
        // TODO: Once we update LLVM versions, we might have access to function
        // `mightHaveTerminator` For now, we have ''inlined'' the definition here.
        bool mightHaveTerminator =
            !block->empty() && block->back().mightHaveTrait<OpTrait::IsTerminator>();
        if (mightHaveTerminator) {
            rewriter.setInsertionPoint(block->getTerminator());
        }
        else {
            rewriter.setInsertionPointToEnd(block);
        }

        // Here we place an await until the very end.
        // The reason for this is because if the call returns no values,
        // then we will never execute it.
        // TODO: We could just remove the call altogether.
        // But we should make sure that there are no side effects in the call.
        // One possible side effect is printing to stdout.
        rewriter.create<async::AwaitOp>(op.getLoc(), executeOp.getResults().front());

        for (auto refCountedValue : executeOp.getResults()) {
            rewriter.create<async::RuntimeDropRefOp>(op.getLoc(), refCountedValue,
                                                     rewriter.getI64IntegerAttr(1));
        }
    }

    void insertAwaitOps(func::CallOp op, async::ExecuteOp executeOp,
                        PatternRewriter &rewriter) const
    {
        // If there are no results for the call, just return
        if (op.getResults().size() == 0) {
            return;
        }
        // The async.execute instruction returns a promise.
        //
        //    %token, %promise = async.execute {
        //       // body...
        //       async.yield %0 : !some.type
        //    }
        //
        //  Then we would wait on the promise or the token like so:
        //
        //     async.await %token
        //     async.await %promise
        //
        //  If called async for the first time and the value is not available, the thread will wait
        //  until the value becomes available. If the value is available, the function returns
        //  practically immediately. For example.
        //
        //     %1 = async.await %promise // promise is not yet available, will wait.
        //     %2 = async.await %promise // promise is now available, will not wait.
        //
        //  This makes the job of deciding where to place awaits really easy.
        //  We just need to place awaits just before every use of the original value.
        //  For example, take the following piece of code
        //
        //     %quantum_results = call @qnode
        //
        //     // Arbitrary control flow
        //     // This use might be inside a different basic block.
        //     %some_val = some.op %quantum_results
        //
        //  First we translate the call to an async.execute (which was performed before this
        //  function was called) When this function was called, the transformation is not yet
        //  complete. Because there are still uses of %quantum_results.
        //
        //     %async_quantum_results = call @qnode
        //
        //     // Arbitrary control flow
        //     // This use might be inside a different basic block.
        //     %some_val = some.op %quantum_results
        //
        //  This function will perform the following change:
        //
        //     %async_quantum_results = call @qnode
        //
        //     // Arbitrary control flow
        //     // This use might be inside a different basic block.
        //     %await_quantum_results = async.await %async_quantum_results
        //     %some_val = some.op %await_quantum_results
        //
        //  And it will do so for every use of %quantum_results
        //  E.g.
        //
        //     %async_quantum_results = call @qnode
        //
        //     // Arbitrary control flow
        //     // This use might be inside a different basic block.
        //     %await_quantum_results_0 = async.await %async_quantum_results
        //     %some_val_0 = some.op %await_quantum_results_0
        //     %await_quantum_results_1 = async.await %async_quantum_results_1
        //     %some_val_1 = some.op %await_quantum_results_1
        //
        //  However, we can do a bit better and avoid this extra awaits that are
        //  dominated by the first await. Keeping the awaits is not incur a large increase
        //  in run time. Removing the awaits involves using the dominator analysis, which we need
        //  some time to investigate how to use. Not much, but sufficient enough for a future
        //  improvement. See TODO inside replaceUsesWithIf.
        auto results = executeOp.getResults();
        // The first value is just a token.
        std::vector<Value> bodyReturns(results.begin() + 1, results.end());

        // It is guaranteed that op.getResults().size() and bodyReturns.size() are equal.
        for (auto &&[oldVal, newVal] : llvm::zip(op.getResults(), bodyReturns)) {
            auto _users = oldVal.getUsers();
            // Insert users into a vector to avoid modifying users during a loop.
            std::vector<Operation *> users(_users.begin(), _users.end());
            for (auto user : users) {
                // Now we can safely modify users
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(user);
                auto awaitOp = rewriter.create<async::AwaitOp>(op.getLoc(), newVal);
                auto awaitVal = awaitOp.getResults();
                rewriter.replaceUsesWithIf(oldVal, awaitVal, [&](OpOperand &use) {
                    // TODO:
                    // Change the line below to use.getOwner is strictly dominated by user.
                    // For introductory explanation on dominators see here:
                    //
                    //    https://en.wikipedia.org/wiki/Dominator_(graph_theory)
                    return use.getOwner() == user;
                });
            }
        }
    }

    mlir::LogicalResult matchAndRewrite(func::CallOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        SymbolRefAttr symbol = dyn_cast_if_present<SymbolRefAttr>(op.getCallableForCallee());
        func::FuncOp func = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, symbol);
        if (!func) {
            // Nothing to change.
            return failure();
        }

        // Check for Call ops that have QNode func ops
        if (!func->hasAttrOfType<UnitAttr>("qnode")) {
            // Nothing to change. (For functions which are not qnodes).
            return failure();
        }

        if (op->hasAttrOfType<UnitAttr>("transformed")) {
            // Nothing to change. (For func calls that have already been transformed by this pass).
            return failure();
        }

        TypeRange retTy = op.getResultTypes();
        SmallVector<Value> dependencies; /* = empty */
        SmallVector<Value> operands;     /* = empty */
        auto noopExec = [&](OpBuilder &executeBuilder, Location executeLoc,
                            ValueRange executeArgs) {};

        rewriter.modifyOpInPlace(op, [&] { op->setAttr("transformed", rewriter.getUnitAttr()); });
        IRMapping map;
        auto executeOp =
            rewriter.create<async::ExecuteOp>(op.getLoc(), retTy, dependencies, operands, noopExec);
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(executeOp.getBody(), executeOp.getBody()->end());
            Operation *cloneOp = op->clone(map);
            rewriter.insert(cloneOp);
            rewriter.create<async::YieldOp>(op.getLoc(), cloneOp->getResults());
        }

        insertDropRefOps(op, executeOp, rewriter);
        insertAwaitOps(op, executeOp, rewriter);

        rewriter.eraseOp(op);

        return success();
    }
};

void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
