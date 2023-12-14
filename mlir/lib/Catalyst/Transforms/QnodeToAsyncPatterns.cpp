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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace catalyst {

struct CallOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

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
            // Nothing to change. (For functions which are not qnodes).
            return failure();
        }

        TypeRange retTy;                 /* = empty */
        SmallVector<Value> dependencies; /* = empty */
        SmallVector<Value> operands;     /* = empty */
        auto noopExec = [&](OpBuilder &executeBuilder, Location executeLoc,
                            ValueRange executeArgs) {};
        auto executeOp =
            rewriter.create<async::ExecuteOp>(op.getLoc(), retTy, dependencies, operands, noopExec);
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(executeOp.getBody(), executeOp.getBody()->end());
            rewriter.create<async::YieldOp>(op.getLoc(), ValueRange{});
        }

        auto asyncValues = executeOp.getResults();
        for (auto val : asyncValues) {
            rewriter.create<async::AwaitOp>(op.getLoc(), val);
        }
        rewriter.updateRootInPlace(op, [&] { op->setAttr("transformed", rewriter.getUnitAttr()); });
        return success();
    }
};

void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
