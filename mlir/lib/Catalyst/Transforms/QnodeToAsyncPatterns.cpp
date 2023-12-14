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
        async::FuncOp asyncFuncOp =
            dyn_cast_or_null<async::FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, symbol));
        if (!asyncFuncOp) {
            // Nothing to change.
            return failure();
        }

        // Check for Call ops that have QNode func ops
        if (!asyncFuncOp->hasAttrOfType<UnitAttr>("qnode")) {
            // Nothing to change. (For functions which are not qnodes).
            return failure();
        }

        TypeConverter conv;
        conv.addConversion([](RankedTensorType type) -> Type {
            return MemRefType::get(type.getShape(), type.getElementType());
        });
        auto callOp = rewriter.create<async::CallOp>(op.getLoc(), asyncFuncOp, op.getArgOperands());
        auto newResults = callOp.getResults();
        auto oldResults = op.getResults();
        for (auto [newResult, oldResult] : llvm::zip(newResults, oldResults)) {
            auto awaitOp = rewriter.create<async::AwaitOp>(op.getLoc(), newResult);
            auto memrefAsync = awaitOp.getResult();
            auto memrefType = oldResult.getType().cast<MemRefType>();
            auto memrefHeap = rewriter.create<memref::AllocOp>(op.getLoc(), memrefType);
            rewriter.create<memref::CopyOp>(op.getLoc(), memrefAsync, memrefHeap);
            rewriter.create<async::RuntimeDropRefOp>(op.getLoc(), newResult,
                                                     rewriter.getI64IntegerAttr(1));
            oldResult.replaceAllUsesWith(memrefHeap.getResult());
        }

        rewriter.replaceOp(op, callOp);
        return success();
    }
};

void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
