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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

const char *transformedAttr = "catalyst.transformed";

bool hasQnodeAttribute(LLVM::LLVMFuncOp callOp)
{
    return (bool)(callOp->getAttrOfType<UnitAttr>("qnode"));
}

bool hasTransformedAttribute(LLVM::CallOp callOp)
{
    return (bool)(callOp->getAttrOfType<UnitAttr>(transformedAttr));
}

LLVM::LLVMFuncOp getCaller(LLVM::CallOp callOp)
{
    mlir::Operation *currentOperation = callOp;
    do {
        currentOperation = currentOperation->getParentOp();
    } while (!isa<LLVM::LLVMFuncOp>(currentOperation));
    return cast<LLVM::LLVMFuncOp>(currentOperation);
}

std::optional<LLVM::LLVMFuncOp> getCalleeSafe(LLVM::CallOp callOp)
{
    std::optional<LLVM::LLVMFuncOp> callee;
    auto calleeAttr = callOp.getCalleeAttr();
    auto caller = getCaller(callOp);
    if (!calleeAttr) {
        callee = std::nullopt;
    }
    else {
        callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(caller, calleeAttr);
    }
    return callee;
}

LLVM::LLVMFuncOp getCalleeUnsafe(LLVM::CallOp callOp)
{
    std::optional<LLVM::LLVMFuncOp> optionalCallee = getCalleeSafe(callOp);
    if (!optionalCallee) {
        callOp->emitError() << "Couldn't resolve callee for call.";
    }
    return optionalCallee.value();
}

void setTransformedAttribute(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(callOp,
                               [&] { callOp->setAttr(transformedAttr, rewriter.getUnitAttr()); });
}

struct DetectQnodeTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult DetectQnodeTransform::match(LLVM::CallOp callOp) const
{
    // Only match with direct calls to qnodes.
    // TODO: This should actually be about async.
    // Right now we use the `qnode` attribute to determine async regions.
    // But that might not be the case in the future,
    // So, change this whenever we no longer create async.execute operations based on qnode.
    std::optional<LLVM::LLVMFuncOp> candidate = getCalleeSafe(callOp);
    bool validCandidate =
        candidate && hasQnodeAttribute(candidate.value()) && !hasTransformedAttribute(callOp);
    return validCandidate ? success() : failure();
}

void DetectQnodeTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    setTransformedAttribute(callOp, rewriter);
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_DETECTQNODEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetectQnodePass : impl::DetectQnodePassBase<DetectQnodePass> {
    using DetectQnodePassBase::DetectQnodePassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<DetectQnodeTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDetectQnodePass() { return std::make_unique<DetectQnodePass>(); }

} // namespace catalyst
