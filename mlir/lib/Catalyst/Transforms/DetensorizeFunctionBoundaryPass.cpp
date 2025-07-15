#define DEBUG_TYPE "myhelloworld"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace {
bool isScalarTensor(Value v)
{
    Type type = v.getType();
    if (!isa<TensorType>(type)) {
        return false;
    }
    return cast<TensorType>(type).getRank() == 0;
};

template <typename## #Op> bool hasScalarTensorOperand(## #Op op)
{
    for (Value result : op->getOperands()) {
        if (isScalarTensor(result)) {
            return true;
        }
    }
    return false;
}

template <typename## #Op> bool hasScalarTensorResult(## #Op op)
{
    for (Value result : op->getResults()) {
        if (isScalarTensor(result)) {
            return true;
        }
    }
    return false;
}

struct DetensorizeForOp : public OpRewritePattern<##> {
    using OpRewritePattern<##>::OpRewritePattern;
};

} // namespace

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZEFUNCTIONBOUNDARYPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeFounctionBoundaryPass
    : public impl::DetensorizeFounctionBoundaryPassBase<DetensorizeFounctionBoundaryPass> {
    using impl::DetensorizeFounctionBoundaryPassBase<
        DetensorizeFounctionBoundaryPass>::DetensorizeFounctionBoundaryPassBase;
    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<DetensorizeForOp>(context);
        patterns.add<DetensorizeIfOp>(context);
        patterns.add<DetensorizeWhileOp>(context);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> createDetensorizeFounctionBoundaryPass()
{
    return std::make_unique<DetensorizeFounctionBoundaryPass>();
}

} // namespace catalyst
