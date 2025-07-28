#define DEBUG_TYPE "myhelloworld"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

struct DetensorizeExtractPattern : public OpConversionPattern<tensor::ExtractOp> {
    using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getTensor());
        return success();
    }
};

struct DetensorizeFromElementsPattern : public OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        assert(adaptor.getElements().size() == 1);
        rewriter.replaceOp(op, adaptor.getElements()[0]);
        return success();
    }
};

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZEFUNCTIONBOUNDARYPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeFunctionBoundaryPass
    : public impl::DetensorizeFunctionBoundaryPassBase<DetensorizeFunctionBoundaryPass> {
    using impl::DetensorizeFunctionBoundaryPassBase<
        DetensorizeFunctionBoundaryPass>::DetensorizeFunctionBoundaryPassBase;
    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);

        TypeConverter typeConverter;

        typeConverter.addConversion([&](Type type) -> std::optional<Type> {
            if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
                // Convert 0-D tensors to their underlying element type.
                if (rankedType.getRank() == 0) {
                    return rankedType.getElementType();
                }
            }
            // Keep all other types as they are.
            return type;
        });

        typeConverter.addArgumentMaterialization([](OpBuilder &builder, RankedTensorType resultType,
                                                    ValueRange inputs, Location loc) -> Value {
            if (resultType.getRank() == 0) {
                return builder.create<tensor::FromElementsOp>(loc, resultType, inputs).getResult();
            }
            return Value();
        });

        typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                                  ValueRange inputs, Location loc) -> Value {
            if (inputs.size() == 1) {
                if (auto rankedType = dyn_cast<RankedTensorType>(inputs[0].getType())) {
                    if (rankedType.getRank() == 0 && rankedType.getElementType() == resultType) {
                        return builder.create<tensor::ExtractOp>(loc, inputs[0], ValueRange{})
                            .getResult();
                    }
                }
            }
            return Value();
        });

        target.addDynamicallyLegalOp<tensor::FromElementsOp>([](tensor::FromElementsOp op) {
            if (auto rankedType = dyn_cast<RankedTensorType>(op.getResult().getType())) {
                return rankedType.getRank() != 0;
            }
            return true;
        });

        target.addDynamicallyLegalOp<tensor::ExtractOp>([](tensor::ExtractOp op) {
            if (auto rankedType = dyn_cast<RankedTensorType>(op.getTensor().getType())) {
                return rankedType.getRank() != 0;
            }
            return true;
        });

        target.addDynamicallyLegalOp<func::FuncOp>(
            [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return typeConverter.isSignatureLegal(op.getCalleeType()); });
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

        RewritePatternSet patterns(context);
        patterns.add<DetensorizeExtractPattern>(typeConverter, context);
        patterns.add<DetensorizeFromElementsPattern>(typeConverter, context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        populateCallOpTypeConversionPattern(patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDetensorizeFunctionBoundaryPass()
{
    return std::make_unique<DetensorizeFunctionBoundaryPass>();
}

} // namespace catalyst
