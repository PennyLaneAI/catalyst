#define DEBUG_TYPE "detensorize-func-boundary"

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
        mlir::Value tensor = op.getTensor();
        mlir::Operation *definingOp = tensor.getDefiningOp();

        if (mlir::isa<mlir::BlockArgument>(tensor) ||
            (definingOp && mlir::isa<mlir::func::CallOp>(definingOp))) {
            llvm::errs() << "matched detensorized extract\n";
            rewriter.replaceOp(op, adaptor.getTensor());
            return success();
        }
        llvm::errs() << "not matched detensorized extract\n";

        return failure();
    }
};

struct DetensorizeFromElementsPattern : public OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getOperands().size() != 1) {
            llvm::errs() << "FromElementsOp has more than one operand, cannot detensorize\n";
            return failure();
        }
        mlir::Value result = op.getResult();

        if (!result.hasOneUse()) {
            llvm::errs() << "result.hasOneUse() " << result.hasOneUse() << "\n";
            return failure();
        }
        mlir::Operation *user = *result.user_begin();

        if (mlir::isa<mlir::func::ReturnOp>(user) || mlir::isa<mlir::func::CallOp>(user)) {
            rewriter.replaceOp(op, adaptor.getElements()[0]);
            return success();
        }

        return failure();
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

        typeConverter.addTargetMaterialization([](OpBuilder &builder, RankedTensorType resultType,
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

        // target.addDynamicallyLegalOp<tensor::FromElementsOp>([](tensor::FromElementsOp op) {
        //     if (auto rankedType = dyn_cast<RankedTensorType>(op.getResult().getType())) {
        //         return rankedType.getRank() != 0;
        //     }
        //     return true;
        // });

        // target.addDynamicallyLegalOp<tensor::ExtractOp>([](tensor::ExtractOp op) {
        //     if (auto rankedType = dyn_cast<RankedTensorType>(op.getTensor().getType())) {
        //         return rankedType.getRank() != 0;
        //     }
        //     return true;
        // });

        target.addDynamicallyLegalOp<tensor::FromElementsOp>([](tensor::FromElementsOp op) {
            if (op.getOperands().size() != 1) {
                return true; // Is Legal.
            }

            mlir::Value result = op.getResult();
            if (!result.hasOneUse()) {
                return true; // Is Legal.
            }

            mlir::Operation *user = *result.user_begin();

            bool patternWillMatch =
                mlir::isa<mlir::func::ReturnOp>(user) || mlir::isa<mlir::func::CallOp>(user);

            return !patternWillMatch;
        });

        target.addDynamicallyLegalOp<tensor::ExtractOp>([](tensor::ExtractOp op) {
            mlir::Value tensor = op.getTensor();
            auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());

            if (!tensorType || tensorType.getRank() != 0) {
                return true;
            }
            mlir::Operation *definingOp = tensor.getDefiningOp();
            bool patternWouldMatch = mlir::isa<mlir::BlockArgument>(tensor) ||
                                     (definingOp && mlir::isa<mlir::func::CallOp>(definingOp));

            return !patternWouldMatch;
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
