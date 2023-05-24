// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

struct DynVectorBuilder {
    Value dataField, sizeField, capacityField;
    Type elementType;

    static FailureOr<DynVectorBuilder> get(Location loc, TypeConverter *typeConverter,
                                           TypedValue<ParameterVectorType> vector, OpBuilder &b)
    {
        SmallVector<Type> resultTypes;
        if (failed(typeConverter->convertType(vector.getType(), resultTypes))) {
            return failure();
        }

        auto unpacked = b.create<UnrealizedConversionCastOp>(loc, resultTypes, vector);
        return DynVectorBuilder{.dataField = unpacked.getResult(0),
                                .sizeField = unpacked.getResult(1),
                                .capacityField = unpacked.getResult(2),
                                .elementType = vector.getType().getElementType()};
    }

    FlatSymbolRefAttr getOrInsertPushFunction(Location loc, ModuleOp moduleOp, OpBuilder &b) const
    {
        MLIRContext *ctx = b.getContext();
        std::string funcName = "__grad_vec_push";
        llvm::raw_string_ostream nameStream{funcName};
        nameStream << elementType;
        if (moduleOp.lookupSymbol<func::FuncOp>(funcName))
            return SymbolRefAttr::get(ctx, funcName);

        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(moduleOp.getBody());

        auto pushFnType = FunctionType::get(
            ctx, /*inputs=*/
            {dataField.getType(), sizeField.getType(), capacityField.getType(), elementType},
            /*outputs=*/{});
        auto pushFn = b.create<func::FuncOp>(loc, funcName, pushFnType);
        pushFn.setPrivate();

        Block *entryBlock = pushFn.addEntryBlock();
        b.setInsertionPointToStart(entryBlock);
        BlockArgument elementsField = pushFn.getArgument(0);
        BlockArgument sizeField = pushFn.getArgument(1);
        BlockArgument capacityField = pushFn.getArgument(2);
        BlockArgument value = pushFn.getArgument(3);

        Value sizeVal = b.create<memref::LoadOp>(loc, sizeField);
        Value capacityVal = b.create<memref::LoadOp>(loc, capacityField);

        Value predicate =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, sizeVal, capacityVal);
        b.create<scf::IfOp>(loc, predicate, [&](OpBuilder &thenBuilder, Location loc) {
            Value two = thenBuilder.create<arith::ConstantIndexOp>(loc, 2);
            Value newCapacity = thenBuilder.create<arith::MulIOp>(loc, capacityVal, two);
            Value oldElements = thenBuilder.create<memref::LoadOp>(loc, elementsField);
            Value newElements = thenBuilder.create<memref::ReallocOp>(
                loc, cast<MemRefType>(oldElements.getType()), oldElements, newCapacity);
            thenBuilder.create<memref::StoreOp>(loc, newElements, elementsField);
            thenBuilder.create<memref::StoreOp>(loc, newCapacity, capacityField);
            thenBuilder.create<scf::YieldOp>(loc);
        });

        Value elementsVal = b.create<memref::LoadOp>(loc, elementsField);
        b.create<memref::StoreOp>(loc, value, elementsVal,
                                  /*indices=*/sizeVal);

        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        Value newSize = b.create<arith::AddIOp>(loc, sizeVal, one);

        b.create<memref::StoreOp>(loc, newSize, sizeField);
        b.create<func::ReturnOp>(loc);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void emitPush(Location loc, Value value, OpBuilder &b, FlatSymbolRefAttr pushFn) const
    {
        b.create<func::CallOp>(loc, pushFn, /*results=*/TypeRange{},
                               /*operands=*/ValueRange{dataField, sizeField, capacityField, value});
    }
};

struct LowerInitVector : public OpConversionPattern<InitVectorOp> {
    using OpConversionPattern<InitVectorOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(InitVectorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> resultTypes;
        if (failed(getTypeConverter()->convertType(op.getType(), resultTypes))) {
            op.emitError() << "Failed to convert type " << op.getType();
            return failure();
        }
        Value capacity = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
        Value initialSize = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto dataType = resultTypes[0].cast<MemRefType>();
        auto sizeType = resultTypes[1].cast<MemRefType>();
        auto capacityType = resultTypes[2].cast<MemRefType>();
        Value buffer = rewriter.create<memref::AllocOp>(
            op.getLoc(), dataType.getElementType().cast<MemRefType>(),
            /*dynamicSize=*/capacity);
        Value bufferField = rewriter.create<memref::AllocaOp>(op.getLoc(), dataType);
        Value sizeField = rewriter.create<memref::AllocaOp>(op.getLoc(), sizeType);
        Value capacityField = rewriter.create<memref::AllocaOp>(op.getLoc(), capacityType);
        rewriter.create<memref::StoreOp>(op.getLoc(), buffer, bufferField);
        rewriter.create<memref::StoreOp>(op.getLoc(), initialSize, sizeField);
        rewriter.create<memref::StoreOp>(op.getLoc(), capacity, capacityField);
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), ValueRange{bufferField, sizeField, capacityField});
        return success();
    }
};

struct LowerPushVector : public OpConversionPattern<PushVectorOp> {
    using OpConversionPattern<PushVectorOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(PushVectorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        FailureOr<DynVectorBuilder> dynVectorBuilder =
            DynVectorBuilder::get(op.getLoc(), getTypeConverter(), op.getVector(), rewriter);
        if (failed(dynVectorBuilder)) {
            return failure();
        }
        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr pushFn =
            dynVectorBuilder.value().getOrInsertPushFunction(op.getLoc(), moduleOp, rewriter);
        dynVectorBuilder.value().emitPush(op.getLoc(), op.getValue(), rewriter, pushFn);
        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerVectorSize : public OpConversionPattern<VectorSizeOp> {
    using OpConversionPattern<VectorSizeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(VectorSizeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        FailureOr<DynVectorBuilder> dynVectorBuilder =
            DynVectorBuilder::get(op.getLoc(), getTypeConverter(), op.getVector(), rewriter);
        if (failed(dynVectorBuilder)) {
            return failure();
        }

        Value size =
            rewriter.create<memref::LoadOp>(op.getLoc(), dynVectorBuilder.value().sizeField);
        rewriter.replaceOp(op, size);
        return success();
    }
};

struct GradientLoweringPass : public OperationPass<ModuleOp> {
    GradientLoweringPass() : OperationPass<ModuleOp>(TypeID::get<GradientLoweringPass>()) {}
    GradientLoweringPass(const GradientLoweringPass &other) : OperationPass<ModuleOp>(other) {}

    StringRef getName() const override { return "GradientLoweringPass"; }

    StringRef getArgument() const override { return "lower-gradients"; }

    StringRef getDescription() const override
    {
        return "Lower gradient operation to MLIR operation.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<index::IndexDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp op = getOperation();
        TypeConverter vectorTypeConverter;
        vectorTypeConverter.addConversion([](Type type) -> llvm::Optional<Type> {
            if (MemRefType::isValidElementType(type)) {
                return type;
            }
            return llvm::None;
        });
        vectorTypeConverter.addConversion(
            [](ParameterVectorType type, SmallVectorImpl<Type> &resultTypes) {
                // Data
                resultTypes.push_back(MemRefType::get(
                    {}, MemRefType::get({ShapedType::kDynamic}, type.getElementType())));
                auto indexMemRef = MemRefType::get({}, IndexType::get(type.getContext()));
                // Size
                resultTypes.push_back(indexMemRef);
                // Capacity
                resultTypes.push_back(indexMemRef);
                return success();
            });

        RewritePatternSet gradientPatterns(&getContext());
        populateLoweringPatterns(gradientPatterns, lowerOnly);

        // This is required to remove qubit values returned by if/for ops in the
        // quantum gradient function of the parameter-shift pattern.
        scf::IfOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        scf::ForOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::InsertOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::DeallocOp::getCanonicalizationPatterns(gradientPatterns, &getContext());

        if (failed(applyPatternsAndFoldGreedily(op, std::move(gradientPatterns)))) {
            return signalPassFailure();
        }

        RewritePatternSet gradientVectorPatterns(&getContext());
        gradientVectorPatterns.add<LowerInitVector>(vectorTypeConverter, &getContext());
        gradientVectorPatterns.add<LowerPushVector>(vectorTypeConverter, &getContext());
        gradientVectorPatterns.add<LowerVectorSize>(vectorTypeConverter, &getContext());
        ConversionTarget target(getContext());
        target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                               scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addIllegalOp<InitVectorOp, PushVectorOp, VectorSizeOp>();

        if (failed(applyPartialConversion(op, target, std::move(gradientVectorPatterns)))) {
            return signalPassFailure();
        }
    }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<GradientLoweringPass>(*this);
    }

  protected:
    Option<std::string> lowerOnly{
        *this, "only", llvm::cl::desc("Restrict lowering to a specific type of gradient.")};
};

} // namespace gradient

std::unique_ptr<Pass> createGradientLoweringPass()
{
    return std::make_unique<gradient::GradientLoweringPass>();
}

} // namespace catalyst
