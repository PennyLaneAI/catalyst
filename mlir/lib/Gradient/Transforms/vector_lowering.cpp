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
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"

using namespace mlir;

namespace catalyst {
namespace gradient {
/**
 * A utility builder that aids in lowering dynamically-resizable parameter vectors.
 *
 * ParameterVectors are lowered to a size, a capacity, and a memref that stores the vector data.
 * Each component is made mutable by being stored in a rank-0 memref, such that the vector data is
 * lowered to (memref<memref<?x{element-type}>>, memref<index>, memref<index>).
 */
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

struct LowerInitVector : public OpConversionPattern<VectorInitOp> {
    using OpConversionPattern<VectorInitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(VectorInitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> resultTypes;
        if (failed(getTypeConverter()->convertType(op.getType(), resultTypes))) {
            op.emitError() << "Failed to convert type " << op.getType();
            return failure();
        }
        Value capacity = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 32);
        Value initialSize = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto dataType = resultTypes[0].cast<MemRefType>();
        auto sizeType = resultTypes[1].cast<MemRefType>();
        auto capacityType = resultTypes[2].cast<MemRefType>();
        Value buffer = rewriter.create<memref::AllocOp>(
            op.getLoc(), dataType.getElementType().cast<MemRefType>(),
            /*dynamicSize=*/capacity);
        Value bufferField = rewriter.create<memref::AllocOp>(op.getLoc(), dataType);
        Value sizeField = rewriter.create<memref::AllocOp>(op.getLoc(), sizeType);
        Value capacityField = rewriter.create<memref::AllocOp>(op.getLoc(), capacityType);
        rewriter.create<memref::StoreOp>(op.getLoc(), buffer, bufferField);
        rewriter.create<memref::StoreOp>(op.getLoc(), initialSize, sizeField);
        rewriter.create<memref::StoreOp>(op.getLoc(), capacity, capacityField);
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, op.getType(), ValueRange{bufferField, sizeField, capacityField});
        return success();
    }
};

struct LowerPushVector : public OpConversionPattern<VectorPushOp> {
    using OpConversionPattern<VectorPushOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(VectorPushOp op, OpAdaptor adaptor,
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

struct LowerVectorLoadData : public OpConversionPattern<VectorLoadDataOp> {
    using OpConversionPattern<VectorLoadDataOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(VectorLoadDataOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        FailureOr<DynVectorBuilder> dynVectorBuilder =
            DynVectorBuilder::get(op.getLoc(), getTypeConverter(), op.getVector(), rewriter);
        if (failed(dynVectorBuilder)) {
            return failure();
        }

        // Ensure the result memref has the correct underlying size (which may be different than the
        // vector's underlying memref due to the geometric reallocation).
        Value data =
            rewriter.create<memref::LoadOp>(op.getLoc(), dynVectorBuilder.value().dataField);
        auto memrefType = cast<MemRefType>(data.getType());
        Value size =
            rewriter.create<memref::LoadOp>(op.getLoc(), dynVectorBuilder.value().sizeField);
        SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)}, sizes{size},
            strides{rewriter.getIndexAttr(1)};
        Value dataView = rewriter.create<memref::SubViewOp>(op.getLoc(), memrefType, data, offsets,
                                                            sizes, strides);
        rewriter.replaceOp(op, dataView);
        return success();
    }
};

void populateVectorLoweringPatterns(TypeConverter &vectorTypeConverter, RewritePatternSet &patterns)
{
    patterns.add<LowerInitVector>(vectorTypeConverter, patterns.getContext());
    patterns.add<LowerPushVector>(vectorTypeConverter, patterns.getContext());
    patterns.add<LowerVectorSize>(vectorTypeConverter, patterns.getContext());
    patterns.add<LowerVectorLoadData>(vectorTypeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
