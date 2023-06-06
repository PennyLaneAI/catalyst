#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst;

namespace {
/**
 * A utility builder that aids in lowering dynamically-resizable array lists.
 *
 * ArrayLists are lowered to a size, a capacity, and a memref that stores the list data.
 * Each component is made mutable by being stored in a rank-0 memref, such that the list data is
 * lowered to (memref<memref<?x{element-type}>>, memref<index>, memref<index>).
 */
struct ArrayListBuilder {
    Value dataField;
    Value sizeField;
    Value capacityField;
    Type elementType;

    static FailureOr<ArrayListBuilder> get(Location loc, TypeConverter *typeConverter,
                                           TypedValue<ArrayListType> list, OpBuilder &b)
    {
        SmallVector<Type> resultTypes;
        if (failed(typeConverter->convertType(list.getType(), resultTypes))) {
            return failure();
        }

        auto unpacked = b.create<UnrealizedConversionCastOp>(loc, resultTypes, list);
        return ArrayListBuilder{.dataField = unpacked.getResult(0),
                                .sizeField = unpacked.getResult(1),
                                .capacityField = unpacked.getResult(2),
                                .elementType = list.getType().getElementType()};
    }

    FlatSymbolRefAttr getOrInsertPushFunction(Location loc, ModuleOp moduleOp, OpBuilder &b) const
    {
        MLIRContext *ctx = b.getContext();
        std::string funcName = "__catalyst_arraylist_push";
        llvm::raw_string_ostream nameStream{funcName};
        nameStream << elementType;
        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

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

struct LowerListInit : public OpConversionPattern<ListInitOp> {
    using OpConversionPattern<ListInitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListInitOp op, OpAdaptor adaptor,
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

struct LowerListPush : public OpConversionPattern<ListPushOp> {
    using OpConversionPattern<ListPushOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListPushOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        FailureOr<ArrayListBuilder> dynVectorBuilder =
            ArrayListBuilder::get(op.getLoc(), getTypeConverter(), op.getList(), rewriter);
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

struct LowerListLoadData : public OpConversionPattern<ListLoadDataOp> {
    using OpConversionPattern<ListLoadDataOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListLoadDataOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        FailureOr<ArrayListBuilder> dynVectorBuilder =
            ArrayListBuilder::get(op.getLoc(), getTypeConverter(), op.getList(), rewriter);
        if (failed(dynVectorBuilder)) {
            return failure();
        }

        // Ensure the result memref has the correct underlying size (which may be different than the
        // list's underlying memref due to the geometric reallocation).
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

struct ArrayListToMemRefPass : catalyst::impl::ArrayListToMemRefPassBase<ArrayListToMemRefPass> {
    using ArrayListToMemRefPassBase::ArrayListToMemRefPassBase;

    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        TypeConverter arraylistTypeConverter;

        arraylistTypeConverter.addConversion([](Type type) -> llvm::Optional<Type> {
            if (MemRefType::isValidElementType(type)) {
                return type;
            }
            return llvm::None;
        });
        arraylistTypeConverter.addConversion(
            [](ArrayListType type, SmallVectorImpl<Type> &resultTypes) {
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

        RewritePatternSet patterns(context);
        patterns.add<LowerListInit>(arraylistTypeConverter, context);
        patterns.add<LowerListPush>(arraylistTypeConverter, context);
        patterns.add<LowerListLoadData>(arraylistTypeConverter, context);

        ConversionTarget target(getContext());
        target.addLegalDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                               scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addIllegalOp<ListInitOp, ListPushOp, ListLoadDataOp>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> catalyst::createArrayListToMemRefPass()
{
    return std::make_unique<ArrayListToMemRefPass>();
}
