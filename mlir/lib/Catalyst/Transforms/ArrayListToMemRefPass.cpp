#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_ARRAYLISTTOMEMREFPASS
#include "Catalyst/Transforms/Passes.h.inc"
} // namespace catalyst

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

    static FailureOr<ArrayListBuilder> get(Location loc, const TypeConverter *typeConverter,
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

    FlatSymbolRefAttr getOrInsertPopFunction(Location loc, ModuleOp moduleOp,
                                             OpBuilder &builder) const
    {
        MLIRContext *ctx = builder.getContext();
        std::string funcName = "__catalyst_arraylist_pop";
        llvm::raw_string_ostream nameStream{funcName};
        nameStream << elementType;
        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto popFnType =
            FunctionType::get(ctx, /*inputs=*/
                              {dataField.getType(), sizeField.getType(), capacityField.getType()},
                              /*outputs=*/elementType);
        auto popFn = builder.create<func::FuncOp>(loc, funcName, popFnType);
        popFn.setPrivate();

        Block *entryBlock = popFn.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        Region::BlockArgListType arguments = popFn.getArguments();
        BlockArgument elementsField = arguments[0];
        BlockArgument sizeField = arguments[1];

        Value elementsVal = builder.create<memref::LoadOp>(loc, elementsField);
        Value sizeVal = builder.create<memref::LoadOp>(loc, sizeField);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        Value newSize = builder.create<arith::SubIOp>(loc, sizeVal, one);
        Value poppedVal = builder.create<memref::LoadOp>(loc, elementsVal, newSize);

        builder.create<memref::StoreOp>(loc, newSize, sizeField);
        builder.create<func::ReturnOp>(loc, poppedVal);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void emitPush(Location loc, Value value, OpBuilder &b, FlatSymbolRefAttr pushFn) const
    {
        b.create<func::CallOp>(loc, pushFn, /*results=*/TypeRange{},
                               /*operands=*/ValueRange{dataField, sizeField, capacityField, value});
    }

    Value emitPop(Location loc, OpBuilder &builder, FlatSymbolRefAttr popFn) const
    {
        auto callOp = builder.create<func::CallOp>(
            loc, popFn, /*results=*/elementType,
            /*operands=*/ValueRange{dataField, sizeField, capacityField});
        return callOp.getResult(0);
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
        auto dataType = cast<MemRefType>(resultTypes[0]);
        auto sizeType = cast<MemRefType>(resultTypes[1]);
        auto capacityType = cast<MemRefType>(resultTypes[2]);
        Value buffer = rewriter.create<memref::AllocOp>(op.getLoc(),
                                                        cast<MemRefType>(dataType.getElementType()),
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

struct LowerListDealloc : public OpConversionPattern<ListDeallocOp> {
    using OpConversionPattern<ListDeallocOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListDeallocOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }

        Value data = rewriter.create<memref::LoadOp>(op.getLoc(), arraylistBuilder->dataField);
        rewriter.create<memref::DeallocOp>(op.getLoc(), data);
        rewriter.create<memref::DeallocOp>(op.getLoc(), arraylistBuilder->dataField);
        rewriter.create<memref::DeallocOp>(op.getLoc(), arraylistBuilder->sizeField);
        rewriter.create<memref::DeallocOp>(op.getLoc(), arraylistBuilder->capacityField);
        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerListPush : public OpConversionPattern<ListPushOp> {
    using OpConversionPattern<ListPushOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListPushOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }
        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr pushFn =
            arraylistBuilder.value().getOrInsertPushFunction(op.getLoc(), moduleOp, rewriter);
        arraylistBuilder.value().emitPush(op.getLoc(), op.getValue(), rewriter, pushFn);
        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerListPop : public OpConversionPattern<ListPopOp> {
    using OpConversionPattern<ListPopOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListPopOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }
        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr popFn =
            arraylistBuilder.value().getOrInsertPopFunction(op.getLoc(), moduleOp, rewriter);
        Value poppedVal = arraylistBuilder->emitPop(op.getLoc(), rewriter, popFn);
        rewriter.replaceOp(op, poppedVal);
        return success();
    }
};

struct LowerListLoadData : public OpConversionPattern<ListLoadDataOp> {
    using OpConversionPattern<ListLoadDataOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListLoadDataOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }

        // Ensure the result memref has the correct underlying size (which may be different than the
        // list's underlying memref due to the geometric reallocation).
        Value data =
            rewriter.create<memref::LoadOp>(op.getLoc(), arraylistBuilder.value().dataField);
        auto memrefType = cast<MemRefType>(data.getType());
        Value size =
            rewriter.create<memref::LoadOp>(op.getLoc(), arraylistBuilder.value().sizeField);
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

        arraylistTypeConverter.addConversion([](Type type) -> std::optional<Type> {
            if (MemRefType::isValidElementType(type)) {
                return type;
            }
            return std::nullopt;
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
        patterns.add<LowerListDealloc>(arraylistTypeConverter, context);
        patterns.add<LowerListPush>(arraylistTypeConverter, context);
        patterns.add<LowerListPop>(arraylistTypeConverter, context);
        patterns.add<LowerListLoadData>(arraylistTypeConverter, context);

        ConversionTarget target(getContext());
        target.addLegalDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                               scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addIllegalOp<ListInitOp, ListDeallocOp, ListPushOp, ListPopOp, ListLoadDataOp>();

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
