#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"

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

        auto unpacked = UnrealizedConversionCastOp::create(b, loc, resultTypes, list);
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
        auto pushFn = func::FuncOp::create(b, loc, funcName, pushFnType);
        pushFn.setPrivate();

        Block *entryBlock = pushFn.addEntryBlock();
        b.setInsertionPointToStart(entryBlock);
        BlockArgument elementsField = pushFn.getArgument(0);
        BlockArgument sizeField = pushFn.getArgument(1);
        BlockArgument capacityField = pushFn.getArgument(2);
        BlockArgument value = pushFn.getArgument(3);

        Value sizeVal = memref::LoadOp::create(b, loc, sizeField);
        Value capacityVal = memref::LoadOp::create(b, loc, capacityField);

        Value predicate =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, sizeVal, capacityVal);
        scf::IfOp::create(b, loc, predicate, [&](OpBuilder &thenBuilder, Location loc) {
            Value two = arith::ConstantIndexOp::create(thenBuilder, loc, 2);
            Value newCapacity = arith::MulIOp::create(thenBuilder, loc, capacityVal, two);
            Value oldElements = memref::LoadOp::create(thenBuilder, loc, elementsField);
            Value newElements =
                memref::ReallocOp::create(thenBuilder, loc, cast<MemRefType>(oldElements.getType()),
                                          oldElements, newCapacity);
            memref::StoreOp::create(thenBuilder, loc, newElements, elementsField);
            memref::StoreOp::create(thenBuilder, loc, newCapacity, capacityField);
            scf::YieldOp::create(thenBuilder, loc);
        });

        Value elementsVal = memref::LoadOp::create(b, loc, elementsField);
        memref::StoreOp::create(b, loc, value, elementsVal,
                                /*indices=*/sizeVal);

        Value one = arith::ConstantIndexOp::create(b, loc, 1);
        Value newSize = arith::AddIOp::create(b, loc, sizeVal, one);

        memref::StoreOp::create(b, loc, newSize, sizeField);
        func::ReturnOp::create(b, loc);
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
        auto popFn = func::FuncOp::create(builder, loc, funcName, popFnType);
        popFn.setPrivate();

        Block *entryBlock = popFn.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        Region::BlockArgListType arguments = popFn.getArguments();
        BlockArgument elementsField = arguments[0];
        BlockArgument sizeField = arguments[1];

        Value elementsVal = memref::LoadOp::create(builder, loc, elementsField);
        Value sizeVal = memref::LoadOp::create(builder, loc, sizeField);
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        Value newSize = arith::SubIOp::create(builder, loc, sizeVal, one);
        Value poppedVal = memref::LoadOp::create(builder, loc, elementsVal, newSize);

        memref::StoreOp::create(builder, loc, newSize, sizeField);
        func::ReturnOp::create(builder, loc, poppedVal);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void emitPush(Location loc, Value value, OpBuilder &b, FlatSymbolRefAttr pushFn) const
    {
        func::CallOp::create(b, loc, pushFn, /*results=*/TypeRange{},
                             /*operands=*/ValueRange{dataField, sizeField, capacityField, value});
    }

    Value emitPop(Location loc, OpBuilder &builder, FlatSymbolRefAttr popFn) const
    {
        auto callOp =
            func::CallOp::create(builder, loc, popFn, /*results=*/elementType,
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
        Value capacity = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 32);
        Value initialSize = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
        auto dataType = cast<MemRefType>(resultTypes[0]);
        auto sizeType = cast<MemRefType>(resultTypes[1]);
        auto capacityType = cast<MemRefType>(resultTypes[2]);
        Value buffer = memref::AllocOp::create(rewriter, op.getLoc(),
                                               cast<MemRefType>(dataType.getElementType()),
                                               /*dynamicSize=*/capacity);
        Value bufferField = memref::AllocOp::create(rewriter, op.getLoc(), dataType);
        Value sizeField = memref::AllocOp::create(rewriter, op.getLoc(), sizeType);
        Value capacityField = memref::AllocOp::create(rewriter, op.getLoc(), capacityType);
        memref::StoreOp::create(rewriter, op.getLoc(), buffer, bufferField);
        memref::StoreOp::create(rewriter, op.getLoc(), initialSize, sizeField);
        memref::StoreOp::create(rewriter, op.getLoc(), capacity, capacityField);
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

        Value data = memref::LoadOp::create(rewriter, op.getLoc(), arraylistBuilder->dataField);
        memref::DeallocOp::create(rewriter, op.getLoc(), data);
        memref::DeallocOp::create(rewriter, op.getLoc(), arraylistBuilder->dataField);
        memref::DeallocOp::create(rewriter, op.getLoc(), arraylistBuilder->sizeField);
        memref::DeallocOp::create(rewriter, op.getLoc(), arraylistBuilder->capacityField);
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
            memref::LoadOp::create(rewriter, op.getLoc(), arraylistBuilder.value().dataField);
        auto memrefType = cast<MemRefType>(data.getType());
        Value size =
            memref::LoadOp::create(rewriter, op.getLoc(), arraylistBuilder.value().sizeField);
        SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)}, sizes{size},
            strides{rewriter.getIndexAttr(1)};
        Value dataView = memref::SubViewOp::create(rewriter, op.getLoc(), memrefType, data, offsets,
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
