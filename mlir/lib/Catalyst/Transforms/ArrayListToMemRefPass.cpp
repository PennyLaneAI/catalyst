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
                                           TypedValue<ArrayListType> list, OpBuilder &b) {
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

    FlatSymbolRefAttr getOrInsertPushFunction(Location loc, ModuleOp moduleOp, OpBuilder &b) const {
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
                                             OpBuilder &builder) const {
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

    /// Copy `count` elements from `source[sourceOffset + i]` to `destination[destinationOffset +
    /// i]` between two rank-1 memrefs.
    ///
    /// This is an explicit loop rather than a `memref.copy` between `memref.subview`s of the two
    /// buffers. A subview at a dynamic offset has a non-identity layout, and
    /// `finalize-memref-to-llvm` lowers a copy involving such a memref to a call to the
    /// `memrefCopy` runtime symbol, which compiled Catalyst programs do not link.
    static void emitElementCopyLoop(OpBuilder &b, Location loc, Value source, Value sourceOffset,
                                    Value destination, Value destinationOffset, Value count) {
        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        Value one = arith::ConstantIndexOp::create(b, loc, 1);
        scf::ForOp loop = scf::ForOp::create(b, loc, zero, count, one);

        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        Value offset = loop.getInductionVar();
        Value sourceIndex = arith::AddIOp::create(b, loc, sourceOffset, offset);
        Value destinationIndex = arith::AddIOp::create(b, loc, destinationOffset, offset);
        Value element = memref::LoadOp::create(b, loc, source, sourceIndex);
        memref::StoreOp::create(b, loc, element, destination, destinationIndex);
    }

    /// Build (or look up) `__catalyst_arraylist_push_block<element-type>`, which appends every
    /// element of a contiguous rank-1 block to the end of the list, growing the storage as needed.
    ///
    /// The elements are *copied* into the list's own storage, so the caller keeps ownership of the
    /// block and may reuse or free it right after the call. Without that copy the list would retain
    /// a pointer that no MLIR pass can see, which is what makes storing whole buffers in a list
    /// safe under buffer hoisting and deallocation.
    FlatSymbolRefAttr getOrInsertPushBlockFunction(Location loc, ModuleOp moduleOp,
                                                   OpBuilder &b) const {
        MLIRContext *ctx = b.getContext();
        std::string funcName = "__catalyst_arraylist_push_block";
        llvm::raw_string_ostream nameStream{funcName};
        nameStream << elementType;
        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(moduleOp.getBody());

        auto blockType = MemRefType::get({ShapedType::kDynamic}, elementType);
        auto pushFnType = FunctionType::get(
            ctx, /*inputs=*/
            {dataField.getType(), sizeField.getType(), capacityField.getType(), blockType},
            /*outputs=*/{});
        auto pushFn = func::FuncOp::create(b, loc, funcName, pushFnType);
        pushFn.setPrivate();

        Block *entryBlock = pushFn.addEntryBlock();
        b.setInsertionPointToStart(entryBlock);
        BlockArgument elementsField = pushFn.getArgument(0);
        BlockArgument sizeField = pushFn.getArgument(1);
        BlockArgument capacityField = pushFn.getArgument(2);
        BlockArgument block = pushFn.getArgument(3);

        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        Value sizeVal = memref::LoadOp::create(b, loc, sizeField);
        Value capacityVal = memref::LoadOp::create(b, loc, capacityField);
        Value count = memref::DimOp::create(b, loc, block, zero);
        Value newSize = arith::AddIOp::create(b, loc, sizeVal, count);

        Value predicate =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, newSize, capacityVal);
        scf::IfOp::create(b, loc, predicate, [&](OpBuilder &thenBuilder, Location loc) {
            Value two = arith::ConstantIndexOp::create(thenBuilder, loc, 2);
            Value doubled = arith::MulIOp::create(thenBuilder, loc, capacityVal, two);
            // Keep the geometric growth of the element-wise push, but never grow by less than
            // this block needs.
            Value newCapacity = arith::MaxUIOp::create(thenBuilder, loc, doubled, newSize);
            Value oldElements = memref::LoadOp::create(thenBuilder, loc, elementsField);
            Value newElements =
                memref::ReallocOp::create(thenBuilder, loc, cast<MemRefType>(oldElements.getType()),
                                          oldElements, newCapacity);
            memref::StoreOp::create(thenBuilder, loc, newElements, elementsField);
            memref::StoreOp::create(thenBuilder, loc, newCapacity, capacityField);
            scf::YieldOp::create(thenBuilder, loc);
        });

        Value elementsVal = memref::LoadOp::create(b, loc, elementsField);
        emitElementCopyLoop(b, loc, /*source=*/block, /*sourceOffset=*/zero,
                            /*destination=*/elementsVal, /*destinationOffset=*/sizeVal, count);
        memref::StoreOp::create(b, loc, newSize, sizeField);
        func::ReturnOp::create(b, loc);
        return SymbolRefAttr::get(ctx, funcName);
    }

    /// Build (or look up) `__catalyst_arraylist_pop_block<element-type>`, which removes as many
    /// elements from the end of the list as the destination block holds and copies them into it.
    /// The destination is owned by the caller, so the popped elements stay valid after the list is
    /// deallocated.
    FlatSymbolRefAttr getOrInsertPopBlockFunction(Location loc, ModuleOp moduleOp,
                                                  OpBuilder &b) const {
        MLIRContext *ctx = b.getContext();
        std::string funcName = "__catalyst_arraylist_pop_block";
        llvm::raw_string_ostream nameStream{funcName};
        nameStream << elementType;
        if (moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
            return SymbolRefAttr::get(ctx, funcName);
        }

        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(moduleOp.getBody());

        auto blockType = MemRefType::get({ShapedType::kDynamic}, elementType);
        auto popFnType = FunctionType::get(
            ctx, /*inputs=*/
            {dataField.getType(), sizeField.getType(), capacityField.getType(), blockType},
            /*outputs=*/{});
        auto popFn = func::FuncOp::create(b, loc, funcName, popFnType);
        popFn.setPrivate();

        Block *entryBlock = popFn.addEntryBlock();
        b.setInsertionPointToStart(entryBlock);
        BlockArgument elementsField = popFn.getArgument(0);
        BlockArgument sizeField = popFn.getArgument(1);
        BlockArgument destination = popFn.getArgument(3);

        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        Value sizeVal = memref::LoadOp::create(b, loc, sizeField);
        Value count = memref::DimOp::create(b, loc, destination, zero);
        Value newSize = arith::SubIOp::create(b, loc, sizeVal, count);

        Value elementsVal = memref::LoadOp::create(b, loc, elementsField);
        emitElementCopyLoop(b, loc, /*source=*/elementsVal, /*sourceOffset=*/newSize,
                            /*destination=*/destination, /*destinationOffset=*/zero, count);
        memref::StoreOp::create(b, loc, newSize, sizeField);
        func::ReturnOp::create(b, loc);
        return SymbolRefAttr::get(ctx, funcName);
    }

    void emitPush(Location loc, Value value, OpBuilder &b, FlatSymbolRefAttr pushFn) const {
        func::CallOp::create(b, loc, pushFn, /*results=*/TypeRange{},
                             /*operands=*/ValueRange{dataField, sizeField, capacityField, value});
    }

    /// Emit a call to one of the block helpers, casting `block` to the dynamically sized rank-1
    /// memref they take.
    void emitBlockCall(Location loc, Value block, OpBuilder &b, FlatSymbolRefAttr blockFn) const {
        auto dynamicBlockType = MemRefType::get({ShapedType::kDynamic}, elementType);
        if (block.getType() != dynamicBlockType) {
            block = memref::CastOp::create(b, loc, dynamicBlockType, block);
        }
        func::CallOp::create(b, loc, blockFn, /*results=*/TypeRange{},
                             /*operands=*/ValueRange{dataField, sizeField, capacityField, block});
    }

    Value emitPop(Location loc, OpBuilder &builder, FlatSymbolRefAttr popFn) const {
        auto callOp =
            func::CallOp::create(builder, loc, popFn, /*results=*/elementType,
                                 /*operands=*/ValueRange{dataField, sizeField, capacityField});
        return callOp.getResult(0);
    }
};

struct LowerListInit : public OpConversionPattern<ListInitOp> {
    using OpConversionPattern<ListInitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListInitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
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
                                  ConversionPatternRewriter &rewriter) const override {
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
                                  ConversionPatternRewriter &rewriter) const override {
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
                                  ConversionPatternRewriter &rewriter) const override {
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

/// Returns the block operand of a block push/pop as the contiguous rank-1 memref the generated
/// helpers expect, or failure (with a diagnostic) if bufferization has not run on it.
static FailureOr<Value> getBufferizedBlock(Operation *op, Value block, StringRef name,
                                           Type elementType) {
    auto memrefType = dyn_cast<MemRefType>(block.getType());
    if (!memrefType || memrefType.getRank() != 1 || !memrefType.getLayout().isIdentity() ||
        memrefType.getElementType() != elementType) {
        return op->emitOpError() << "expects '" << name << "' to be a contiguous rank-1 memref of "
                                 << elementType << " here, but got " << block.getType()
                                 << ". Run one-shot-bufferize before this pass.";
    }
    return block;
}

struct LowerListPushBlock : public OpConversionPattern<ListPushBlockOp> {
    using OpConversionPattern<ListPushBlockOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListPushBlockOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }
        FailureOr<Value> block =
            getBufferizedBlock(op, op.getElements(), "elements", arraylistBuilder->elementType);
        if (failed(block)) {
            return failure();
        }

        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr pushBlockFn =
            arraylistBuilder->getOrInsertPushBlockFunction(op.getLoc(), moduleOp, rewriter);
        arraylistBuilder->emitBlockCall(op.getLoc(), *block, rewriter, pushBlockFn);
        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerListPopBlock : public OpConversionPattern<ListPopBlockOp> {
    using OpConversionPattern<ListPopBlockOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListPopBlockOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto typeConverter = getTypeConverter();
        FailureOr<ArrayListBuilder> arraylistBuilder =
            ArrayListBuilder::get(op.getLoc(), typeConverter, op.getList(), rewriter);
        if (failed(arraylistBuilder)) {
            return failure();
        }
        FailureOr<Value> destination = getBufferizedBlock(op, op.getDestination(), "destination",
                                                          arraylistBuilder->elementType);
        if (failed(destination)) {
            return failure();
        }

        auto moduleOp = op->getParentOfType<ModuleOp>();
        FlatSymbolRefAttr popBlockFn =
            arraylistBuilder->getOrInsertPopBlockFunction(op.getLoc(), moduleOp, rewriter);
        arraylistBuilder->emitBlockCall(op.getLoc(), *destination, rewriter, popBlockFn);
        rewriter.eraseOp(op);
        return success();
    }
};

struct LowerListLoadData : public OpConversionPattern<ListLoadDataOp> {
    using OpConversionPattern<ListLoadDataOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ListLoadDataOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
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

    void runOnOperation() override {
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
        patterns.add<LowerListPushBlock>(arraylistTypeConverter, context);
        patterns.add<LowerListPopBlock>(arraylistTypeConverter, context);
        patterns.add<LowerListLoadData>(arraylistTypeConverter, context);

        ConversionTarget target(getContext());
        target.addLegalDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                               scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addIllegalOp<ListInitOp, ListDeallocOp, ListPushOp, ListPopOp, ListPushBlockOp,
                            ListPopBlockOp, ListLoadDataOp>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace
