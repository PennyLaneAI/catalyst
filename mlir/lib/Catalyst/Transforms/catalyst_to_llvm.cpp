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

#include <unordered_map>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace {

LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, StringRef value,
                      ModuleOp mod)
{
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = rewriter.create<LLVM::GlobalOp>(
            loc, LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size()),
            true, LLVM::Linkage::Internal, key, rewriter.getStringAttr(value));
    }

    auto idx =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(rewriter.getContext(), 64),
                                          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    return rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)),
        rewriter.create<LLVM::AddressOfOp>(loc, glb), ArrayRef<Value>({idx, idx}));
}

enum NumericType : int8_t {
    index = 0,
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    c64,
    c128,
};

std::optional<int8_t> encodeNumericType(Type elemType)
{
    int8_t typeEncoding;
    if (isa<IndexType>(elemType)) {
        typeEncoding = NumericType::index;
    }
    else if (auto intType = dyn_cast<IntegerType>(elemType)) {
        switch (intType.getWidth()) {
        case 1:
            typeEncoding = NumericType::i1;
            break;
        case 8:
            typeEncoding = NumericType::i8;
            break;
        case 16:
            typeEncoding = NumericType::i16;
            break;
        case 32:
            typeEncoding = NumericType::i32;
            break;
        case 64:
            typeEncoding = NumericType::i64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto floatType = dyn_cast<FloatType>(elemType)) {
        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::f32;
            break;
        case 64:
            typeEncoding = NumericType::f64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto cmplxType = dyn_cast<ComplexType>(elemType)) {
        auto floatType = dyn_cast<FloatType>(cmplxType.getElementType());
        if (!floatType)
            return std::nullopt;

        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::c64;
            break;
        case 64:
            typeEncoding = NumericType::c128;
            break;
        default:
            return std::nullopt;
        }
    }
    else {
        return std::nullopt;
    }
    return typeEncoding;
}

struct PrintOpPattern : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {

        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type voidPtrType = LLVM::LLVMPointerType::get(ctx);

        if (op.getConstVal().has_value()) {
            ModuleOp mod = op->getParentOfType<ModuleOp>();

            StringRef qirName = "__quantum__rt__print_string";

            Type charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, charPtrType);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            StringRef stringValue = op.getConstVal().value();
            std::string symbolName = std::to_string(std::hash<std::string>()(stringValue.str()));
            Value global = getGlobalString(loc, rewriter, symbolName, stringValue, mod);
            rewriter.create<LLVM::CallOp>(loc, fnDecl, global);
            rewriter.eraseOp(op);
        }
        else {
            StringRef qirName = "__quantum__rt__print_tensor";

            // C interface for the print function is an unranked & opaque memref descriptor:
            // {
            //    i64 rank,
            //    void* memref_descriptor,
            //    i8 type_encoding
            // }
            // Where the type_encoding is a simple enum for all supported numeric types:
            //   i1, i16, i32, i64, f32, f64, c64, c128 (see runtime Types.h)
            Type structType = LLVM::LLVMStructType::getLiteral(
                ctx, {IntegerType::get(ctx, 64), voidPtrType, IntegerType::get(ctx, 8)});
            Type structPtrType = LLVM::LLVMPointerType::get(structType);
            SmallVector<Type> argTypes{structPtrType, IntegerType::get(ctx, 1)};
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, argTypes);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            Value memref = op.getVal();
            MemRefType memrefType = cast<MemRefType>(memref.getType());
            Value llvmMemref = adaptor.getVal();
            Type llvmMemrefType = llvmMemref.getType();
            Value structValue = rewriter.create<LLVM::UndefOp>(loc, structType);

            Value rank = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(memrefType.getRank()));
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, rank, 0);

            Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
            Value memrefPtr = rewriter.create<LLVM::AllocaOp>(
                loc, LLVM::LLVMPointerType::get(llvmMemrefType), c1);
            rewriter.create<LLVM::StoreOp>(loc, llvmMemref, memrefPtr);
            memrefPtr = rewriter.create<LLVM::BitcastOp>(loc, voidPtrType, memrefPtr);
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, memrefPtr, 1);

            Type elemType = memrefType.getElementType();
            std::optional<int8_t> typeEncoding = encodeNumericType(elemType);
            if (!typeEncoding.has_value()) {
                return op.emitOpError("Unsupported element type for printing!");
            }
            Value typeValue = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI8IntegerAttr(typeEncoding.value()));
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, typeValue, 2);

            Value structPtr =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(structType), c1);
            rewriter.create<LLVM::StoreOp>(loc, structValue, structPtr);

            Value printDescriptor = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(),
                                                                      op.getPrintDescriptor());
            SmallVector<Value> callArgs{structPtr, printDescriptor};
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, callArgs);
        }

        return success();
    }
};

enum class PrimitiveType : uint8_t {
    // Invalid primitive type to serve as default.
    PRIMITIVE_TYPE_INVALID = 0,

    // Predicates are two-state booleans.
    PRED = 1,

    // Signed integral values of fixed width.
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,

    // Unsigned integral values of fixed width.
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,

    // Floating-point values of fixed width.
    F16 = 10,
    F32 = 11,
    F64 = 12,
};

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth)
{
    switch (src_bitwidth) {
    case 8:
        return PrimitiveType::S8;
    case 16:
        return PrimitiveType::S16;
    case 32:
        return PrimitiveType::S32;
    case 64:
        return PrimitiveType::S64;
    default:
        return PrimitiveType::PRIMITIVE_TYPE_INVALID;
    }
}

PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth)
{
    switch (src_bitwidth) {
    case 8:
        return PrimitiveType::U8;
    case 16:
        return PrimitiveType::U16;
    case 32:
        return PrimitiveType::U32;
    case 64:
        return PrimitiveType::U64;
    default:
        return PrimitiveType::PRIMITIVE_TYPE_INVALID;
    }
}
static PrimitiveType ScalarPrimitiveType(Type type)
{
    // Integer types.
    if (type.isInteger(1))
        return PrimitiveType::PRED;
    if (auto int_type = type.dyn_cast<mlir::IntegerType>()) {
        unsigned int width = int_type.getWidth();
        auto primitive_type = int_type.isUnsigned()
                                  // Unsigned integer types.
                                  ? UnsignedIntegralTypeForBitWidth(width)
                                  // Signed integer types.
                                  : SignedIntegralTypeForBitWidth(width);
        return primitive_type;
    }

    // Floating point types.
    if (type.isF16())
        return PrimitiveType::F16;
    if (type.isF32())
        return PrimitiveType::F32;
    if (type.isF64())
        return PrimitiveType::F64;

    assert(false && "unsupported type id");
    return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

// Encodes memref as LLVM struct value:
//
//   { i8: dtype, i8: rank, ptr<i8>: data,
//     array<2*rank x i64>: sizes_and_strides }
//
// This is a type erased version of the MLIR memref descriptor without base
// pointer. We pack sizes and strides as a single array member, so that on
// the runtime side we can read it back using C flexible array member.
// If the descriptor value is null, we only encode statically known info: dtype,
// rank, and dims, otherwise we also encode dynamic info
Value EncodeMemRef(Location loc, PatternRewriter &rewriter, MemRefType memref_ty, Value descriptor)
{
    auto ctx = rewriter.getContext();
    // Encode sizes together with strides as a single array.
    int64_t sizes_and_strides_size = 2 * memref_ty.getRank();

    // Encoded memref type: !llvm.struct<(i8, i8, ptr<i8>, array<... x i64>)>.
    Type i8 = rewriter.getI8Type();
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    Type arr = LLVM::LLVMArrayType::get(rewriter.getI64Type(), sizes_and_strides_size);
    auto type = LLVM::LLVMStructType::getLiteral(ctx, {i8, i8, ptr, arr});

    // Helper to unpack MLIR strided memref descriptor value.
    std::optional<MemRefDescriptor> desc = std::nullopt;
    if (descriptor) {
        desc = MemRefDescriptor(descriptor);
    }

    PrimitiveType element_dtype = ScalarPrimitiveType(memref_ty.getElementType());

    // Create values for filling encoded memref struct.
    Value dtype = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(static_cast<uint8_t>(element_dtype)));
    Value rank =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8IntegerAttr(memref_ty.getRank()));

    auto i64 = [&](int64_t i) { return rewriter.getI64IntegerAttr(i); };

    // Get the statically known strides and offset from the memref type.
    llvm::SmallVector<int64_t> strides;
    int64_t memref_offset;
    if (failed(getStridesAndOffset(memref_ty, strides, memref_offset)))
        strides.resize(memref_ty.getRank(), ShapedType::kDynamic);

    // Build encoded memref sizes + strides: !llvm.array<... x i64>
    Value payload = rewriter.create<LLVM::UndefOp>(loc, type.getBody()[3]);
    for (unsigned i = 0; i < memref_ty.getRank(); ++i) {
        int64_t dim_size = memref_ty.getDimSize(i);
        int64_t stride_size = strides[i];

        Value dim = ShapedType::isDynamic(dim_size) && desc.has_value()
                        ? desc->size(rewriter, loc, i)
                        : rewriter.create<LLVM::ConstantOp>(loc, i64(dim_size));

        Value stride = ShapedType::isDynamic(stride_size) && desc.has_value()
                           ? desc->stride(rewriter, loc, i)
                           : rewriter.create<LLVM::ConstantOp>(loc, i64(stride_size));

        auto stride_pos = memref_ty.getRank() + i;

        payload = rewriter.create<LLVM::InsertValueOp>(loc, payload, dim, i);
        payload = rewriter.create<LLVM::InsertValueOp>(loc, payload, stride, stride_pos);
    }

    // Construct encoded memref value.
    Value memref = rewriter.create<LLVM::UndefOp>(loc, type);
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, dtype, 0);
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, rank, 1);
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, payload, 3);

    // Previous values almost always are known at compile time, and inserting
    // dynamic values into the struct after all statically know values leads to a
    // better canonicalization and cleaner final LLVM IR.
    if (desc.has_value()) {
        Value offset = (memref_offset == ShapedType::kDynamic)
                           ? desc->offset(rewriter, loc)
                           : rewriter.create<LLVM::ConstantOp>(loc, i64(memref_offset)).getResult();
        auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value data = rewriter.create<LLVM::GEPOp>(loc, ptr, memref_ty.getElementType(),
                                                  desc->alignedPtr(rewriter, loc), offset);
        memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, data, 2);
    }

    return memref;
}

// Convert EncodedMemRef back to llvm MemRef descriptor, e.g.,
//   !llvm.struct<(i8, i8, ptr, array<2 x i64>)>
//     --->>> (note that memref descriptor still uses typed LLVM pointers)
//   !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
Value DecodeMemref(Location loc, PatternRewriter &rewriter, Type type, Type converted,
                   LLVM::AllocaOp alloca)
{
    auto ctx = rewriter.getContext();
    auto memref_type = cast<MemRefType>(type);
    auto memref_desc = MemRefDescriptor::undef(rewriter, loc, converted);
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    // Encode sizes together with strides as a single array.
    int64_t sizes_and_strides_size = 2 * memref_type.getRank();

    // Encoded memref type: !llvm.struct<(i8, i8, ptr<i8>, array<... x i64>)>.
    Type i8 = rewriter.getI8Type();
    Type arr = LLVM::LLVMArrayType::get(rewriter.getI64Type(), sizes_and_strides_size);
    auto encoded = LLVM::LLVMStructType::getLiteral(ctx, {i8, i8, ptr, arr});

    Value c0 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    Value c2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(2));

    // Fill memref descriptor pointers and offset.
    Value gep = rewriter.create<LLVM::GEPOp>(loc, ptr, encoded, alloca, ValueRange({c0, c2}));
    Value data_ptr = rewriter.create<LLVM::LoadOp>(loc, ptr, gep);
    memref_desc.setAllocatedPtr(rewriter, loc, data_ptr);
    memref_desc.setAlignedPtr(rewriter, loc, data_ptr);

    // Get the statically known strides and offset from the memref type.
    SmallVector<int64_t> strides;
    int64_t memref_offset;
    getStridesAndOffset(memref_type, strides, memref_offset);
    memref_desc.setConstantOffset(rewriter, loc, memref_offset);

    // Fill memref descriptor dimensions and strides.
    for (unsigned i = 0; i < memref_type.getRank(); ++i) {
        memref_desc.setConstantSize(rewriter, loc, i, memref_type.getDimSize(i));
        memref_desc.setConstantStride(rewriter, loc, i, strides[i]);
    }

    auto casted = rewriter.create<UnrealizedConversionCastOp>(loc, memref_type, Value(memref_desc));
    return casted.getResult(0);
}

struct CustomCallOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = op.getContext();
        Location loc = op.getLoc();
        // Create function
        Type ptr = LLVM::LLVMPointerType::get(ctx);

        FunctionType type = FunctionType::get(ctx, {/*args=*/ptr, /*rets=*/ptr}, {});
        auto customCallFnOp = rewriter.create<func::FuncOp>(loc, op.getCallTargetName(), type);
        customCallFnOp.setPrivate();

        // Setup args and res
        int32_t numberArg = op.getNumberOriginalArgAttr()[0];
        SmallVector<Value> operands = op.getOperands();
        SmallVector<Value> args = {operands.begin(), operands.end() - numberArg};
        SmallVector<Value> res = {operands.begin() + numberArg, operands.end()};

        SmallVector<Value> operandsConverted = adaptor.getOperands();
        SmallVector<Value> argsConverted = {operandsConverted.begin(),
                                            operandsConverted.end() - numberArg};
        SmallVector<Value> resConverted = {operandsConverted.begin() + numberArg,
                                           operandsConverted.end()};

        // Encode args
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        SmallVector<mlir::LLVM::AllocaOp> encoded;
        for (auto tuple : llvm::drop_begin(llvm::zip(args, argsConverted))) {
            auto memref_type = std::get<0>(tuple).getType().cast<MemRefType>();
            auto encoded_arg = EncodeMemRef(loc, rewriter, memref_type, std::get<1>(tuple));
            LLVM::AllocaOp alloca = rewriter.create<LLVM::AllocaOp>(loc, ptr, type, c1, 0);
            // Start the lifetime of encoded value.
            rewriter.create<LLVM::LifetimeStartOp>(loc, rewriter.getI64IntegerAttr(-1), alloca);
            // Use volatile store to suppress expensive LLVM optimizations.
            rewriter.create<LLVM::StoreOp>(loc, encoded_arg, alloca, /*alignment=*/0,
                                           /*isVolatile=*/true);
            encoded.push_back(alloca);
        }

        // We store encoded arguments as `!llvm.array<ptr x len>`.
        size_t len = encoded.size();
        Type typeArgs = LLVM::LLVMArrayType::get(ptr, len);

        // Prepare an array for encoded arguments.
        Value arrArgs = rewriter.create<LLVM::UndefOp>(loc, typeArgs);
        auto insertValueArgs = [&](Value value, int64_t offset) {
            arrArgs = rewriter.create<LLVM::InsertValueOp>(loc, arrArgs, value, offset);
        };
        // Store pointer to encoded arguments into the allocated storage.
        for (const auto &pair : llvm::enumerate(encoded)) {
            int64_t offset = pair.index();
            insertValueArgs(pair.value(), offset);
        }
        // Get allocation for packed arguments pointers.
        LLVM::AllocaOp alloca = rewriter.create<LLVM::AllocaOp>(loc, ptr, type, c1, 0);

        // Start the lifetime of the encoded arguments pointers.
        rewriter.create<LLVM::LifetimeStartOp>(loc, rewriter.getI64IntegerAttr(-1), alloca);

        // Store constructed arguments pointers array into the alloca. Use volatile
        // store to suppress expensive LLVM optimizations.
        rewriter.create<LLVM::StoreOp>(loc, arrArgs, alloca, /*alignment=*/0, /*isVolatile=*/true);

        // Alloca that encodes the custom call arguments.
        auto encodedArguments = alloca.getResult();

        // Results: ########

        // Encode all returns as a set of pointers
        SmallVector<mlir::LLVM::AllocaOp> encodedRes;
        for (auto tuple : llvm::zip(res, resConverted)) {
            auto memref_type = std::get<0>(tuple).getType().cast<MemRefType>();
            auto encoded_res = EncodeMemRef(loc, rewriter, memref_type, std::get<1>(tuple));
            LLVM::AllocaOp alloca = rewriter.create<LLVM::AllocaOp>(loc, ptr, type, c1, 0);
            // Start the lifetime of encoded value.
            rewriter.create<LLVM::LifetimeStartOp>(loc, rewriter.getI64IntegerAttr(-1), alloca);
            // Use volatile store to suppress expensive LLVM optimizations.
            rewriter.create<LLVM::StoreOp>(loc, encoded_res, alloca, /*alignment=*/0,
                                           /*isVolatile=*/true);
            encodedRes.push_back(alloca);
        }

        // We store encoded results as `!llvm.array<ptr x len>`.
        size_t lenRes = encodedRes.size();
        Type typeRes = LLVM::LLVMArrayType::get(ptr, lenRes);

        // Prepare an array for encoding results.
        Value arrRes = rewriter.create<LLVM::UndefOp>(loc, typeRes);
        auto insertValueRes = [&](Value value, int64_t offset) {
            arrRes = rewriter.create<LLVM::InsertValueOp>(loc, arrArgs, value, offset);
        };

        // Store encoded results into the allocated storage.
        for (const auto &pair : llvm::enumerate(encodedRes)) {
            int64_t offset = pair.index();
            insertValueRes(pair.value(), offset);
        }

        // Get allocation for packed results pointers.
        LLVM::AllocaOp allocaRes = rewriter.create<LLVM::AllocaOp>(loc, ptr, type, c1, 0);

        // Start the lifetime of the encoded results pointers allocation.
        rewriter.create<LLVM::LifetimeStartOp>(loc, rewriter.getI64IntegerAttr(-1), allocaRes);

        // Store constructed results pointers array on the stack. Use volatile
        // store to suppress expensive LLVM optimizations.
        rewriter.create<LLVM::StoreOp>(loc, arrRes, allocaRes, /*alignment=*/0, /*isVolatile=*/true);

        // Alloca that encodes the custom call returns.
        auto encodedResults = allocaRes;
        // Call op
        func::CallOp call = rewriter.create<func::CallOp>(
            loc, customCallFnOp, ValueRange{encodedArguments, encodedResults});

        // Decode res
        SmallVector<Value> decodedResults;
        for (auto tuple : llvm::zip(res, resConverted, encodedRes)) {
            auto decodedReturn = DecodeMemref(loc, rewriter, std::get<0>(tuple).getType(),
                                              std::get<1>(tuple).getType(), std::get<2>(tuple));
            decodedResults.push_back(decodedReturn);
        }
        auto size = rewriter.getI64IntegerAttr(-1);

        // // End the lifetime of encoded arguments and results pointers.
        // if (auto *alloca = std::get_if<LLVM::AllocaOp>(&args->encoded))
        //     b.create<LLVM::LifetimeEndOp>(size, *alloca);
        // if (auto *alloca = std::get_if<LLVM::AllocaOp>(&rets->encoded))
        //     b.create<LLVM::LifetimeEndOp>(size, *alloca);

        // // End the lifetime of arguments encoded on a stack.
        // for (auto &arg : args->values)
        //     if (auto *alloca = std::get_if<LLVM::AllocaOp>(&arg))
        //         b.create<LLVM::LifetimeEndOp>(size, *alloca);

        // // End the lifetime of results encoded on a stack.
        // for (LLVM::AllocaOp alloca : rets->allocas)
        //     b.create<LLVM::LifetimeEndOp>(size, alloca);

        rewriter.replaceOp(op, ValueRange(decodedResults));
        return success();
    }
};

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_CATALYSTCONVERSIONPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct CatalystConversionPass : impl::CatalystConversionPassBase<CatalystConversionPass> {
    using CatalystConversionPassBase::CatalystConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LLVMTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        patterns.add<CustomCallOpPattern>(typeConverter, context);
        patterns.add<PrintOpPattern>(typeConverter, context);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<CatalystDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createCatalystConversionPass()
{
    return std::make_unique<catalyst::CatalystConversionPass>();
}

} // namespace catalyst
