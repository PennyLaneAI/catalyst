// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "RTIO/IR/RTIOOps.h"
#include "RTIO/Transforms/Patterns.h"

#include "ARTIQRuntimeBuilder.hpp"

using namespace mlir;
using namespace catalyst::rtio;

namespace {

//===----------------------------------------------------------------------===//
// RPC helpers
//===----------------------------------------------------------------------===//

/// Map an MLIR type to its ARTIQ RPC wire-protocol tag character.
///   i32             -> 'i'
///   i64             -> 'I'
///   f32 / f64       -> 'f'
///   memref<...xi8>  -> 's'
///   everything else -> 'O'
static char tagCodeForType(Type ty) {
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
        if (intTy.getWidth() <= 32) {
            return 'i';
        }
        return 'I';
    }
    if (isa<Float64Type>(ty) || isa<Float32Type>(ty))
        return 'f';
    if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
        if (memrefTy.getElementType().isInteger(8))
            return 's';
    }
    return 'O';
}

/// Build the ARTIQ tag string from operand and result MLIR types.
/// ARTIQ wire format is  <args>:<return>
///
/// When `numKeywordArgs > 0`, the last N args are keywords and prefixed with 'k' in the tag.
/// Example:
///   positional (ptr) + keywords (str, i32, i32) -> "Okskiki:n"
static std::string buildTagFromTypes(TypeRange argTypes, TypeRange resultTypes,
                                     size_t numKeywordArgs = 0) {
    std::string tag;
    size_t numPositional = argTypes.size() - numKeywordArgs;
    for (size_t i = 0; i < argTypes.size(); i++) {
        if (i >= numPositional) {
            tag += 'k';
        }
        tag += tagCodeForType(argTypes[i]);
    }
    tag += ':';
    if (resultTypes.empty()) {
        tag += 'n';
    } else {
        tag += tagCodeForType(resultTypes[0]);
    }
    return tag;
}

/// Ensure an LLVM global constant for the given byte string exists in the module.
static Value getOrCreateStringGlobal(ConversionPatternRewriter &rewriter, ModuleOp module,
                                     Location loc, StringRef str) {
    MLIRContext *ctx = rewriter.getContext();

    std::string globalName = "__rtio_str_" + str.str();
    for (char &c : globalName) {
        if (!llvm::isAlnum(c) && c != '_') {
            c = '_';
        }
    }

    auto i8Ty = IntegerType::get(ctx, 8);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    LLVM::GlobalOp globalOp;
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
        globalOp = existing;
    } else {
        SmallVector<uint8_t> bytes(str.begin(), str.end());

        auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, bytes.size());
        auto dataAttrType = RankedTensorType::get({(int64_t)bytes.size()}, i8Ty);
        auto dataAttr = DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(bytes));

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        globalOp = LLVM::GlobalOp::create(rewriter, loc, arrayTy, /*isConstant=*/true,
                                          LLVM::Linkage::Private, globalName, dataAttr);
    }

    auto arrayTy = cast<LLVM::LLVMArrayType>(globalOp.getType());
    Value addr = LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalOp.getName());
    SmallVector<LLVM::GEPArg> indices = {0, 0};
    return LLVM::GEPOp::create(rewriter, loc, ptrTy, arrayTy, addr, indices);
}

/// Alloca a `{ptr, i32}` fat-pointer and fill it with {ptr, (value)len}.
static Value buildFatPointer(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                             Value dataPtr, Value len) {
    Type ptrTy = LLVM::LLVMPointerType::get(ctx);
    Type i32Ty = IntegerType::get(ctx, 32);
    auto fatPtrTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i32Ty});

    Value one = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    Value slot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, fatPtrTy, one);

    LLVM::StoreOp::create(rewriter, loc, dataPtr, slot);
    SmallVector<LLVM::GEPArg> lenIdx = {0, 1};
    Value lenField = LLVM::GEPOp::create(rewriter, loc, ptrTy, fatPtrTy, slot, lenIdx);
    LLVM::StoreOp::create(rewriter, loc, len, lenField);

    return slot;
}

/// Helper to get a fat-pointer from {ptr, (int)len}.
static Value buildTagStruct(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                            Value tagDataPtr, int32_t tagLen) {
    Value lenVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(tagLen));
    return buildFatPointer(rewriter, loc, ctx, tagDataPtr, lenVal);
}

/// Return the LLVM type for a given tag code.
static Type llvmTypeForTagCode(char code, MLIRContext *ctx) {
    Type ptrTy = LLVM::LLVMPointerType::get(ctx);
    Type i32Ty = IntegerType::get(ctx, 32);
    switch (code) {
    case 'i':
        return i32Ty;
    case 'I':
        return IntegerType::get(ctx, 64);
    case 'f':
        return Float64Type::get(ctx);
    case 's':
        return LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i32Ty});
    case 'O':
    default:
        return ptrTy;
    }
}

/// Extract {aligned_ptr, length_i32} from a converted `memref<?xi8>`.
/// Layout: `{allocPtr, alignPtr, offset, sizes[1], strides[1]}`.
static std::pair<Value, Value> extractStringFromMemref(ConversionPatternRewriter &rewriter,
                                                       Location loc, Value memrefDesc) {
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i64Ty = IntegerType::get(rewriter.getContext(), 64);
    Type i32Ty = IntegerType::get(rewriter.getContext(), 32);

    Value alignedPtr =
        LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memrefDesc, ArrayRef<int64_t>{1});
    Value size64 =
        LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, memrefDesc, ArrayRef<int64_t>{3, 0});
    Value size32 = arith::TruncIOp::create(rewriter, loc, i32Ty, size64);
    return {alignedPtr, size32};
}

/// Box a normal argument into a stack slot.
static Value boxNormalArg(ConversionPatternRewriter &rewriter, Location loc, Value arg, Value one) {
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type argTy = arg.getType();
    Value slot = isa<LLVM::LLVMPointerType>(argTy)
                     ? LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, one)
                     : LLVM::AllocaOp::create(rewriter, loc, ptrTy, argTy, one);
    LLVM::StoreOp::create(rewriter, loc, arg, slot);
    return slot;
}

/// Box a string argument (tag 's').
static Value boxStringArg(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                          Value convertedArg) {
    auto [dataPtr, len] = extractStringFromMemref(rewriter, loc, convertedArg);
    return buildFatPointer(rewriter, loc, ctx, dataPtr, len);
}

/// Box a keyword argument (tag 'k' + valueCode).
///
/// Layout for `ki`:  `{ {ptr,i32}, i32 }`       (keyword name + i32 value)
/// Layout for `ks`:  `{ {ptr,i32}, {ptr,i32} }` (keyword name + string value)
static Value boxKeywordArg(ConversionPatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                           ModuleOp module, StringRef kwName, Value convertedArg, char valueCode,
                           Value one) {
    Type ptrTy = LLVM::LLVMPointerType::get(ctx);
    Type i32Ty = IntegerType::get(ctx, 32);
    auto fatPtrTy = LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i32Ty});

    Type valueTy = llvmTypeForTagCode(valueCode, ctx);
    auto kwStructTy = LLVM::LLVMStructType::getLiteral(ctx, {fatPtrTy, valueTy});

    Value kwSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, kwStructTy, one);

    Value kwNameGlobal = getOrCreateStringGlobal(rewriter, module, loc, kwName);
    LLVM::StoreOp::create(rewriter, loc, kwNameGlobal, kwSlot);

    SmallVector<LLVM::GEPArg> kwLenIdx = {0, 0, 1};
    Value kwLenField = LLVM::GEPOp::create(rewriter, loc, ptrTy, kwStructTy, kwSlot, kwLenIdx);
    Value kwLenVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(kwName.size()));
    LLVM::StoreOp::create(rewriter, loc, kwLenVal, kwLenField);

    SmallVector<LLVM::GEPArg> valIdx = {0, 1};
    Value valField = LLVM::GEPOp::create(rewriter, loc, ptrTy, kwStructTy, kwSlot, valIdx);

    if (valueCode == 's') {
        auto [dataPtr, strLen] = extractStringFromMemref(rewriter, loc, convertedArg);
        LLVM::StoreOp::create(rewriter, loc, dataPtr, valField);
        SmallVector<LLVM::GEPArg> strLenIdx = {0, 1, 1};
        Value strLenField =
            LLVM::GEPOp::create(rewriter, loc, ptrTy, kwStructTy, kwSlot, strLenIdx);
        LLVM::StoreOp::create(rewriter, loc, strLen, strLenField);
    } else {
        LLVM::StoreOp::create(rewriter, loc, convertedArg, valField);
    }

    return kwSlot;
}

/// Get slot size in bytes for ARTIQ RPC return type code.
static int getSlotSizeForReturnCode(char code) {
    switch (code) {
    case 'i':
        return 4;
    case 'n':
    case 'I':
    case 'f':
    case 's':
    case 'O':
        return 8;
    default:
        llvm_unreachable("unknown ARTIQ RPC return type code");
    }
}

//===----------------------------------------------------------------------===//
// RPC Conversion Pattern
//===----------------------------------------------------------------------===//

/// Lower `rtio.rpc` to ARTIQ `rpc_send` / `rpc_recv` calls.
///
/// Tag format: `<args> : <return>`
///
/// Argument boxing by tag code:
///   'i'/'I'/'f'/'O'  - normal: alloca + store into ptr array
///   's'              - string: extract ptr+len from memref, box as {ptr,i32}
///   'k'+code         - keyword: build `{ {ptr,i32}, <value> }` struct
///
/// Sync calls include a recv drain loop:
///   size = rpc_recv(slot);
///   while (size != 0) { buf = alloca(size); size = rpc_recv(buf); }
struct RPCOpLowering : public OpConversionPattern<RTIORPCOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIORPCOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto rpcIdAttr = op->getAttrOfType<IntegerAttr>("rpc_id");

        // It should be assigned by the RPC ID assignment sub-pass.
        assert(rpcIdAttr && "rtio.rpc is missing rpc_id attribute");

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        ModuleOp module = op->getParentOfType<ModuleOp>();

        Type i8Ty = IntegerType::get(ctx, 8);
        Type i32Ty = IntegerType::get(ctx, 32);
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);

        ARTIQRuntimeBuilder artiq(rewriter, op);

        auto kwNamesAttr = op->getAttrOfType<ArrayAttr>("keyword_names");
        size_t numKw = kwNamesAttr ? kwNamesAttr.size() : 0;

        // 1. Derive tag string (<args>:<return>) and build {ptr, i32} struct
        std::string tag = buildTagFromTypes(op.getArgs().getTypes(), op.getResultTypes(), numKw);
        Value tagDataPtr = getOrCreateStringGlobal(rewriter, module, loc, tag);
        rewriter.setInsertionPoint(op);
        Value tagStruct = buildTagStruct(rewriter, loc, ctx, tagDataPtr, tag.size());

        // 2. Build args array
        ValueRange convertedArgs = adaptor.getArgs();
        TypeRange origArgTypes = op.getArgs().getTypes();
        size_t numArgs = convertedArgs.size();
        size_t numPositional = numArgs - numKw;

        Value one = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));

        auto ptrArrayTy = LLVM::LLVMArrayType::get(ptrTy, std::max(numArgs, (size_t)1));
        Value argsArray = LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrArrayTy, one);

        for (size_t i = 0; i < numArgs; i++) {
            Value arg = convertedArgs[i];
            char code = tagCodeForType(origArgTypes[i]);

            Value argSlot;
            if (i >= numPositional) {
                StringRef kwName = cast<StringAttr>(kwNamesAttr[i - numPositional]).getValue();
                argSlot = boxKeywordArg(rewriter, loc, ctx, module, kwName, arg, code, one);
            } else if (code == 's') {
                argSlot = boxStringArg(rewriter, loc, ctx, arg);
            } else {
                argSlot = boxNormalArg(rewriter, loc, arg, one);
            }

            SmallVector<LLVM::GEPArg> idxs = {0, static_cast<int32_t>(i)};
            Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrArrayTy, argsArray, idxs);
            LLVM::StoreOp::create(rewriter, loc, argSlot, elemPtr);
        }

        // 3. Call rpc_send / rpc_send_async
        Value serviceId = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(i32Ty, rpcIdAttr.getInt()));

        SmallVector<Value> replacementValues;

        if (op.getIsAsync()) {
            artiq.rpcSendAsync(serviceId, tagStruct, argsArray);
        } else {
            artiq.rpcSend(serviceId, tagStruct, argsArray);

            // 4. recv loop: drain chunks until rpc_recv returns 0
            char retCode = tag.back();
            int slotSize = getSlotSizeForReturnCode(retCode);
            auto slotTy = LLVM::LLVMArrayType::get(i8Ty, slotSize);
            Value resultSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, slotTy, one);

            Value initSize = artiq.rpcRecv(resultSlot);
            Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));

            auto whileOp = scf::WhileOp::create(rewriter, loc, /*resultTypes=*/TypeRange{i32Ty},
                                                /*operands=*/ValueRange{initSize});

            {
                Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, {i32Ty}, {loc});
                Value sz = before->getArgument(0);
                Value cond =
                    arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, sz, zero);
                scf::ConditionOp::create(rewriter, loc, cond, ValueRange{sz});
            }

            {
                Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, {i32Ty}, {loc});
                Value chunkSize = after->getArgument(0);
                Value buf = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i8Ty, chunkSize);
                Value nextSize = artiq.rpcRecv(buf);
                scf::YieldOp::create(rewriter, loc, ValueRange{nextSize});
            }

            rewriter.setInsertionPointAfter(whileOp);

            if (retCode != 'n' && op.getNumResults() == 1) {
                Type resultTy = getTypeConverter()->convertType(op.getResult(0).getType());
                Value loaded = LLVM::LoadOp::create(rewriter, loc, resultTy, resultSlot);
                replacementValues.push_back(loaded);
            }
        }

        rewriter.replaceOp(op, replacementValues);
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace rtio {

void populateRTIORPCConversionPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
    patterns.add<RPCOpLowering>(typeConverter, patterns.getContext());
}

} // namespace rtio
} // namespace catalyst
