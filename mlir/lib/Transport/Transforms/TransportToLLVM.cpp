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

// Lower the `transport` dialect to `llvm.call`s on the __catalyst__transport__*
// CAPI (runtime/include/TransportCAPI.h).

#include "llvm/ADT/StringExtras.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Transport/IR/TransportOps.h"
#include "Transport/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::transport;

namespace catalyst {
namespace transport {

#define GEN_PASS_DEF_CONVERTTRANSPORTTOLLVMPASS
#include "Transport/Transforms/Passes.h.inc"

namespace {

LLVM::LLVMPointerType ptrTy(MLIRContext *ctx) { return LLVM::LLVMPointerType::get(ctx); }
IntegerType i32Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 32); }
IntegerType i64Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 64); }

ModuleOp moduleOf(Operation *op) { return op->getParentOfType<ModuleOp>(); }

Value emitCall(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod, StringRef name,
               ArrayRef<Type> paramTys, Type resultTy, ValueRange args) {
    Type rty = resultTy ? resultTy : LLVM::LLVMVoidType::get(rewriter.getContext());
    auto fn = LLVM::lookupOrCreateFn(rewriter, mod, name, paramTys, rty);
    assert(succeeded(fn) && "failed to declare transport CAPI function");
    auto call = LLVM::CallOp::create(rewriter, loc, *fn, args);
    return call.getNumResults() ? call.getResult() : Value();
}

std::string globalStrKey(StringRef prefix, StringRef value) {
    std::string key = prefix.str();
    for (char c : value) {
        bool ok = llvm::isAlnum(c) || c == '_';
        key.push_back(ok ? c : '_');
    }
    return key;
}

Value globalStr(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod, StringRef prefix,
                StringRef value) {
    std::string data = value.str();
    data.push_back('\0');
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), data.size());
    StringAttr dataAttr = rewriter.getStringAttr(data);

    // Since the key is lossy, an existing global holding something else is not ours to reuse.
    std::string base = globalStrKey(prefix, value);
    std::string symName = base;
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(symName);
    for (unsigned n = 0; glb && glb.getValueOrNull() != dataAttr; ++n) {
        symName = base + "." + std::to_string(n);
        glb = mod.lookupSymbol<LLVM::GlobalOp>(symName);
    }

    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = LLVM::GlobalOp::create(rewriter, loc, type, /*isConstant=*/true,
                                     LLVM::Linkage::Internal, symName, dataAttr);
    }
    return LLVM::GEPOp::create(rewriter, loc, ptrTy(rewriter.getContext()), type,
                               LLVM::AddressOfOp::create(rewriter, loc, glb),
                               ArrayRef<LLVM::GEPArg>{0, 0}, LLVM::GEPNoWrapFlags::inbounds);
}

void emitCheckedStatus(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod,
                       StringRef name, ArrayRef<Type> paramTys, ValueRange args, StringRef what) {
    auto *ctx = rewriter.getContext();
    Value rc = emitCall(rewriter, loc, mod, name, paramTys, i32Ty(ctx), args);
    Value msg = globalStr(rewriter, loc, mod, "transport_check_", what);
    emitCall(rewriter, loc, mod, "__catalyst__transport__check", {i32Ty(ctx), ptrTy(ctx)}, Type(),
             {rc, msg});
}

Value emitCheckedSession(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod,
                         StringRef name, ArrayRef<Type> paramTys, ValueRange args, StringRef what) {
    auto *ctx = rewriter.getContext();
    Value s = emitCall(rewriter, loc, mod, name, paramTys, ptrTy(ctx), args);
    Value msg = globalStr(rewriter, loc, mod, "transport_session_", what);
    emitCall(rewriter, loc, mod, "__catalyst__transport__check_session", {ptrTy(ctx), ptrTy(ctx)},
             Type(), {s, msg});
    return s;
}

Value constInt(ConversionPatternRewriter &rewriter, Location loc, Type ty, int64_t v) {
    return LLVM::ConstantOp::create(rewriter, loc, ty, rewriter.getIntegerAttr(ty, v));
}

// From a lowered 1-D memref descriptor (an LLVM struct), extract the aligned data
// pointer and the buffer's size in bytes (num elements * element byte width).
std::pair<Value, Value> memrefPtrAndBytes(ConversionPatternRewriter &rewriter, Location loc,
                                          Value descriptor, MemRefType memTy) {
    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, descriptor, ArrayRef<int64_t>{1});
    Value nelem = LLVM::ExtractValueOp::create(rewriter, loc, descriptor, ArrayRef<int64_t>{3, 0});
    Type elemTy = memTy.getElementType();
    int64_t elemBytes = isa<IndexType>(elemTy) ? 8 : (elemTy.getIntOrFloatBitWidth() + 7) / 8;
    Value ebytes = constInt(rewriter, loc, i64Ty(rewriter.getContext()), elemBytes);
    Value bytes = LLVM::MulOp::create(rewriter, loc, nelem, ebytes);
    return {ptr, bytes};
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct CreateLowering : public OpConversionPattern<CreateOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CreateOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        auto sessTy = cast<SessionType>(op.getSession().getType());
        Value lib = globalStr(rewriter, op.getLoc(), mod, "transport_backend_", op.getBackendLib());
        Value cfg = globalStr(rewriter, op.getLoc(), mod, "transport_config_", op.getConfig());
        Value key = globalStr(rewriter, op.getLoc(), mod, "transport_key_", op.getKey());
        Value role =
            constInt(rewriter, op.getLoc(), i32Ty(ctx), static_cast<int64_t>(sessTy.getRole()));
        Value s = emitCheckedSession(rewriter, op.getLoc(), mod, "__catalyst__transport__create",
                                     {ptrTy(ctx), ptrTy(ctx), i32Ty(ctx), ptrTy(ctx)},
                                     {lib, cfg, role, key}, "create");
        rewriter.replaceOp(op, s);
        return success();
    }
};

template <typename OpT, bool Async> struct ConnectLoweringBase : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();
        // peer / oob_port are optional: memcpy pairs by session key, so its transport.connect
        // carries neither. Absent attrs lower to a null pointer + 0, which the CAPI reads as
        // an empty peer string (TransportCAPI.cpp: `peer ? peer : ""`).
        auto peerAttr = op.getPeerAttr();
        Value peer =
            peerAttr ? globalStr(rewriter, op.getLoc(), mod, "transport_peer_", peerAttr.getValue())
                     : Value(LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy(ctx)));
        Value port =
            constInt(rewriter, op.getLoc(), IntegerType::get(ctx, 16), op.getOobPort().value_or(0));
        if (Async) {
            Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__connect_async",
                               {ptrTy(ctx), ptrTy(ctx), IntegerType::get(ctx, 16)}, i64Ty(ctx),
                               {adaptor.getSession(), peer, port});
            rewriter.replaceOp(op, r);
        } else {
            emitCheckedStatus(rewriter, op.getLoc(), mod, "__catalyst__transport__connect",
                              {ptrTy(ctx), ptrTy(ctx), IntegerType::get(ctx, 16)},
                              {adaptor.getSession(), peer, port}, "connect");
            rewriter.eraseOp(op);
        }
        return success();
    }
};
using ConnectLowering = ConnectLoweringBase<ConnectOp, false>;
using ConnectAsyncLowering = ConnectLoweringBase<ConnectAsyncOp, true>;

template <typename OpT, bool Async>
struct ExchangeKeysLoweringBase : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();
        if (Async) {
            Value r =
                emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__exchange_keys_async",
                         {ptrTy(ctx)}, i64Ty(ctx), {adaptor.getSession()});
            rewriter.replaceOp(op, r);
        } else {
            emitCheckedStatus(rewriter, op.getLoc(), mod, "__catalyst__transport__exchange_keys",
                              {ptrTy(ctx)}, {adaptor.getSession()}, "exchange_keys");
            rewriter.eraseOp(op);
        }
        return success();
    }
};
using ExchangeKeysLowering = ExchangeKeysLoweringBase<ExchangeKeysOp, false>;
using ExchangeKeysAsyncLowering = ExchangeKeysLoweringBase<ExchangeKeysAsyncOp, true>;

struct AwaitLowering : public OpConversionPattern<AwaitOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(AwaitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        emitCheckedStatus(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__await",
                          {i64Ty(ctx)}, {adaptor.getToken()}, "await");
        rewriter.eraseOp(op);
        return success();
    }
};

struct EstablishChannelLowering : public OpConversionPattern<EstablishChannelOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(EstablishChannelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        Value transport =
            globalStr(rewriter, op.getLoc(), moduleOf(op), "transport_kind_", op.getTransport());
        emitCheckedStatus(rewriter, op.getLoc(), moduleOf(op),
                          "__catalyst__transport__establish_channel", {ptrTy(ctx), ptrTy(ctx)},
                          {adaptor.getSession(), transport}, "establish_channel");
        rewriter.eraseOp(op);
        return success();
    }
};

struct SetCoprocessorFnLowering : public OpConversionPattern<SetCoprocessorFnOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(SetCoprocessorFnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value sym = globalStr(rewriter, op.getLoc(), mod, "transport_coproc_fn_", op.getSymbol());
        emitCheckedStatus(rewriter, op.getLoc(), mod, "__catalyst__transport__set_coprocessor_fn",
                          {ptrTy(ctx), ptrTy(ctx)}, {adaptor.getSession(), sym},
                          "set_coprocessor_fn");
        rewriter.eraseOp(op);
        return success();
    }
};

struct SetMessageSizesLowering : public OpConversionPattern<SetMessageSizesOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(SetMessageSizesOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        Value idx = constInt(rewriter, op.getLoc(), i32Ty(ctx), op.getWorkItemIdx());
        Value inB = constInt(rewriter, op.getLoc(), i64Ty(ctx), op.getInBytes());
        Value outB = constInt(rewriter, op.getLoc(), i64Ty(ctx), op.getOutBytes());
        emitCheckedStatus(rewriter, op.getLoc(), moduleOf(op),
                          "__catalyst__transport__set_message_sizes",
                          {ptrTy(ctx), i32Ty(ctx), i64Ty(ctx), i64Ty(ctx)},
                          {adaptor.getSession(), idx, inB, outB}, "set_message_sizes");
        rewriter.eraseOp(op);
        return success();
    }
};

struct ReplySlotLowering : public OpConversionPattern<ReplySlotOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ReplySlotOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        Location loc = op.getLoc();
        auto memTy = cast<MemRefType>(op.getSlot().getType());
        if (!memTy.hasStaticShape()) {
            return rewriter.notifyMatchFailure(op, "reply_slot result needs a static shape");
        }
        if (!memTy.getLayout().isIdentity()) {
            return rewriter.notifyMatchFailure(op, "reply_slot result needs an identity layout");
        }
        Value ptr = emitCall(rewriter, loc, moduleOf(op), "__catalyst__transport__reply_slot",
                             {ptrTy(ctx)}, ptrTy(ctx), {adaptor.getSession()});
        Type descTy = getTypeConverter()->convertType(memTy);
        if (!descTy) {
            return rewriter.notifyMatchFailure(op, "cannot convert reply_slot result type");
        }
        Value zero = constInt(rewriter, loc, i64Ty(ctx), 0);
        Value one = constInt(rewriter, loc, i64Ty(ctx), 1);
        Value nelem = constInt(rewriter, loc, i64Ty(ctx), memTy.getShape()[0]);
        Value desc = LLVM::UndefOp::create(rewriter, loc, descTy);
        desc = LLVM::InsertValueOp::create(rewriter, loc, desc, ptr, ArrayRef<int64_t>{0});
        desc = LLVM::InsertValueOp::create(rewriter, loc, desc, ptr, ArrayRef<int64_t>{1});
        desc = LLVM::InsertValueOp::create(rewriter, loc, desc, zero, ArrayRef<int64_t>{2});
        desc = LLVM::InsertValueOp::create(rewriter, loc, desc, nelem, ArrayRef<int64_t>{3, 0});
        desc = LLVM::InsertValueOp::create(rewriter, loc, desc, one, ArrayRef<int64_t>{4, 0});
        rewriter.replaceOp(op, desc);
        return success();
    }
};

struct StagePayloadLowering : public OpConversionPattern<StagePayloadOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(StagePayloadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        auto memTy = dyn_cast<MemRefType>(op.getPayload().getType());
        if (!memTy) {
            return rewriter.notifyMatchFailure(op, "payload must be bufferized (memref)");
        }
        if (!memTy.getLayout().isIdentity()) {
            return rewriter.notifyMatchFailure(op, "payload must have identity layout");
        }
        auto [srcPtr, bytes] =
            memrefPtrAndBytes(rewriter, op.getLoc(), adaptor.getPayload(), memTy);
        Value decoderId = constInt(rewriter, op.getLoc(), i32Ty(ctx), op.getDecoderId());
        emitCheckedStatus(rewriter, op.getLoc(), moduleOf(op),
                          "__catalyst__transport__stage_payload",
                          {ptrTy(ctx), ptrTy(ctx), i64Ty(ctx), i32Ty(ctx)},
                          {adaptor.getSession(), srcPtr, bytes, decoderId}, "stage_payload");
        rewriter.eraseOp(op);
        return success();
    }
};

struct PostLowering : public OpConversionPattern<PostOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(PostOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        Value idx = constInt(rewriter, op.getLoc(), i32Ty(ctx), op.getWorkItemIdx());
        emitCheckedStatus(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__post",
                          {ptrTy(ctx), i32Ty(ctx)}, {adaptor.getSession(), idx}, "post");
        rewriter.eraseOp(op);
        return success();
    }
};

struct CollectLowering : public OpConversionPattern<CollectOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CollectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        if (!op.getDest()) {
            return rewriter.notifyMatchFailure(op,
                                               "collect must be bufferized (dest-passing form)");
        }
        auto memTy = cast<MemRefType>(op.getDest().getType());
        // The reply is written contiguously through the aligned pointer, ignoring the memref's
        // strides and offset, so a non-identity dest layout would scatter the bytes wrongly.
        if (!memTy.getLayout().isIdentity()) {
            return rewriter.notifyMatchFailure(op, "collect dest must have identity layout");
        }
        auto [dstPtr, bytes] = memrefPtrAndBytes(rewriter, op.getLoc(), adaptor.getDest(), memTy);
        emitCheckedStatus(rewriter, op.getLoc(), mod, "__catalyst__transport__collect",
                          {ptrTy(ctx), ptrTy(ctx), i64Ty(ctx)},
                          {adaptor.getSession(), dstPtr, bytes}, "collect");
        rewriter.eraseOp(op);
        return success();
    }
};

struct LastRttLowering : public OpConversionPattern<LastRttNsOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LastRttNsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Value r =
            emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__last_rtt_ns",
                     {ptrTy(op.getContext())}, i64Ty(op.getContext()), {adaptor.getSession()});
        rewriter.replaceOp(op, r);
        return success();
    }
};

// Void-returning single-session ops: start / stop / close / destroy.
template <typename OpT> struct VoidSessionLowering : public OpConversionPattern<OpT> {
    VoidSessionLowering(const TypeConverter &tc, MLIRContext *ctx, StringRef sym)
        : OpConversionPattern<OpT>(tc, ctx), symbol(sym) {}
    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        emitCall(rewriter, op.getLoc(), op->template getParentOfType<ModuleOp>(), symbol,
                 {ptrTy(op.getContext())}, Type(), {adaptor.getSession()});
        rewriter.eraseOp(op);
        return success();
    }
    std::string symbol;
};

// get_session: look the session up by role from the runtime registry (populated at create).
struct GetSessionLowering : public OpConversionPattern<GetSessionOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(GetSessionOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        auto sessTy = cast<SessionType>(op.getSession().getType());
        Value role =
            constInt(rewriter, op.getLoc(), i32Ty(ctx), static_cast<int64_t>(sessTy.getRole()));
        Value key = globalStr(rewriter, op.getLoc(), mod, "transport_key_", op.getKey());
        Value s =
            emitCheckedSession(rewriter, op.getLoc(), mod, "__catalyst__transport__get_session",
                               {i32Ty(ctx), ptrTy(ctx)}, {role, key}, "get_session");
        rewriter.replaceOp(op, s);
        return success();
    }
};

} // namespace

struct ConvertTransportToLLVMPass
    : public impl::ConvertTransportToLLVMPassBase<ConvertTransportToLLVMPass> {
    using ConvertTransportToLLVMPassBase::ConvertTransportToLLVMPassBase;

    void runOnOperation() override {
        MLIRContext *ctx = &getContext();
        LLVMTypeConverter tc(ctx);
        tc.addConversion([ctx](SessionType) -> Type { return LLVM::LLVMPointerType::get(ctx); });
        tc.addConversion([ctx](TokenType) -> Type { return IntegerType::get(ctx, 64); });

        RewritePatternSet patterns(ctx);
        patterns.add<ReplySlotLowering>(tc, ctx);
        patterns.add<CreateLowering, ConnectLowering, ConnectAsyncLowering, ExchangeKeysLowering,
                     ExchangeKeysAsyncLowering, AwaitLowering, EstablishChannelLowering,
                     SetCoprocessorFnLowering, SetMessageSizesLowering, StagePayloadLowering,
                     PostLowering, CollectLowering, LastRttLowering, GetSessionLowering>(tc, ctx);
        patterns.add<VoidSessionLowering<StartOp>>(tc, ctx, "__catalyst__transport__start");
        patterns.add<VoidSessionLowering<StopOp>>(tc, ctx, "__catalyst__transport__stop");
        patterns.add<VoidSessionLowering<DestroyOp>>(tc, ctx, "__catalyst__transport__destroy");

        ConversionTarget target(*ctx);
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<TransportDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace transport
} // namespace catalyst
