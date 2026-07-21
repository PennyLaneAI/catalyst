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
// CAPI (runtime/include/TransportCAPI.h). Controller-side only.

#include "llvm/ADT/Twine.h"
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

// sizeof(CatalystTransportPeerRef) = {u32, u64, u64}; over-allocate for alignment.
constexpr int64_t kPeerRefBytes = 32;

LLVM::LLVMPointerType ptrTy(MLIRContext *ctx) { return LLVM::LLVMPointerType::get(ctx); }

ModuleOp moduleOf(Operation *op) { return op->getParentOfType<ModuleOp>(); }

// Declare-or-reuse a CAPI function and emit a call to it. A null resultTy means
// the function returns void.
Value emitCall(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod, StringRef name,
               ArrayRef<Type> paramTys, Type resultTy, ValueRange args)
{
    Type rty = resultTy ? resultTy : LLVM::LLVMVoidType::get(rewriter.getContext());
    auto fn = LLVM::lookupOrCreateFn(rewriter, mod, name, paramTys, rty);
    assert(succeeded(fn) && "failed to declare transport CAPI function");
    auto call = LLVM::CallOp::create(rewriter, loc, *fn, args);
    return call.getNumResults() ? call.getResult() : Value();
}

// Materialize a null-terminated global string and return a ptr to its data.
Value globalStr(ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod, StringRef prefix,
                StringRef value)
{
    static int counter = 0;
    std::string symName = (prefix + Twine(counter++)).str();
    return LLVM::createGlobalString(loc, rewriter, symName, Twine(value).concat(Twine('\0')).str(),
                                    LLVM::Linkage::Internal);
}

Value constInt(ConversionPatternRewriter &rewriter, Location loc, Type ty, int64_t v)
{
    return LLVM::ConstantOp::create(rewriter, loc, ty, rewriter.getIntegerAttr(ty, v));
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct ControllerCreateLowering : public OpConversionPattern<ControllerCreateOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ControllerCreateOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value lib = globalStr(rewriter, op.getLoc(), mod, "transport_backend_", op.getBackendLib());
        Value cfg = globalStr(rewriter, op.getLoc(), mod, "transport_config_", op.getConfig());
        Value s = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__controller_create",
                           {ptrTy(ctx), ptrTy(ctx)}, ptrTy(ctx), {lib, cfg});
        rewriter.replaceOp(op, s);
        return success();
    }
};

struct ConnectLowering : public OpConversionPattern<ConnectOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ConnectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value peer = globalStr(rewriter, op.getLoc(), mod, "transport_peer_", op.getPeer());
        Value port = constInt(rewriter, op.getLoc(), rewriter.getI16Type(), op.getOobPort());
        Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__connect",
                           {ptrTy(ctx), ptrTy(ctx), rewriter.getI16Type()}, rewriter.getI32Type(),
                           {adaptor.getSession(), peer, port});
        rewriter.replaceOp(op, r);
        return success();
    }
};

struct ExchangeKeysLowering : public OpConversionPattern<ExchangeKeysOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ExchangeKeysOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value peerBuf = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), ptrTy(ctx), rewriter.getI8Type(),
            constInt(rewriter, op.getLoc(), rewriter.getI64Type(), kPeerRefBytes), /*alignment=*/8);
        Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__exchange_keys",
                           {ptrTy(ctx), ptrTy(ctx)}, rewriter.getI32Type(),
                           {adaptor.getSession(), peerBuf});
        rewriter.replaceOp(op, {r, peerBuf});
        return success();
    }
};

struct EstablishChannelLowering : public OpConversionPattern<EstablishChannelOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(EstablishChannelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value dp = constInt(rewriter, op.getLoc(), rewriter.getI32Type(), op.getDataPath());
        Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__establish_channel",
                           {ptrTy(ctx), rewriter.getI32Type(), ptrTy(ctx)}, rewriter.getI32Type(),
                           {adaptor.getSession(), dp, adaptor.getPeer()});
        rewriter.replaceOp(op, r);
        return success();
    }
};

struct CommitWorkItemLowering : public OpConversionPattern<CommitWorkItemOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CommitWorkItemOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value idx = constInt(rewriter, op.getLoc(), rewriter.getI32Type(), op.getWorkItemIdx());
        Value inB = constInt(rewriter, op.getLoc(), rewriter.getI64Type(), op.getInBytes());
        Value outB = constInt(rewriter, op.getLoc(), rewriter.getI64Type(), op.getOutBytes());
        Value r = emitCall(
            rewriter, op.getLoc(), mod, "__catalyst__transport__commit_work_item",
            {ptrTy(ctx), rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI64Type()},
            rewriter.getI32Type(), {adaptor.getSession(), idx, inB, outB});
        rewriter.replaceOp(op, r);
        return success();
    }
};

// Void-returning single-session ops: start / stop / close / destroy.
template <typename OpT> struct VoidSessionLowering : public OpConversionPattern<OpT> {
    VoidSessionLowering(const TypeConverter &tc, MLIRContext *ctx, StringRef sym)
        : OpConversionPattern<OpT>(tc, ctx), symbol(sym)
    {
    }
    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        emitCall(rewriter, op.getLoc(), op->template getParentOfType<ModuleOp>(), symbol,
                 {ptrTy(op.getContext())}, Type(), {adaptor.getSession()});
        rewriter.eraseOp(op);
        return success();
    }
    std::string symbol;
};

struct KickLowering : public OpConversionPattern<KickOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(KickOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        // slot = data_slot(s); store payload -> slot; kick(s, idx)
        Value slot = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__data_slot",
                              {ptrTy(ctx)}, ptrTy(ctx), {adaptor.getSession()});
        LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getPayload(), slot);
        Value idx = constInt(rewriter, op.getLoc(), rewriter.getI32Type(), op.getWorkItemIdx());
        Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__kick",
                           {ptrTy(ctx), rewriter.getI32Type()}, rewriter.getI32Type(),
                           {adaptor.getSession(), idx});
        rewriter.replaceOp(op, r);
        return success();
    }
};

struct CollectLowering : public OpConversionPattern<CollectOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CollectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value one = constInt(rewriter, op.getLoc(), rewriter.getI64Type(), 1);
        Value buf = LLVM::AllocaOp::create(rewriter, op.getLoc(), ptrTy(ctx), rewriter.getI64Type(),
                                           one, /*alignment=*/8);
        Value bytes = constInt(rewriter, op.getLoc(), rewriter.getI64Type(), op.getBytes());
        emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__collect",
                 {ptrTy(ctx), ptrTy(ctx), rewriter.getI64Type()}, rewriter.getI32Type(),
                 {adaptor.getSession(), buf, bytes});
        Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), rewriter.getI64Type(), buf);
        rewriter.replaceOp(op, loaded);
        return success();
    }
};

struct LastRttLowering : public OpConversionPattern<LastRttNsOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LastRttNsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Value r =
            emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__last_rtt_ns",
                     {ptrTy(op.getContext())}, rewriter.getI64Type(), {adaptor.getSession()});
        rewriter.replaceOp(op, r);
        return success();
    }
};

} // namespace

struct ConvertTransportToLLVMPass
    : public impl::ConvertTransportToLLVMPassBase<ConvertTransportToLLVMPass> {
    using ConvertTransportToLLVMPassBase::ConvertTransportToLLVMPassBase;

    void runOnOperation() override
    {
        MLIRContext *ctx = &getContext();
        LLVMTypeConverter tc(ctx);
        tc.addConversion([ctx](SessionType) -> Type { return LLVM::LLVMPointerType::get(ctx); });
        tc.addConversion([ctx](PeerType) -> Type { return LLVM::LLVMPointerType::get(ctx); });

        RewritePatternSet patterns(ctx);
        patterns.add<ControllerCreateLowering, ConnectLowering, ExchangeKeysLowering,
                     EstablishChannelLowering, CommitWorkItemLowering, KickLowering,
                     CollectLowering, LastRttLowering>(tc, ctx);
        patterns.add<VoidSessionLowering<StartOp>>(tc, ctx, "__catalyst__transport__start");
        patterns.add<VoidSessionLowering<StopOp>>(tc, ctx, "__catalyst__transport__stop");
        patterns.add<VoidSessionLowering<CloseOp>>(tc, ctx, "__catalyst__transport__close");
        patterns.add<VoidSessionLowering<DestroyOp>>(tc, ctx, "__catalyst__transport__destroy");

        ConversionTarget target(*ctx);
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<TransportDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace transport
} // namespace catalyst
