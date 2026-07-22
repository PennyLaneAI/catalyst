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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

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
               ArrayRef<Type> paramTys, Type resultTy, ValueRange args)
{
    Type rty = resultTy ? resultTy : LLVM::LLVMVoidType::get(rewriter.getContext());
    auto fn = LLVM::lookupOrCreateFn(rewriter, mod, name, paramTys, rty);
    assert(succeeded(fn) && "failed to declare transport CAPI function");
    auto call = LLVM::CallOp::create(rewriter, loc, *fn, args);
    return call.getNumResults() ? call.getResult() : Value();
}

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

struct CreateLowering : public OpConversionPattern<CreateOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CreateOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        auto sessTy = cast<SessionType>(op.getSession().getType());
        Value lib = globalStr(rewriter, op.getLoc(), mod, "transport_backend_", op.getBackendLib());
        Value cfg = globalStr(rewriter, op.getLoc(), mod, "transport_config_", op.getConfig());
        Value role =
            constInt(rewriter, op.getLoc(), i32Ty(ctx), static_cast<int64_t>(sessTy.getRole()));
        Value s = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__create",
                           {ptrTy(ctx), ptrTy(ctx), i32Ty(ctx)}, ptrTy(ctx), {lib, cfg, role});
        rewriter.replaceOp(op, s);
        return success();
    }
};

template <typename OpT, bool Async> struct ConnectLoweringBase : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();
        Value peer = globalStr(rewriter, op.getLoc(), mod, "transport_peer_", op.getPeer());
        Value port = constInt(rewriter, op.getLoc(), IntegerType::get(ctx, 16), op.getOobPort());
        if (Async) {
            Value r = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__connect_async",
                               {ptrTy(ctx), ptrTy(ctx), IntegerType::get(ctx, 16)}, i64Ty(ctx),
                               {adaptor.getSession(), peer, port});
            rewriter.replaceOp(op, r);
        }
        else {
            emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__connect",
                     {ptrTy(ctx), ptrTy(ctx), IntegerType::get(ctx, 16)}, i32Ty(ctx),
                     {adaptor.getSession(), peer, port});
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
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();
        if (Async) {
            Value r =
                emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__exchange_keys_async",
                         {ptrTy(ctx)}, i64Ty(ctx), {adaptor.getSession()});
            rewriter.replaceOp(op, r);
        }
        else {
            emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__exchange_keys",
                     {ptrTy(ctx)}, i32Ty(ctx), {adaptor.getSession()});
            rewriter.eraseOp(op);
        }
        return success();
    }
};
using ExchangeKeysLowering = ExchangeKeysLoweringBase<ExchangeKeysOp, false>;
using ExchangeKeysAsyncLowering = ExchangeKeysLoweringBase<ExchangeKeysAsyncOp, true>;

struct BarrierLowering : public OpConversionPattern<BarrierOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__barrier",
                 {i64Ty(ctx)}, i32Ty(ctx), {adaptor.getToken()});
        rewriter.eraseOp(op);
        return success();
    }
};

struct EstablishChannelLowering : public OpConversionPattern<EstablishChannelOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(EstablishChannelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        Value dp =
            constInt(rewriter, op.getLoc(), i32Ty(ctx), static_cast<int64_t>(op.getDataPath()));
        emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__establish_channel",
                 {ptrTy(ctx), i32Ty(ctx)}, i32Ty(ctx), {adaptor.getSession(), dp});
        rewriter.eraseOp(op);
        return success();
    }
};

struct SetCoprocessorFnLowering : public OpConversionPattern<SetCoprocessorFnOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(SetCoprocessorFnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__set_coprocessor_fn",
                 {ptrTy(op.getContext())}, i32Ty(op.getContext()), {adaptor.getSession()});
        rewriter.eraseOp(op);
        return success();
    }
};

struct CommitWorkItemLowering : public OpConversionPattern<CommitWorkItemOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CommitWorkItemOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        Value idx = constInt(rewriter, op.getLoc(), i32Ty(ctx), op.getWorkItemIdx());
        Value inB = constInt(rewriter, op.getLoc(), i64Ty(ctx), op.getInBytes());
        Value outB = constInt(rewriter, op.getLoc(), i64Ty(ctx), op.getOutBytes());
        emitCall(rewriter, op.getLoc(), moduleOf(op), "__catalyst__transport__commit_work_item",
                 {ptrTy(ctx), i32Ty(ctx), i64Ty(ctx), i64Ty(ctx)}, i32Ty(ctx),
                 {adaptor.getSession(), idx, inB, outB});
        rewriter.eraseOp(op);
        return success();
    }
};

struct KickLowering : public OpConversionPattern<KickOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(KickOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto *ctx = op.getContext();
        ModuleOp mod = moduleOf(op);
        Value slot = emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__data_slot",
                              {ptrTy(ctx)}, ptrTy(ctx), {adaptor.getSession()});
        LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getPayload(), slot);
        Value idx = constInt(rewriter, op.getLoc(), i32Ty(ctx), op.getWorkItemIdx());
        emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__kick",
                 {ptrTy(ctx), i32Ty(ctx)}, i32Ty(ctx), {adaptor.getSession(), idx});
        rewriter.eraseOp(op);
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
        Value one = constInt(rewriter, op.getLoc(), i64Ty(ctx), 1);
        Value buf = LLVM::AllocaOp::create(rewriter, op.getLoc(), ptrTy(ctx), i64Ty(ctx), one,
                                           /*alignment=*/8);
        Value bytes = constInt(rewriter, op.getLoc(), i64Ty(ctx), op.getBytes());
        emitCall(rewriter, op.getLoc(), mod, "__catalyst__transport__collect",
                 {ptrTy(ctx), ptrTy(ctx), i64Ty(ctx)}, i32Ty(ctx),
                 {adaptor.getSession(), buf, bytes});
        Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), i64Ty(ctx), buf);
        rewriter.replaceOp(op, loaded);
        return success();
    }
};

struct LastRttLowering : public OpConversionPattern<LastRttNsOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LastRttNsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Value r = emitCall(rewriter, op.getLoc(), moduleOf(op),
                           "__catalyst__transport__last_rtt_ns", {ptrTy(op.getContext())},
                           i64Ty(op.getContext()), {adaptor.getSession()});
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

} // namespace

struct ConvertTransportToLLVMPass
    : public impl::ConvertTransportToLLVMPassBase<ConvertTransportToLLVMPass> {
    using ConvertTransportToLLVMPassBase::ConvertTransportToLLVMPassBase;

    void runOnOperation() override
    {
        MLIRContext *ctx = &getContext();
        LLVMTypeConverter tc(ctx);
        tc.addConversion([ctx](SessionType) -> Type { return LLVM::LLVMPointerType::get(ctx); });
        tc.addConversion([ctx](TokenType) -> Type { return IntegerType::get(ctx, 64); });

        RewritePatternSet patterns(ctx);
        patterns.add<CreateLowering, ConnectLowering, ConnectAsyncLowering, ExchangeKeysLowering,
                     ExchangeKeysAsyncLowering, BarrierLowering, EstablishChannelLowering,
                     SetCoprocessorFnLowering, CommitWorkItemLowering, KickLowering, CollectLowering,
                     LastRttLowering>(tc, ctx);
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
