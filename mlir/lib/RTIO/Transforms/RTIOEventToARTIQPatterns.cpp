// Copyright 2025 Xanadu Quantum Technologies Inc.
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "RTIO/IR/RTIOOps.h"
#include "RTIO/Transforms/Patterns.h"

#include "ARTIQRuntimeBuilder.hpp"
#include "Utils.hpp"

using namespace mlir;
using namespace catalyst::rtio;

namespace {

//===----------------------------------------------------------------------===//
// RPC helpers
//===----------------------------------------------------------------------===//

/// Map an MLIR type to its ARTIQ RPC wire-protocol tag character.
/// i32 -> 'i',  i64 -> 'I',  f64/f32 -> 'f',  everything else -> 'O'
static char tagCodeForType(Type ty)
{
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
        if (intTy.getWidth() <= 32) {
            return 'i';
        }
        return 'I';
    }
    if (isa<Float64Type>(ty) || isa<Float32Type>(ty)) {
        return 'f';
    }
    return 'O';
}

/// Build the ARTIQ tag string from operand and result MLIR types.
/// e.g. (i64, i64, f64) -> void  =>  "n:IIf"
///      () -> i64                =>  "I:"
///      () -> void               =>  "n:"
static std::string buildTagFromTypes(TypeRange argTypes, TypeRange resultTypes)
{
    std::string tag;
    if (resultTypes.empty()) {
        tag += 'n';
    }
    else {
        tag += tagCodeForType(resultTypes[0]);
    }
    tag += ':';
    for (Type t : argTypes) {
        tag += tagCodeForType(t);
    }
    return tag;
}

/// Ensure a null-terminated LLVM global constant for the given string exists
/// in the module.
///
/// The global is inserted at the start of the module body.
//  The AddressOf/GEPs are emitted at the caller's insertion point.
///
/// Example:
/// Given a rtio.rpc operation:
/// ```mlir
/// rtio.rpc @foo rpc_id(1) (%a, %b : i32, i64)
/// ```
///
/// The lowering produces:
/// ┌───────────────────────────────────────────────────────────────────┐
/// │ 1. tag = "n:iI" (derived from arg/result MLIR types)              │
/// │    tagPtr = __rtio_str_n_iI (global)                              │
/// │                                                                   │
/// │ 2. argsArray = alloca [3 x ptr]                                   │
/// │    argSlot[0] = alloca i32; store %a; argsArray[0] = &argSlot[0]  │
/// │    argSlot[1] = alloca i64; store %b; argsArray[1] = &argSlot[1]  │
/// │    argsArray[2] = nullptr                                         │
/// │                                                                   │
/// │ 3. rpc_send(rpc_id, tagPtr, argsArray)                            │
/// │                                                                   │
/// │ 4. (sync only) resultSlot = alloca [8 x i8]; rpc_recv(resultSlot) │
/// └───────────────────────────────────────────────────────────────────┘
static Value getOrCreateStringGlobal(ConversionPatternRewriter &rewriter, ModuleOp module,
                                     Location loc, StringRef str)
{
    MLIRContext *ctx = rewriter.getContext();

    // get a unique symbol name from the string content (__rtio_str_<string>)
    std::string globalName = "__rtio_str_" + str.str();
    for (char &c : globalName) {
        // Replace non-alphanumeric characters with '_'
        // Example: "n:IIf" -> "n_IIf"
        if (!llvm::isAlnum(c) && c != '_') {
            c = '_';
        }
    }

    auto i8Ty = IntegerType::get(ctx, 8);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Create the global only if it does not exist yet.
    LLVM::GlobalOp globalOp;
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
        globalOp = existing;
    }
    else {
        SmallVector<uint8_t> bytes(str.begin(), str.end());
        bytes.push_back('\0');

        auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, bytes.size());
        auto dataAttrType = RankedTensorType::get({(int64_t)bytes.size()}, i8Ty);
        auto dataAttr = DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(bytes));

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        globalOp = LLVM::GlobalOp::create(rewriter, loc, arrayTy, /*isConstant=*/true,
                                          LLVM::Linkage::Private, globalName, dataAttr);
    }

    // Emit AddressOf + GEP[0, 0] to get the pointer to the first byte
    auto arrayTy = cast<LLVM::LLVMArrayType>(globalOp.getType());
    Value addr = LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalOp.getName());
    SmallVector<LLVM::GEPArg> indices = {0, 0};
    return LLVM::GEPOp::create(rewriter, loc, ptrTy, arrayTy, addr, indices);
}

} // namespace

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct PulseOpLowering : public OpConversionPattern<RTIOPulseOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIOPulseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        ARTIQRuntimeBuilder artiq(rewriter, op);

        // Set timeline position
        artiq.atMu(adaptor.getWait());

        if (op->hasAttr("_control")) {
            return lowerControlPulse(op, adaptor, rewriter, artiq);
        }
        else if (op->hasAttr("_slack")) {
            return lowerSlackPulse(op, rewriter, artiq);
        }
        return lowerTTLPulse(op, adaptor, rewriter, artiq);
    }

  private:
    LogicalResult lowerControlPulse(RTIOPulseOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    ARTIQRuntimeBuilder &artiq) const
    {
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        auto setFreqFunc = mod.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::setFrequency);
        if (!setFreqFunc) {
            return op->emitError("Cannot find ") << ARTIQFuncNames::setFrequency << " function";
        }

        Value amplitude = artiq.constF64(1.0);
        LLVM::CallOp::create(rewriter, op.getLoc(), setFreqFunc,
                             ValueRange{adaptor.getChannel(), adaptor.getFrequency(),
                                        adaptor.getPhase(), amplitude});

        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }

    LogicalResult lowerSlackPulse(RTIOPulseOp op, ConversionPatternRewriter &rewriter,
                                  ARTIQRuntimeBuilder &artiq) const
    {
        artiq.delayMu(artiq.constI64(ARTIQHardwareConfig::freqSetSlackDelay));
        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }

    LogicalResult lowerTTLPulse(RTIOPulseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                ARTIQRuntimeBuilder &artiq) const
    {
        Value channelAddr = computeChannelDeviceAddr(rewriter, op, adaptor.getChannel());
        Value durationMu = artiq.secToMu(adaptor.getDuration());

        // Enforce minimum pulse duration to avoid 0 duratoin events
        Value minDuration = artiq.constI64(ARTIQHardwareConfig::minTTLPulseMu);
        durationMu = arith::MaxSIOp::create(rewriter, op.getLoc(), durationMu, minDuration);

        artiq.ttlOn(channelAddr);
        artiq.delayMu(durationMu);
        artiq.ttlOff(channelAddr);

        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }
};

struct SyncOpLowering : public OpConversionPattern<RTIOSyncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIOSyncOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        ValueRange events = adaptor.getEvents();

        if (events.size() == 1) {
            rewriter.replaceOp(op, events[0]);
            return success();
        }

        // Compute maximum timestamp
        Value maxTime = events[0];
        for (size_t i = 1; i < events.size(); ++i) {
            maxTime = arith::MaxSIOp::create(rewriter, op.getLoc(), maxTime, events[i]);
        }

        ARTIQRuntimeBuilder artiq(rewriter, op);
        artiq.atMu(maxTime);
        rewriter.replaceOp(op, artiq.nowMu());
        return success();
    }
};

struct EmptyOpLowering : public OpConversionPattern<RTIOEmptyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIOEmptyOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        ARTIQRuntimeBuilder artiq(rewriter, op);
        rewriter.replaceOp(op, artiq.nowMu());
        return success();
    }
};

struct ChannelOpLowering : public OpConversionPattern<RTIOChannelOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIOChannelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        int32_t channelId = extractChannelId(op.getChannel());
        Type resultType = getTypeConverter()->convertType(op.getChannel().getType());
        Value result = arith::ConstantOp::create(rewriter, op.getLoc(),
                                                 rewriter.getIntegerAttr(resultType, channelId));
        rewriter.replaceOp(op, result);
        return success();
    }
};

/// Get slot size in bytes for ARTIQ RPC return type code.
static int getSlotSizeForReturnCode(char code)
{
    switch (code) {
    case 'i':
        return 4;
    case 'I':
    case 'f':
    case 's':
    case 'O':
        return 8;
    default:
        llvm_unreachable("unknown ARTIQ RPC return type code");
    }
}

/// Lower `rtio.rpc` to ARTIQ `rpc_send` / `rpc_recv` calls.
///
/// Protocol:
///   1. Derive the ARTIQ tag string from MLIR operand/result types.
///   2. Build args array.
///   3. Call rpc_send(id, tag_ptr, args_ptr) (or rpc_send_async for async).
///   4. For sync calls, call rpc_recv(slot_ptr) to wait for the host reply.
struct RPCOpLowering : public OpConversionPattern<RTIORPCOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(RTIORPCOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto rpcIdAttr = op->getAttrOfType<IntegerAttr>("rpc_id");
        if (!rpcIdAttr) {
            return op->emitError("rtio.rpc is missing rpc_id attribute, please "
                                 "run the RPC ID assignment sub-pass first");
        }

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        ModuleOp module = op->getParentOfType<ModuleOp>();

        Type i8Ty = IntegerType::get(ctx, 8);
        Type i32Ty = IntegerType::get(ctx, 32);
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);

        ARTIQRuntimeBuilder artiq(rewriter, op);

        // 1. Derive tag from MLIR types and create a string global
        std::string tag = buildTagFromTypes(op.getArgs().getTypes(), op.getResultTypes());
        Value tagPtr = getOrCreateStringGlobal(rewriter, module, loc, tag);
        rewriter.setInsertionPoint(op);

        // 2. Build args array (null-terminated void*[N+1] alloca on the stack)
        ValueRange args = adaptor.getArgs();
        size_t numArgs = args.size();

        auto ptrArrayTy = LLVM::LLVMArrayType::get(ptrTy, numArgs + 1);
        Value one = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value argsArray = LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrArrayTy, one);

        for (size_t i = 0; i < numArgs; i++) {
            Value arg = args[i];
            Type argTy = arg.getType();

            // Box the argument into a stack slot
            Value argSlot;
            if (isa<LLVM::LLVMPointerType>(argTy)) {
                argSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, one);
            }
            else {
                argSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, argTy, one);
            }
            LLVM::StoreOp::create(rewriter, loc, arg, argSlot);

            // Store the slot pointer into ptrArray[i]
            SmallVector<LLVM::GEPArg> idxs = {0, static_cast<int32_t>(i)};
            Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrArrayTy, argsArray, idxs);
            LLVM::StoreOp::create(rewriter, loc, argSlot, elemPtr);
        }

        // Null-terminate the args array
        Value null = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
        SmallVector<LLVM::GEPArg> nullIdxs = {0, static_cast<int32_t>(numArgs)};
        Value nullElemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrArrayTy, argsArray, nullIdxs);
        LLVM::StoreOp::create(rewriter, loc, null, nullElemPtr);

        // 3. Call rpc_send / rpc_send_async
        Value serviceId = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(i32Ty, rpcIdAttr.getInt()));

        SmallVector<Value> replacementValues;

        if (op.getIsAsync()) {
            artiq.rpcSendAsync(serviceId, tagPtr, argsArray);
        }
        else {
            artiq.rpcSend(serviceId, tagPtr, argsArray);

            // Determine return-type tag code for recv slot sizing
            char retCode = tag[0];
            int slotSize = (retCode == 'n') ? 8 : getSlotSizeForReturnCode(retCode);
            auto slotTy = LLVM::LLVMArrayType::get(i8Ty, slotSize);
            Value resultSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, slotTy, one);
            artiq.rpcRecv(resultSlot);

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

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// Decomposes a pulse with _frequency attribute into control + slack pulses
struct DecomposePulsePattern : public OpRewritePattern<RTIOPulseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(RTIOPulseOp op, PatternRewriter &rewriter) const override
    {
        if (!op->hasAttr("_frequency")) {
            return failure();
        }

        op->removeAttr("_frequency");
        Location loc = op.getLoc();

        // Create control pulse (frequency setting)
        auto controlPulse = cast<RTIOPulseOp>(rewriter.clone(*op.getOperation()));
        controlPulse->setAttr("_control", rewriter.getUnitAttr());

        // Create slack pulse (timing delay)
        auto slackPulse = cast<RTIOPulseOp>(rewriter.clone(*op.getOperation()));
        slackPulse->setAttr("_slack", rewriter.getUnitAttr());

        // Sync both pulses
        auto eventType = EventType::get(rewriter.getContext());
        Value syncEvent = RTIOSyncOp::create(
            rewriter, loc, eventType, ValueRange{controlPulse.getEvent(), slackPulse.getEvent()});

        rewriter.replaceOp(op, syncEvent);
        return success();
    }
};

/// Removes redundant transitive dependencies from sync operations
struct SimplifySyncPattern : public OpRewritePattern<RTIOSyncOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(RTIOSyncOp op, PatternRewriter &rewriter) const override
    {
        auto events = op.getEvents();
        if (events.size() <= 1) {
            return failure();
        }

        // Find events that aren't reachable from other events
        SmallVector<Value> requiredEvents;
        for (Value event : events) {
            bool isRedundant = llvm::any_of(events, [&](Value other) {
                if (event == other) {
                    return false;
                }
                return canReach(event, other);
            });
            if (!isRedundant) {
                requiredEvents.push_back(event);
            }
        }

        if (requiredEvents.size() == events.size() || requiredEvents.empty()) {
            return failure();
        }

        if (requiredEvents.size() == 1) {
            rewriter.replaceOp(op, requiredEvents[0]);
        }
        else {
            rewriter.replaceOpWithNewOp<RTIOSyncOp>(op, op.getType(), requiredEvents);
        }
        return success();
    }

  private:
    // Check if target is reachable from 'from' by traversing event dependencies.
    static bool canReach(Value target, Value from)
    {
        if (target == from) {
            return true;
        }

        DenseSet<Value> visited;
        SmallVector<Value, 16> queue;
        queue.push_back(from);

        while (!queue.empty()) {
            Value current = queue.pop_back_val();

            if (current == target) {
                return true;
            }

            if (!visited.insert(current).second) {
                continue;
            }

            Operation *defOp = current.getDefiningOp();
            if (!defOp) {
                continue;
            }

            if (auto pulse = dyn_cast<RTIOPulseOp>(defOp)) {
                queue.push_back(pulse.getWait());
            }
            else if (auto sync = dyn_cast<RTIOSyncOp>(defOp)) {
                for (Value ev : sync.getEvents()) {
                    queue.push_back(ev);
                }
            }
        }
        return false;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population Functions
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace rtio {

void populateRTIOToARTIQConversionPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns)
{
    patterns
        .add<SyncOpLowering, EmptyOpLowering, ChannelOpLowering, PulseOpLowering, RPCOpLowering>(
            typeConverter, patterns.getContext());
}

void populateRTIORewritePatterns(RewritePatternSet &patterns)
{
    patterns.add<DecomposePulsePattern, SimplifySyncPattern>(patterns.getContext());
}

void populateRTIOSyncSimplifyPatterns(RewritePatternSet &patterns)
{
    patterns.add<SimplifySyncPattern>(patterns.getContext());
}

void populateRTIOPulseDecomposePatterns(RewritePatternSet &patterns)
{
    patterns.add<DecomposePulsePattern>(patterns.getContext());
}

} // namespace rtio
} // namespace catalyst
