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

#include <deque>

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "RTIO/IR/RTIOOps.h"
#include "RTIO/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::rtio;

namespace catalyst {
namespace rtio {

#define GEN_PASS_DEF_RTIOEVENTTOARTIQPASS
#include "RTIO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

namespace FuncNames {
constexpr StringLiteral setFrequency = "__rtio_set_frequency";
constexpr StringLiteral secToMu = "__rtio_sec_to_mu";
constexpr StringLiteral configSpi = "__rtio_config_spi";
constexpr StringLiteral nowMu = "now_mu";
constexpr StringLiteral atMu = "at_mu";
constexpr StringLiteral delayMu = "delay_mu";
constexpr StringLiteral rtioOutput = "rtio_output";
constexpr StringLiteral rtioInit = "rtio_init";
constexpr StringLiteral rtioGetCounter = "rtio_get_counter";
constexpr StringLiteral kernel = "__kernel__";
} // namespace FuncNames

namespace HardwareConfig {
constexpr double nanosecondPeriod = 1e-9;
constexpr double ftwScaleFactor = 4.294967296; // 2^32 / 1e9
constexpr double powScaleFactor = 65536.0;     // 2^16
constexpr int32_t maxAmplitude = 0x3FFF;       // 14-bit max ASF
constexpr int32_t profile0Instruction = 0x0E000000;
constexpr int64_t initSlackDelay = 125000;
constexpr int64_t freqSetSlackDelay = 10000; // 1e-5s in mu
constexpr int32_t spiDiv = 8;
constexpr int32_t spiLen8 = 8;
constexpr int32_t spiLen32 = 32;
constexpr int32_t spiFlagsKeepCS = 2;
constexpr int32_t spiFlagsReleaseCS = 0;
constexpr int64_t ioUpdatePulseWidth = 8;
} // namespace HardwareConfig

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

using ScheduleGroupsMap = DenseMap<int, llvm::SetVector<Operation *>>;
using GroupingPredicate =
    std::function<bool(rtio::RTIOPulseOp reference, rtio::RTIOPulseOp candidate)>;

//===----------------------------------------------------------------------===//
// ARTIQ Runtime Builder
//===----------------------------------------------------------------------===//

class ARTIQRuntimeBuilder {
  public:
    ARTIQRuntimeBuilder(OpBuilder &builder, Operation *contextOp)
        : builder(builder), contextOp(contextOp), ctx(builder.getContext()),
          i32Ty(IntegerType::get(ctx, 32)), i64Ty(IntegerType::get(ctx, 64)),
          f64Ty(Float64Type::get(ctx)), voidTy(LLVM::LLVMVoidType::get(ctx))
    {
    }

    // Timing management
    Value nowMu()
    {
        auto func = ensureFunc(FuncNames::nowMu, LLVM::LLVMFunctionType::get(i64Ty, {}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    void atMu(Value time)
    {
        auto func = ensureFunc(FuncNames::atMu, LLVM::LLVMFunctionType::get(voidTy, {i64Ty}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{time});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    void delayMu(Value duration)
    {
        auto func = ensureFunc(FuncNames::delayMu, LLVM::LLVMFunctionType::get(voidTy, {i64Ty}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{duration});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    // RTIO operations
    void rtioOutput(Value addr, Value val)
    {
        auto func =
            ensureFunc(FuncNames::rtioOutput, LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{addr, val});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    void rtioInit()
    {
        auto func = ensureFunc(FuncNames::rtioInit, LLVM::LLVMFunctionType::get(voidTy, {}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    Value rtioGetCounter()
    {
        auto func = ensureFunc(FuncNames::rtioGetCounter, LLVM::LLVMFunctionType::get(i64Ty, {}));
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    // Duration conversion
    Value secToMu(Value durationSec)
    {
        ensureSecToMuFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::secToMu);
        auto call = builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{durationSec});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    // SPI configuration
    void configSpi(Value baseAddr, Value cs, Value len, Value div, Value flags)
    {
        ensureConfigSpiFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::configSpi);
        auto call =
            builder.create<LLVM::CallOp>(getLoc(), func, ValueRange{baseAddr, cs, len, div, flags});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    // Frequency setting
    void setFrequency(Value channelId, Value freqHz, Value phaseTurns, Value amplitude)
    {
        ensureSetFrequencyFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::setFrequency);
        builder.create<LLVM::CallOp>(getLoc(), func,
                                     ValueRange{channelId, freqHz, phaseTurns, amplitude});
    }

    // TTL operations
    void ttlOn(Value channelAddr) { rtioOutput(channelAddr, constI32(1)); }

    void ttlOff(Value channelAddr) { rtioOutput(channelAddr, constI32(0)); }

    // Constant creation helpers
    Value constI32(int32_t val)
    {
        return builder.create<arith::ConstantOp>(getLoc(), builder.getI32IntegerAttr(val));
    }

    Value constI64(int64_t val)
    {
        return builder.create<arith::ConstantOp>(getLoc(), builder.getI64IntegerAttr(val));
    }

    Value constF64(double val)
    {
        return builder.create<arith::ConstantOp>(getLoc(), builder.getF64FloatAttr(val));
    }

    // Accessors
    Type getI32Type() const { return i32Ty; }
    Type getI64Type() const { return i64Ty; }
    Location getLoc() const { return contextOp->getLoc(); }
    ModuleOp getModule() const { return contextOp->getParentOfType<ModuleOp>(); }

  private:
    OpBuilder &builder;
    Operation *contextOp;
    MLIRContext *ctx;
    Type i32Ty, i64Ty, f64Ty, voidTy;

    LLVM::LLVMFuncOp ensureFunc(StringRef name, LLVM::LLVMFunctionType funcTy)
    {
        auto module = getModule();
        if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
            return func;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        return builder.create<LLVM::LLVMFuncOp>(contextOp->getLoc(), name, funcTy,
                                                LLVM::Linkage::External);
    }

    void ensureSecToMuFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::secToMu)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(i64Ty, {f64Ty});
        auto func = builder.create<LLVM::LLVMFuncOp>(getLoc(), FuncNames::secToMu, funcTy,
                                                     LLVM::Linkage::Internal);
        func.setCConv(LLVM::CConv::Fast);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);
        Value durationSec = entry->getArgument(0);

        // duration_mu = round(duration_sec / 1e-9)
        Value nsPerMu = constF64(HardwareConfig::nanosecondPeriod);
        Value durationNs = builder.create<arith::DivFOp>(getLoc(), durationSec, nsPerMu);
        Value rounded = builder.create<math::RoundOp>(getLoc(), durationNs);
        Value result = builder.create<arith::FPToSIOp>(getLoc(), i64Ty, rounded);
        builder.create<LLVM::ReturnOp>(getLoc(), result);
    }

    void ensureConfigSpiFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::configSpi)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty, i32Ty, i32Ty, i32Ty});
        auto func = builder.create<LLVM::LLVMFuncOp>(getLoc(), FuncNames::configSpi, funcTy,
                                                     LLVM::Linkage::Internal);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value baseAddr = entry->getArgument(0);
        Value cs = entry->getArgument(1);
        Value len = entry->getArgument(2);
        Value div = entry->getArgument(3);
        Value flags = entry->getArgument(4);

        // Config register address = Base | 1
        Value configAddr = builder.create<arith::OrIOp>(getLoc(), baseAddr, constI32(1));

        // Pack: (CS << 24) | ((div - 2) << 16) | ((len - 1) << 8) | flags
        Value csShifted = builder.create<arith::ShLIOp>(getLoc(), cs, constI32(24));
        Value divOffset = builder.create<arith::SubIOp>(getLoc(), div, constI32(2));
        Value divShifted = builder.create<arith::ShLIOp>(getLoc(), divOffset, constI32(16));
        Value lenOffset = builder.create<arith::SubIOp>(getLoc(), len, constI32(1));
        Value lenShifted = builder.create<arith::ShLIOp>(getLoc(), lenOffset, constI32(8));

        Value packed = builder.create<arith::OrIOp>(getLoc(), csShifted, divShifted);
        packed = builder.create<arith::OrIOp>(getLoc(), packed, lenShifted);
        packed = builder.create<arith::OrIOp>(getLoc(), packed, flags);

        rtioOutput(configAddr, packed);
        builder.create<LLVM::ReturnOp>(getLoc(), ValueRange{});
    }

    void ensureSetFrequencyFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::setFrequency)) {
            return;
        }

        ensureConfigSpiFunc();

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, f64Ty, f64Ty, f64Ty});
        auto func = builder.create<LLVM::LLVMFuncOp>(getLoc(), FuncNames::setFrequency, funcTy,
                                                     LLVM::Linkage::Internal);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value channelId = entry->getArgument(0);
        Value freqHz = entry->getArgument(1);
        Value phaseTurns = entry->getArgument(2);
        Value amplitude = entry->getArgument(3);

        // Get hardware configuration from module
        auto [spiBaseAddr, csOffset, ioUpdateAddr] = getHardwareAddresses(module);

        Value cs = builder.create<arith::AddIOp>(getLoc(), channelId, constI32(csOffset));
        Value spiBase = constI32(spiBaseAddr);
        Value ioUpdate = constI32(ioUpdateAddr);

        // Calculate FTW: round(frequency * (2^32 / sys_clk))
        Value ftwScale = constF64(HardwareConfig::ftwScaleFactor);
        Value ftwDouble = builder.create<arith::MulFOp>(getLoc(), freqHz, ftwScale);
        Value ftwRounded = builder.create<math::RoundOp>(getLoc(), ftwDouble);
        Value ftw = builder.create<arith::FPToUIOp>(getLoc(), i32Ty, ftwRounded);

        // Calculate POW: round(phaseTurns * 65536)
        Value powScale = constF64(HardwareConfig::powScaleFactor);
        Value powDouble = builder.create<arith::MulFOp>(getLoc(), phaseTurns, powScale);
        Value powRounded = builder.create<math::RoundOp>(getLoc(), powDouble);
        Value pow = builder.create<arith::FPToUIOp>(getLoc(), i32Ty, powRounded);

        // SPI transfer: Write instruction (Profile 0 -> 0x0E)
        configSpi(spiBase, cs, constI32(HardwareConfig::spiLen8), constI32(HardwareConfig::spiDiv),
                  constI32(HardwareConfig::spiFlagsKeepCS));
        rtioOutput(spiBase, constI32(HardwareConfig::profile0Instruction));

        // Write amplitude + phase (high 32 bits)
        // Convert amplitude (f64, 0.0~1.0) to ASF (i32, 0~0x3FFF)
        configSpi(spiBase, cs, constI32(HardwareConfig::spiLen32), constI32(HardwareConfig::spiDiv),
                  constI32(HardwareConfig::spiFlagsKeepCS));
        Value asfScale = constF64(static_cast<double>(HardwareConfig::maxAmplitude));
        Value asfDouble = builder.create<arith::MulFOp>(getLoc(), amplitude, asfScale);
        Value asfRounded = builder.create<math::RoundOp>(getLoc(), asfDouble);
        Value asf = builder.create<arith::FPToUIOp>(getLoc(), i32Ty, asfRounded);
        Value asfShifted = builder.create<arith::ShLIOp>(getLoc(), asf, constI32(16));
        Value ampPhase = builder.create<arith::OrIOp>(getLoc(), asfShifted, pow);
        rtioOutput(spiBase, ampPhase);

        // Write FTW (low 32 bits)
        configSpi(spiBase, cs, constI32(HardwareConfig::spiLen32), constI32(HardwareConfig::spiDiv),
                  constI32(HardwareConfig::spiFlagsReleaseCS));
        rtioOutput(spiBase, ftw);

        // IO Update pulse
        ttlOn(ioUpdate);
        delayMu(constI64(HardwareConfig::ioUpdatePulseWidth));
        ttlOff(ioUpdate);

        builder.create<LLVM::ReturnOp>(getLoc(), ValueRange{});
    }

    std::tuple<int32_t, int32_t, int32_t> getHardwareAddresses(ModuleOp module)
    {
        auto configAttr = module->getAttrOfType<ConfigAttr>(ConfigAttr::getModuleAttrName());
        assert(configAttr && "rtio.config attribute not found on module");

        auto getChannel = [&](ArrayRef<StringRef> path) -> int64_t {
            Attribute current = configAttr;
            for (StringRef key : path) {
                if (auto dict = dyn_cast<DictionaryAttr>(current)) {
                    current = dict.get(key);
                }
                else if (auto cfg = dyn_cast<ConfigAttr>(current)) {
                    current = cfg.get(key);
                }
                else {
                    return 0;
                }
            }
            return cast<IntegerAttr>(current).getInt();
        };

        int64_t spiChannel = getChannel({"device_db", "spi_urukul0", "arguments", "channel"});
        int64_t csOffset = getChannel({"device_db", "urukul0_ch0", "arguments", "chip_select"});
        int64_t ioUpdateChannel =
            getChannel({"device_db", "ttl_urukul0_io_update", "arguments", "channel"});

        return {static_cast<int32_t>(spiChannel << 8), static_cast<int32_t>(csOffset),
                static_cast<int32_t>(ioUpdateChannel << 8)};
    }
};

//===----------------------------------------------------------------------===//
// Config Helpers
//===----------------------------------------------------------------------===//

int32_t extractChannelId(Type channelType)
{
    auto rtioChannelType = cast<rtio::ChannelType>(channelType);
    assert(rtioChannelType.isStatic() && "Only static channel IDs are supported");
    return rtioChannelType.getChannelId().getInt();
}

Value computeChannelDeviceAddr(OpBuilder &builder, Operation *op, Value channelValue)
{
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto configAttr = module->getAttrOfType<ConfigAttr>(ConfigAttr::getModuleAttrName());
    assert(configAttr && "configAttr not found");

    // Get base channel from config
    Attribute current = configAttr;
    for (StringRef key : {"device_db", "ttl_urukul0_sw0", "arguments", "channel"}) {
        if (auto dict = dyn_cast<DictionaryAttr>(current)) {
            current = dict.get(key);
        }
        else if (auto cfg = dyn_cast<ConfigAttr>(current)) {
            current = cfg.get(key);
        }
    }
    int64_t channelBase = cast<IntegerAttr>(current).getInt();

    APInt channelIdAPInt;
    // Static channel
    if (matchPattern(channelValue, m_ConstantInt(&channelIdAPInt))) {
        int64_t channelId = channelIdAPInt.getSExtValue();
        int32_t addr = static_cast<int32_t>((channelId + channelBase) << 8);
        return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(addr));
    }

    // Dynamic channel: compute at runtime
    Value offset = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(channelBase));
    Value sum = builder.create<arith::AddIOp>(loc, channelValue, offset);
    Value shift = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(8));
    return builder.create<arith::ShLIOp>(loc, sum, shift);
}

//===----------------------------------------------------------------------===//
// Device Setup
//===----------------------------------------------------------------------===//

void setupDevice(OpBuilder &builder, func::FuncOp funcOp)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    ARTIQRuntimeBuilder artiq(builder, funcOp);

    // Initialize RTIO
    artiq.rtioInit();

    // Set initial timeline: at_mu(rtio_get_counter() + slack)
    Value counter = artiq.rtioGetCounter();
    Value slack = artiq.constI64(HardwareConfig::initSlackDelay);
    Value initialTime = builder.create<arith::AddIOp>(funcOp.getLoc(), counter, slack);
    artiq.atMu(initialTime);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct PulseOpLowering : public OpConversionPattern<rtio::RTIOPulseOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(rtio::RTIOPulseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        ARTIQRuntimeBuilder artiq(rewriter, op);

        // Set timeline position
        artiq.atMu(adaptor.getWait());

        if (op->hasAttr("_control")) {
            return lowerControlPulse(op, adaptor, rewriter, artiq);
        }
        if (op->hasAttr("_slack")) {
            return lowerSlackPulse(op, rewriter, artiq);
        }
        return lowerTTLPulse(op, adaptor, rewriter, artiq);
    }

  private:
    LogicalResult lowerControlPulse(rtio::RTIOPulseOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    ARTIQRuntimeBuilder &artiq) const
    {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto setFreqFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(FuncNames::setFrequency);
        if (!setFreqFunc) {
            return op->emitError("Cannot find ") << FuncNames::setFrequency << " function";
        }

        Value amplitude = artiq.constF64(1.0);
        rewriter.create<LLVM::CallOp>(op.getLoc(), setFreqFunc,
                                      ValueRange{adaptor.getChannel(), adaptor.getFrequency(),
                                                 adaptor.getPhase(), amplitude});

        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }

    LogicalResult lowerSlackPulse(rtio::RTIOPulseOp op, ConversionPatternRewriter &rewriter,
                                  ARTIQRuntimeBuilder &artiq) const
    {
        artiq.delayMu(artiq.constI64(HardwareConfig::freqSetSlackDelay));
        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }

    LogicalResult lowerTTLPulse(rtio::RTIOPulseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                ARTIQRuntimeBuilder &artiq) const
    {
        Value channelAddr = computeChannelDeviceAddr(rewriter, op, adaptor.getChannel());
        Value durationMu = artiq.secToMu(adaptor.getDuration());

        artiq.ttlOn(channelAddr);
        artiq.delayMu(durationMu);
        artiq.ttlOff(channelAddr);

        Value newTime = artiq.nowMu();
        rewriter.replaceOp(op, newTime);
        return success();
    }
};

struct SyncOpLowering : public OpConversionPattern<rtio::RTIOSyncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(rtio::RTIOSyncOp op, OpAdaptor adaptor,
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
            maxTime = rewriter.create<arith::MaxSIOp>(op.getLoc(), maxTime, events[i]);
        }

        ARTIQRuntimeBuilder artiq(rewriter, op);
        artiq.atMu(maxTime);
        rewriter.replaceOp(op, artiq.nowMu());
        return success();
    }
};

struct EmptyOpLowering : public OpConversionPattern<rtio::RTIOEmptyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(rtio::RTIOEmptyOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        ARTIQRuntimeBuilder artiq(rewriter, op);
        rewriter.replaceOp(op, artiq.nowMu());
        return success();
    }
};

struct ChannelOpLowering : public OpConversionPattern<rtio::RTIOChannelOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(rtio::RTIOChannelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        int32_t channelId = extractChannelId(op.getChannel().getType());
        Type resultType = getTypeConverter()->convertType(op.getChannel().getType());
        Value result = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIntegerAttr(resultType, channelId));
        rewriter.replaceOp(op, result);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// Decomposes a pulse with _frequency attribute into control + slack pulses
struct DecomposePulsePattern : public OpRewritePattern<rtio::RTIOPulseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(rtio::RTIOPulseOp op, PatternRewriter &rewriter) const override
    {
        if (!op->hasAttr("_frequency")) {
            return failure();
        }

        op->removeAttr("_frequency");
        Location loc = op.getLoc();

        // Create control pulse (frequency setting)
        auto controlPulse = cast<rtio::RTIOPulseOp>(rewriter.clone(*op.getOperation()));
        controlPulse->setAttr("_control", rewriter.getUnitAttr());

        // Create slack pulse (timing delay)
        auto slackPulse = cast<rtio::RTIOPulseOp>(rewriter.clone(*op.getOperation()));
        slackPulse->setAttr("_slack", rewriter.getUnitAttr());

        // Sync both pulses
        auto eventType = rtio::EventType::get(rewriter.getContext());
        Value syncEvent = rewriter.create<rtio::RTIOSyncOp>(
            loc, eventType, ValueRange{controlPulse.getEvent(), slackPulse.getEvent()});

        rewriter.replaceOp(op, syncEvent);
        return success();
    }
};

/// Removes redundant transitive dependencies from sync operations
struct SimplifySyncPattern : public OpRewritePattern<rtio::RTIOSyncOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(rtio::RTIOSyncOp op, PatternRewriter &rewriter) const override
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
                DenseSet<Value> visited;
                return canReach(event, other, visited);
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
            rewriter.replaceOpWithNewOp<rtio::RTIOSyncOp>(op, op.getType(), requiredEvents);
        }
        return success();
    }

  private:
    static bool canReach(Value target, Value from, DenseSet<Value> &visited)
    {
        if (target == from) {
            return true;
        }
        if (!visited.insert(from).second) {
            return false;
        }

        Operation *defOp = from.getDefiningOp();
        if (!defOp) {
            return false;
        }

        // TODO shouldn't use recursive call here
        if (auto pulse = dyn_cast<rtio::RTIOPulseOp>(defOp)) {
            return canReach(target, pulse.getWait(), visited);
        }
        else if (auto sync = dyn_cast<rtio::RTIOSyncOp>(defOp)) {
            return llvm::any_of(sync.getEvents(),
                                [&](Value ev) { return canReach(target, ev, visited); });
        }
        return false;
    }
};

//===----------------------------------------------------------------------===//
// Pulse Scheduling
//===----------------------------------------------------------------------===//

class PulseScheduler {
  public:
    PulseScheduler(func::FuncOp funcOp, OpBuilder &builder, GroupingPredicate predicate)
        : funcOp(funcOp), builder(builder), groupingPredicate(std::move(predicate))
    {
    }

    ScheduleGroupsMap schedule()
    {
        // Collect all pulses
        funcOp.walk([&](rtio::RTIOPulseOp pulse) { allPulses.push_back(pulse); });

        // Build consumer map
        for (auto pulse : allPulses) {
            if (auto producer = pulse.getWait().getDefiningOp<rtio::RTIOPulseOp>()) {
                pulseConsumers[producer].insert(pulse);
            }
        }

        processFromEmptyOps();
        return std::move(groups);
    }

  private:
    func::FuncOp funcOp;
    OpBuilder &builder;
    GroupingPredicate groupingPredicate;

    SmallVector<rtio::RTIOPulseOp> allPulses;
    DenseMap<rtio::RTIOPulseOp, SetVector<rtio::RTIOPulseOp>> pulseConsumers;
    DenseSet<Value> processedEvents;
    DenseSet<rtio::RTIOPulseOp> processedPulses;
    ScheduleGroupsMap groups;
    int nextGroupId = 0;

    SmallVector<rtio::RTIOPulseOp> getEventConsumers(Value event)
    {
        SmallVector<rtio::RTIOPulseOp> consumers;
        for (auto *user : event.getUsers()) {
            if (auto pulse = dyn_cast<rtio::RTIOPulseOp>(user)) {
                if (pulse.getWait() == event) {
                    consumers.push_back(pulse);
                }
            }
        }
        return consumers;
    }

    void processFromEmptyOps()
    {
        std::deque<Value> worklist;
        funcOp.walk([&](rtio::RTIOEmptyOp emptyOp) { worklist.push_back(emptyOp.getResult()); });

        while (!worklist.empty()) {
            Value event = worklist.front();
            worklist.pop_front();

            if (!processedEvents.insert(event).second) {
                continue;
            }

            auto newEvents = processEvent(event);
            for (Value newEvent : newEvents) {
                worklist.push_back(newEvent);
            }
        }
    }

    SmallVector<Value> processEvent(Value event)
    {
        SmallVector<Value> nextEvents;
        auto consumers = getEventConsumers(event);
        if (consumers.empty()) {
            return nextEvents;
        }

        // Group pulses by channel, respecting grouping predicate
        DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> channelPulses;
        DenseMap<int32_t, rtio::RTIOPulseOp> channelLastPulse;
        DenseMap<int32_t, rtio::RTIOPulseOp> channelBoundary;
        SmallVector<rtio::RTIOPulseOp> boundaryWaiters;

        // Initial
        for (auto pulse : consumers) {
            if (processedPulses.contains(pulse)) {
                continue;
            }

            int32_t channel = extractChannelId(pulse.getChannel().getType());
            if (canJoinGroup(pulse, channelPulses)) {
                channelPulses[channel].push_back(pulse);
                channelLastPulse[channel] = pulse;
            }
            else {
                if (!channelBoundary.count(channel)) {
                    channelBoundary[channel] = pulse;
                }
                boundaryWaiters.push_back(pulse);
            }
        }

        if (channelPulses.empty()) {
            return nextEvents;
        }

        // Extend chains on each channel
        extendChannelChains(channelPulses, channelLastPulse, channelBoundary);

        // Record group
        recordGroup(channelPulses);

        // Create sync and update dependencies
        return createSyncAndUpdateDeps(channelPulses, channelLastPulse, channelBoundary,
                                       boundaryWaiters);
    }

    bool canJoinGroup(rtio::RTIOPulseOp pulse,
                      const DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> &channelPulses)
    {
        for (auto &[ch, pulses] : channelPulses) {
            if (!llvm::all_of(pulses,
                              [&](auto existing) { return groupingPredicate(existing, pulse); })) {
                return false;
            }
        }
        return true;
    }

    void extendChannelChains(DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> &channelPulses,
                             DenseMap<int32_t, rtio::RTIOPulseOp> &channelLastPulse,
                             DenseMap<int32_t, rtio::RTIOPulseOp> &channelBoundary)
    {
        DenseSet<int32_t> stopped;

        while (stopped.size() < channelPulses.size()) {
            for (auto &[channel, pulses] : channelPulses) {
                if (stopped.contains(channel)) {
                    continue;
                }

                auto currentPulse = channelLastPulse[channel];
                processedPulses.insert(currentPulse);

                bool foundNext = false;
                for (auto user : pulseConsumers[currentPulse]) {
                    int32_t userChannel = extractChannelId(user.getChannel().getType());
                    if (userChannel != channel || processedPulses.contains(user)) {
                        continue;
                    }

                    if (groupingPredicate(currentPulse, user)) {
                        channelPulses[channel].push_back(user);
                        channelLastPulse[channel] = user;
                    }
                    else {
                        channelBoundary[channel] = user;
                        stopped.insert(channel);
                    }
                    foundNext = true;
                    break;
                }

                if (!foundNext) {
                    stopped.insert(channel);
                }
            }
        }
    }

    void recordGroup(const DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> &channelPulses)
    {
        int groupId = nextGroupId++;
        auto &groupOps = groups[groupId];
        for (auto &[_, pulses] : channelPulses) {
            for (auto pulse : pulses) {
                groupOps.insert(pulse.getOperation());
            }
        }
    }

    SmallVector<Value>
    createSyncAndUpdateDeps(const DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> &channelPulses,
                            DenseMap<int32_t, rtio::RTIOPulseOp> &channelLastPulse,
                            DenseMap<int32_t, rtio::RTIOPulseOp> &channelBoundary,
                            SmallVector<rtio::RTIOPulseOp> &boundaryWaiters)
    {
        SmallVector<Value> nextEvents;

        if (channelPulses.size() > 1 && !channelBoundary.empty()) {
            // Collect events to sync
            SmallVector<Value> eventsToSync;
            for (auto &entry : channelLastPulse) {
                rtio::RTIOPulseOp pulse = entry.second;
                eventsToSync.push_back(pulse.getEvent());
            }

            auto anyPulse = channelLastPulse.begin()->second;
            builder.setInsertionPointAfter(anyPulse);

            auto eventType = rtio::EventType::get(builder.getContext());
            Value syncEvent =
                builder.create<rtio::RTIOSyncOp>(anyPulse.getLoc(), eventType, eventsToSync);

            // Update boundaries and waiters
            for (auto &entry : channelBoundary) {
                rtio::RTIOPulseOp pulse = entry.second;
                pulse.setWait(syncEvent);
            }
            for (auto pulse : boundaryWaiters) {
                pulse.setWait(syncEvent);
            }
            for (auto &entry : channelLastPulse) {
                rtio::RTIOPulseOp pulse = entry.second;
                for (auto user : pulseConsumers[pulse]) {
                    auto userChannel = extractChannelId(user.getChannel().getType());
                    if (!channelBoundary.count(userChannel) ||
                        channelBoundary[userChannel] != user) {
                        if (user.getWait() == pulse.getEvent()) {
                            user.setWait(syncEvent);
                        }
                    }
                }
            }

            nextEvents.push_back(syncEvent);
        }
        else {
            // No sync needed
            for (auto &entry : channelBoundary) {
                rtio::RTIOPulseOp pulse = entry.second;
                nextEvents.push_back(pulse.getWait());
            }
            if (!boundaryWaiters.empty() && !channelLastPulse.empty()) {
                rtio::RTIOPulseOp firstPulse = channelLastPulse.begin()->second;
                Value lastEvent = firstPulse.getEvent();
                for (auto pulse : boundaryWaiters) {
                    pulse.setWait(lastEvent);
                }
                nextEvents.push_back(lastEvent);
            }
            for (auto &entry : channelLastPulse) {
                rtio::RTIOPulseOp pulse = entry.second;
                for (auto *user : pulse.getEvent().getUsers()) {
                    if (auto syncOp = dyn_cast<rtio::RTIOSyncOp>(user)) {
                        nextEvents.push_back(syncOp.getSyncEvent());
                    }
                }
            }
        }

        return nextEvents;
    }
};

//===----------------------------------------------------------------------===//
// Frequency Decomposition
//===----------------------------------------------------------------------===//

void decomposeFrequencyPulses(ScheduleGroupsMap &pulseGroups)
{
    if (pulseGroups.empty()) {
        return;
    }

    auto firstOp = pulseGroups.begin()->second.front();
    OpBuilder builder(firstOp->getContext());

    // Track last frequency per channel (to avoid redundant frequency settings)
    DenseMap<Value, Value> channelLastFreq;

    // Sort groups by ID for deterministic processing
    SmallVector<std::pair<int, llvm::SetVector<Operation *> *>> sortedGroups;
    for (auto &entry : pulseGroups) {
        sortedGroups.push_back({entry.first, &entry.second});
    }
    llvm::sort(sortedGroups, [](const auto &a, const auto &b) { return a.first < b.first; });

    for (auto &[groupId, groupOpsPtr] : sortedGroups) {
        auto &groupOps = *groupOpsPtr;
        if (groupOps.empty()) {
            continue;
        }

        // Find root pulses (pulses whose wait isn't produced by another pulse in this group)
        DenseMap<Value, rtio::RTIOPulseOp> channelRoots;
        for (auto *op : groupOps) {
            auto pulse = cast<rtio::RTIOPulseOp>(op);
            Value wait = pulse.getWait();

            bool isRoot = llvm::none_of(groupOps, [&](Operation *other) {
                return cast<rtio::RTIOPulseOp>(other).getEvent() == wait;
            });

            if (isRoot) {
                Value channel = pulse.getChannel();
                if (!channelRoots.count(channel)) {
                    channelRoots[channel] = pulse;
                }
            }
        }

        if (channelRoots.empty()) {
            continue;
        }

        // Filter to channels needing frequency change
        DenseMap<Value, rtio::RTIOPulseOp> needsFreqSet;
        for (auto &entry : channelRoots) {
            Value channel = entry.first;
            rtio::RTIOPulseOp pulse = entry.second;
            Value freq = pulse.getFrequency();
            auto it = channelLastFreq.find(channel);
            if (it == channelLastFreq.end() || it->second != freq) {
                needsFreqSet[channel] = pulse;
                channelLastFreq[channel] = freq;
            }
        }

        if (needsFreqSet.empty()) {
            continue;
        }

        // Collect original wait events
        SmallVector<Value> originalWaits;
        for (auto &entry : channelRoots) {
            rtio::RTIOPulseOp pulse = entry.second;
            Value wait = pulse.getWait();
            if (!llvm::is_contained(originalWaits, wait)) {
                originalWaits.push_back(wait);
            }
        }

        // Find first root pulse (for insertion point)
        rtio::RTIOPulseOp firstRoot = nullptr;
        for (auto &entry : channelRoots) {
            rtio::RTIOPulseOp pulse = entry.second;
            if (!firstRoot || pulse->isBeforeInBlock(firstRoot)) {
                firstRoot = pulse;
            }
        }

        builder.setInsertionPoint(firstRoot);

        // Create sync
        Value chainStart =
            originalWaits.size() > 1
                ? builder.create<rtio::RTIOSyncOp>(
                      firstRoot.getLoc(), rtio::EventType::get(builder.getContext()), originalWaits)
                : originalWaits[0];

        // Create frequency setting chain
        Value lastFreqEvent = chainStart;
        for (auto &entry : needsFreqSet) {
            rtio::RTIOPulseOp originalPulse = entry.second;
            auto freqPulse = cast<rtio::RTIOPulseOp>(builder.clone(*originalPulse.getOperation()));
            freqPulse.setWait(lastFreqEvent);
            freqPulse->setAttr("_frequency", builder.getUnitAttr());
            lastFreqEvent = freqPulse.getEvent();
        }

        // Update root pulses to wait on last frequency event
        for (auto &entry : channelRoots) {
            rtio::RTIOPulseOp pulse = entry.second;
            pulse.setWait(lastFreqEvent);
        }
    }
}
} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RTIOEventToARTIQPass : public impl::RTIOEventToARTIQPassBase<RTIOEventToARTIQPass> {
    using RTIOEventToARTIQPassBase::RTIOEventToARTIQPassBase;

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        MLIRContext *ctx = &getContext();
        OpBuilder builder(ctx);

        // Schedule pulses into groups
        DenseMap<func::FuncOp, ScheduleGroupsMap> pulseGroups;
        module.walk([&](func::FuncOp funcOp) {
            PulseScheduler scheduler(funcOp, builder, sameChannelSameFrequency);
            pulseGroups[funcOp] = scheduler.schedule();
        });

        // Topological sort to fix dominance
        sortAllBlocks(module);

        // Simplify sync operations
        if (failed(runPatterns<SimplifySyncPattern>(module, "sync simplification")))
            return signalPassFailure();

        // Decompose frequency pulses
        for (auto &[funcOp, groups] : pulseGroups) {
            decomposeFrequencyPulses(groups);
        }

        // Phase 5: Sort again after frequency decomposition
        sortAllBlocks(module);

        // Decompose _frequency pulses into control + slack
        if (failed(runPatterns<DecomposePulsePattern>(module, "pulse decomposition")))
            return signalPassFailure();

        // Setup device initialization
        if (failed(setupKernelDevice(module, builder)))
            return signalPassFailure();

        // Lowering to LLVM
        if (failed(lowerToLLVM(module)))
            return signalPassFailure();
    }

  private:
    static bool sameChannelSameFrequency(RTIOPulseOp ref, RTIOPulseOp candidate)
    {
        if (ref.getChannel() == candidate.getChannel()) {
            return ref.getFrequency() == candidate.getFrequency();
        }
        return true;
    }

    static void sortAllBlocks(ModuleOp module)
    {
        module.walk([](func::FuncOp funcOp) {
            for (auto &block : funcOp.getBody()) {
                sortTopologically(&block);
            }
        });
    }

    template <typename PatternT> LogicalResult runPatterns(ModuleOp module, StringRef description)
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<PatternT>(&getContext());
        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            module.emitError("Failed during ") << description;
            return failure();
        }
        return success();
    }

    LogicalResult setupKernelDevice(ModuleOp module, OpBuilder &builder)
    {
        auto kernelFunc = module.lookupSymbol<func::FuncOp>(FuncNames::kernel);
        if (!kernelFunc) {
            module.emitError("Cannot find ") << FuncNames::kernel << " function";
            return failure();
        }

        // Ensure helper functions exist
        ARTIQRuntimeBuilder artiq(builder, kernelFunc);
        artiq.setFrequency(artiq.constI32(0), artiq.constF64(0), artiq.constF64(0),
                           artiq.constF64(0));

        setupDevice(builder, kernelFunc);
        return success();
    }

    LogicalResult lowerToLLVM(ModuleOp module)
    {
        MLIRContext *ctx = &getContext();
        LLVMTypeConverter typeConverter(ctx);

        typeConverter.addConversion(
            [](rtio::ChannelType type) { return IntegerType::get(type.getContext(), 32); });
        typeConverter.addConversion(
            [](rtio::EventType type) { return IntegerType::get(type.getContext(), 64); });

        RewritePatternSet patterns(ctx);
        patterns.add<SyncOpLowering, EmptyOpLowering, ChannelOpLowering, PulseOpLowering>(
            typeConverter, ctx);

        ConversionTarget target(*ctx);
        target.addIllegalDialect<rtio::RTIODialect>();
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect, math::MathDialect,
                               scf::SCFDialect, func::FuncDialect>();

        return applyPartialConversion(module, target, std::move(patterns));
    }
};

} // namespace rtio
} // namespace catalyst
