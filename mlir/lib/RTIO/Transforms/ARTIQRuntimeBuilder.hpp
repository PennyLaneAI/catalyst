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

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "RTIO/IR/RTIOOps.h" // For ConfigAttr

namespace catalyst {
namespace rtio {

using namespace mlir;

//===----------------------------------------------------------------------===//
// ARTIQ Function Names
//===----------------------------------------------------------------------===//

namespace ARTIQFuncNames {
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
} // namespace ARTIQFuncNames

//===----------------------------------------------------------------------===//
// ARTIQ Hardware Configuration
//===----------------------------------------------------------------------===//

namespace ARTIQHardwareConfig {
constexpr double nanosecondPeriod = 1e-9;
constexpr double ftwScaleFactor = 4.294967296;      // 2^32 / 1e9
constexpr double powScaleFactor = 65536.0;          // 2^16
constexpr int32_t maxAmplitude = 0x3FFF;            // 14-bit max ASF
constexpr int32_t profile7Instruction = 0x15000000; // 0x0E (Profile 0) + 7 = 0x15
constexpr int64_t initSlackDelay = 125000;
constexpr int64_t freqSetSlackDelay = 10000; // 1e-5s in mu
constexpr int32_t spiDiv = 2; // SPI divider: ARTIQ standard is div=2 for fast transfers
constexpr int32_t spiLen8 = 8;
constexpr int32_t spiLen32 = 32;
constexpr int32_t spiFlagsKeepCS = 8;     // SPI_CS_POLARITY (CS low to listen)
constexpr int32_t spiFlagsReleaseCS = 10; // SPI_CS_POLARITY | SPI_END (CS high to release)
constexpr int64_t ioUpdatePulseWidth = 8;
constexpr int64_t refPeriodMu = 8;   // RTIO reference period (Kasli = 8ns @ 125MHz RTIO clock)
constexpr int64_t minTTLPulseMu = 8; // Minimum TTL pulse duration to avoid 0 duration events
} // namespace ARTIQHardwareConfig

//===----------------------------------------------------------------------===//
// ARTIQ Runtime Builder
//===----------------------------------------------------------------------===//

/// Helper class for building ARTIQ runtime function calls.
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
        auto func = ensureFunc(ARTIQFuncNames::nowMu, LLVM::LLVMFunctionType::get(i64Ty, {}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    void atMu(Value time)
    {
        auto func = ensureFunc(ARTIQFuncNames::atMu, LLVM::LLVMFunctionType::get(voidTy, {i64Ty}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{time});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    void delayMu(Value duration)
    {
        auto func =
            ensureFunc(ARTIQFuncNames::delayMu, LLVM::LLVMFunctionType::get(voidTy, {i64Ty}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{duration});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    // RTIO operations
    void rtioOutput(Value addr, Value val)
    {
        auto func = ensureFunc(ARTIQFuncNames::rtioOutput,
                               LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{addr, val});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    void rtioInit()
    {
        auto func = ensureFunc(ARTIQFuncNames::rtioInit, LLVM::LLVMFunctionType::get(voidTy, {}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    Value rtioGetCounter()
    {
        auto func =
            ensureFunc(ARTIQFuncNames::rtioGetCounter, LLVM::LLVMFunctionType::get(i64Ty, {}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    // Duration conversion
    Value secToMu(Value durationSec)
    {
        ensureSecToMuFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::secToMu);
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{durationSec});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
        return call.getResult();
    }

    // SPI configuration
    void configSpi(Value baseAddr, Value cs, Value len, Value div, Value flags)
    {
        ensureConfigSpiFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::configSpi);
        auto call =
            LLVM::CallOp::create(builder, getLoc(), func, ValueRange{baseAddr, cs, len, div, flags});
        call.setCConv(LLVM::CConv::Fast);
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    // Wait for SPI transmission to complete.
    // ARTIQ formula: ref_period_mu * ((length + 1) * div + 1)
    void waitForSpi(int32_t len, int32_t div)
    {
        int64_t duration =
            ARTIQHardwareConfig::refPeriodMu * ((static_cast<int64_t>(len) + 1) * div + 1);
        delayMu(constI64(duration));
    }

    // Frequency setting (continuous phase mode)
    Value setFrequency(Value channelId, Value freqHz, Value phaseTurns, Value amplitude)
    {
        ensureSetFrequencyFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::setFrequency);
        LLVM::CallOp::create(builder, getLoc(), func,
                                     ValueRange{channelId, freqHz, phaseTurns, amplitude});
        return nowMu();
    }

    // TTL operations
    void ttlOn(Value channelAddr) { rtioOutput(channelAddr, constI32(1)); }

    void ttlOff(Value channelAddr) { rtioOutput(channelAddr, constI32(0)); }

    // Constant creation helpers
    Value constI32(int32_t val)
    {
        return arith::ConstantOp::create(builder, getLoc(), builder.getI32IntegerAttr(val));
    }

    Value constI64(int64_t val)
    {
        return arith::ConstantOp::create(builder, getLoc(), builder.getI64IntegerAttr(val));
    }

    Value constF64(double val)
    {
        return arith::ConstantOp::create(builder, getLoc(), builder.getF64FloatAttr(val));
    }

    // Accessors
    Type getI32Type() const { return i32Ty; }
    Type getI64Type() const { return i64Ty; }
    Location getLoc() const { return contextOp->getLoc(); }
    ModuleOp getModule() const { return contextOp->getParentOfType<ModuleOp>(); }

    /// Ensure all ARTIQ helper functions are defined in the module.
    /// This should be called before lowering patterns that depend on these functions.
    void ensureHelperFunctions()
    {
        ensureSecToMuFunc();
        ensureConfigSpiFunc();
        ensureSetFrequencyFunc();
    }

  private:
    OpBuilder &builder;
    Operation *contextOp;
    MLIRContext *ctx;
    Type i32Ty, i64Ty, f64Ty, voidTy;

    LLVM::LLVMFuncOp ensureFunc(StringRef name, LLVM::LLVMFunctionType funcTy)
    {
        PatternRewriter rewriter = PatternRewriter(builder.getContext());
        return catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(rewriter, contextOp, name,
                                                                     funcTy);
    }

    void ensureSecToMuFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::secToMu)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(i64Ty, {f64Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::secToMu, funcTy,
                                                     LLVM::Linkage::Internal);
        func.setCConv(LLVM::CConv::Fast);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);
        Value durationSec = entry->getArgument(0);

        // duration_mu = round(duration_sec / 1e-9)
        Value nsPerMu = constF64(ARTIQHardwareConfig::nanosecondPeriod);
        Value durationNs = arith::DivFOp::create(builder, getLoc(), durationSec, nsPerMu);
        Value rounded = math::RoundOp::create(builder, getLoc(), durationNs);
        Value result = arith::FPToSIOp::create(builder, getLoc(), i64Ty, rounded);
        LLVM::ReturnOp::create(builder, getLoc(), result);
    }

    void ensureConfigSpiFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::configSpi)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty, i32Ty, i32Ty, i32Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::configSpi, funcTy,
                                                     LLVM::Linkage::Internal);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value baseAddr = entry->getArgument(0);
        Value cs = entry->getArgument(1);
        Value len = entry->getArgument(2);
        Value div = entry->getArgument(3);
        Value flags = entry->getArgument(4);

        // Config register address = Base | 1
        Value configAddr = arith::OrIOp::create(builder, getLoc(), baseAddr, constI32(1));

        // Pack: (CS << 24) | ((div - 2) << 16) | ((len - 1) << 8) | flags
        Value csShifted = arith::ShLIOp::create(builder, getLoc(), cs, constI32(24));
        Value divOffset = arith::SubIOp::create(builder, getLoc(), div, constI32(2));
        Value divShifted = arith::ShLIOp::create(builder, getLoc(), divOffset, constI32(16));
        Value lenOffset = arith::SubIOp::create(builder, getLoc(), len, constI32(1));
        Value lenShifted = arith::ShLIOp::create(builder, getLoc(), lenOffset, constI32(8));

        Value packed = arith::OrIOp::create(builder, getLoc(), csShifted, divShifted);
        packed = arith::OrIOp::create(builder, getLoc(), packed, lenShifted);
        packed = arith::OrIOp::create(builder, getLoc(), packed, flags);

        rtioOutput(configAddr, packed);
        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{});
    }

    void ensureSetFrequencyFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::setFrequency)) {
            return;
        }

        ensureConfigSpiFunc();

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, f64Ty, f64Ty, f64Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::setFrequency, funcTy,
                                                     LLVM::Linkage::Internal);

        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        // channelId here is the DDS channel index (0, 1, 2, 3) for the Urukul
        Value channelId = entry->getArgument(0);
        Value freqHz = entry->getArgument(1);
        Value phaseTurns = entry->getArgument(2);
        Value amplitude = entry->getArgument(3);

        // Get hardware configuration from module
        auto [spiBaseAddr, csBase, ioUpdateAddr] = getHardwareAddresses(module);

        // CS calculation: csBase is the chip_select for ch0 (typically 4)
        // For Urukul: ch0->CS=4, ch1->CS=5, ch2->CS=6, ch3->CS=7
        // So CS = csBase + channelId
        Value cs = arith::AddIOp::create(builder, getLoc(), constI32(csBase), channelId);
        Value spiBase = constI32(spiBaseAddr);
        Value ioUpdate = constI32(ioUpdateAddr);

        // Calculate FTW: round(frequency * (2^32 / sys_clk))
        Value ftwScale = constF64(ARTIQHardwareConfig::ftwScaleFactor);
        Value ftwDouble = arith::MulFOp::create(builder, getLoc(), freqHz, ftwScale);
        Value ftwRounded = math::RoundOp::create(builder, getLoc(), ftwDouble);
        Value ftw = arith::FPToUIOp::create(builder, getLoc(), i32Ty, ftwRounded);

        // Calculate POW: round(phaseTurns * 65536)
        Value powScale = constF64(ARTIQHardwareConfig::powScaleFactor);
        Value powDouble = arith::MulFOp::create(builder, getLoc(), phaseTurns, powScale);
        Value powRounded = math::RoundOp::create(builder, getLoc(), powDouble);
        Value pow = arith::FPToUIOp::create(builder, getLoc(), i32Ty, powRounded);

        // SPI Transfer: Write instruction to profile 7 (0x15)
        configSpi(spiBase, cs, constI32(ARTIQHardwareConfig::spiLen8),
                  constI32(ARTIQHardwareConfig::spiDiv),
                  constI32(ARTIQHardwareConfig::spiFlagsKeepCS));
        delayMu(constI64(ARTIQHardwareConfig::refPeriodMu));
        rtioOutput(spiBase, constI32(ARTIQHardwareConfig::profile7Instruction));
        // Wait for SPI transmission to complete
        waitForSpi(ARTIQHardwareConfig::spiLen8, ARTIQHardwareConfig::spiDiv);

        // SPI Transfer: Write amplitude + phase (high 32 bits)
        // Convert amplitude (f64, 0.0~1.0) to ASF (i32, 0~0x3FFF)
        configSpi(spiBase, cs, constI32(ARTIQHardwareConfig::spiLen32),
                  constI32(ARTIQHardwareConfig::spiDiv),
                  constI32(ARTIQHardwareConfig::spiFlagsKeepCS));
        delayMu(constI64(ARTIQHardwareConfig::refPeriodMu));
        Value asfScale = constF64(static_cast<double>(ARTIQHardwareConfig::maxAmplitude));
        Value asfDouble = arith::MulFOp::create(builder, getLoc(), amplitude, asfScale);
        Value asfRounded = math::RoundOp::create(builder, getLoc(), asfDouble);
        Value asf = arith::FPToUIOp::create(builder, getLoc(), i32Ty, asfRounded);
        Value asfShifted = arith::ShLIOp::create(builder, getLoc(), asf, constI32(16));
        Value ampPhase = arith::OrIOp::create(builder, getLoc(), asfShifted, pow);
        rtioOutput(spiBase, ampPhase);
        // Wait for SPI transmission to complete
        waitForSpi(ARTIQHardwareConfig::spiLen32, ARTIQHardwareConfig::spiDiv);

        // SPI Transfer: Write FTW (low 32 bits)
        configSpi(spiBase, cs, constI32(ARTIQHardwareConfig::spiLen32),
                  constI32(ARTIQHardwareConfig::spiDiv),
                  constI32(ARTIQHardwareConfig::spiFlagsReleaseCS));
        delayMu(constI64(ARTIQHardwareConfig::refPeriodMu));
        rtioOutput(spiBase, ftw);
        // Wait for SPI transmission to complete
        waitForSpi(ARTIQHardwareConfig::spiLen32, ARTIQHardwareConfig::spiDiv);

        // IO Update pulse: Toggle IO update TTL
        ttlOn(ioUpdate);
        delayMu(constI64(ARTIQHardwareConfig::ioUpdatePulseWidth));
        ttlOff(ioUpdate);

        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{});
    }

    /// Returns hardware addresses: (spiBaseAddr, csBase, ioUpdateAddr)
    /// - spiBaseAddr: SPI RTIO address (channel << 8)
    /// - csBase: Base chip_select value for ch0 (typically 4 for Urukul)
    ///           Other channels use csBase + channelIndex (ch1=5, ch2=6, ch3=7)
    /// - ioUpdateAddr: IO update TTL RTIO address (channel << 8)
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
        // chip_select from urukul0_ch0 is the base CS (typically 4)
        // ch0->CS=4, ch1->CS=5, ch2->CS=6, ch3->CS=7
        int64_t csBase = getChannel({"device_db", "urukul0_ch0", "arguments", "chip_select"});
        int64_t ioUpdateChannel =
            getChannel({"device_db", "ttl_urukul0_io_update", "arguments", "channel"});

        return {static_cast<int32_t>(spiChannel << 8), static_cast<int32_t>(csBase),
                static_cast<int32_t>(ioUpdateChannel << 8)};
    }
};

} // namespace rtio
} // namespace catalyst
