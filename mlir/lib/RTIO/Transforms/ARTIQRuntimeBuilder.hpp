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

#include <algorithm>
#include <cmath>
#include <random>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "RTIO/IR/RTIOOps.h" // For ConfigAttr
#include "Utils.hpp"

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
constexpr StringLiteral rtioInitDataset = "__rtio_init_dataset";
constexpr StringLiteral rtioTransferMeasurementResults = "__rtio_transfer_measurement_results";
constexpr StringLiteral kernel = "__kernel__";
// ARTIQ RPC runtime
constexpr StringLiteral rpcSend = "rpc_send";
constexpr StringLiteral rpcSendAsync = "rpc_send_async";
constexpr StringLiteral rpcRecv = "rpc_recv";
// RTIO input FIFO read
constexpr StringLiteral rtioInputTimestamp = "rtio_input_timestamp";
// Measurement helper functions
constexpr StringLiteral gateRisingMu = "__rtio_gate_rising_mu";
constexpr StringLiteral mockMeasure = "__rtio_mock_measure";
constexpr StringLiteral rtioCount = "__rtio_count";
constexpr StringLiteral waitUntilMu = "__rtio_wait_until_mu";
} // namespace ARTIQFuncNames

//===----------------------------------------------------------------------===//
// ARTIQ Hardware Configuration
//===----------------------------------------------------------------------===//

namespace ARTIQHardwareConfig {
// Hardware constants
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

// Simulated fluorescence / DMD during measurement window
constexpr int64_t simPhotonPulseOnMu = 100;
constexpr int64_t simPhotonGapMu = 10;
constexpr int64_t simPhotonPeriodMu = simPhotonPulseOnMu + simPhotonGapMu;
constexpr int64_t measurementPhotonLeadMu = 10;
constexpr int64_t measurementStartOffsetMu = 100000;
constexpr int32_t defaultTtlInOutGateLatencyMu = 104;

/// TTL6 trigger pulse fires 200 mu before start for Red Pitaya oscilloscope acquisition.
constexpr int64_t scopeTriggerLeadMu = 200;

/// Compute a Poisson-distributed simulated photon count
inline int32_t computeSimulatedPhotonCount(int64_t durationMu)
{
    int64_t maxPhotons = (durationMu - measurementPhotonLeadMu) / simPhotonPeriodMu;
    if (maxPhotons <= 0) {
        return 0;
    }

    // convert duration (ns) to us
    double durationUs = static_cast<double>(durationMu) / 1000.0;

    static std::mt19937 rng(std::random_device{}());
    std::poisson_distribution<int32_t> dist(durationUs);
    int32_t count = dist(rng);

    return std::min(static_cast<int64_t>(count), maxPhotons);
}

} // namespace ARTIQHardwareConfig

struct MeasurementChannelAddrs {
    int32_t gateRisingAddr = 0;
    int32_t countChannel = 0;
    int32_t acquisitionOutputAddr = 0;
    int32_t dmdOutputAddr = 0;
};

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

    void waitUntilMu(Value time)
    {
        ensureWaitUntilMuFunc();
        auto func = getModule().lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::waitUntilMu);
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{time});
        call.setTailCallKind(LLVM::TailCallKind::Tail);
    }

    // Duration conversion
    Value secToMu(Value durationSec)
    {
        // Constant fold if the duration is a known constant
        if (auto cst = durationSec.getDefiningOp<arith::ConstantOp>()) {
            if (auto fAttr = dyn_cast<FloatAttr>(cst.getValue())) {
                double sec = fAttr.getValueAsDouble();
                int64_t mu =
                    static_cast<int64_t>(std::round(sec / ARTIQHardwareConfig::nanosecondPeriod));
                return constI64(mu);
            }
        }
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
        auto call = LLVM::CallOp::create(builder, getLoc(), func,
                                         ValueRange{baseAddr, cs, len, div, flags});
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

    // RPC calls
    // void rpc_send(int32_t service, {ptr,i32}* tag, void **args)
    void rpcSend(Value serviceId, Value tagStructPtr, Value argsPtr)
    {
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        auto func = ensureFunc(ARTIQFuncNames::rpcSend,
                               LLVM::LLVMFunctionType::get(voidTy, {i32Ty, ptrTy, ptrTy}));
        LLVM::CallOp::create(builder, getLoc(), func, ValueRange{serviceId, tagStructPtr, argsPtr});
    }

    // void rpc_send_async(int32_t service, {ptr,i32}* tag, void **args)
    void rpcSendAsync(Value serviceId, Value tagStructPtr, Value argsPtr)
    {
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        auto func = ensureFunc(ARTIQFuncNames::rpcSendAsync,
                               LLVM::LLVMFunctionType::get(voidTy, {i32Ty, ptrTy, ptrTy}));
        LLVM::CallOp::create(builder, getLoc(), func, ValueRange{serviceId, tagStructPtr, argsPtr});
    }

    // i32 rpc_recv(void *slot)
    Value rpcRecv(Value slot)
    {
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        auto func =
            ensureFunc(ARTIQFuncNames::rpcRecv, LLVM::LLVMFunctionType::get(i32Ty, {ptrTy}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{slot});
        return call.getResult();
    }

    /// i64 rtio_input_timestamp(i64 deadline_mu, i32 channel), returns timestamp or -1.
    Value rtioInputTimestamp(Value deadlineMu, Value inputChannelI32)
    {
        auto func = ensureFunc(ARTIQFuncNames::rtioInputTimestamp,
                               LLVM::LLVMFunctionType::get(i64Ty, {i64Ty, i32Ty}));
        auto call =
            LLVM::CallOp::create(builder, getLoc(), func, ValueRange{deadlineMu, inputChannelI32});
        return call.getResult();
    }

    // TTL operations
    void ttlOn(Value channelAddr) { rtioOutput(channelAddr, constI32(1)); }

    void ttlOff(Value channelAddr) { rtioOutput(channelAddr, constI32(0)); }

    /// void __rtio_gate_rising_mu(i32 sens_addr, i64 duration_mu)
    void gateRisingMu(Value sensAddr, Value durationMu)
    {
        auto func = ensureFunc(ARTIQFuncNames::gateRisingMu,
                               LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i64Ty}));
        LLVM::CallOp::create(builder, getLoc(), func, ValueRange{sensAddr, durationMu});
    }

    /// void __rtio_mock_measure(i64 start_mu, i32 ttl7_addr, i64 photon_count)
    void mockMeasure(Value startMu, Value ttl7Addr, Value photonCount)
    {
        auto func = ensureFunc(ARTIQFuncNames::mockMeasure,
                               LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty, i64Ty}));
        LLVM::CallOp::create(builder, getLoc(), func, ValueRange{startMu, ttl7Addr, photonCount});
    }

    /// i32 __rtio_count(i64 deadline, i32 channel), returns number of edges.
    Value rtioCount(Value deadline, Value channel)
    {
        auto func = ensureFunc(ARTIQFuncNames::rtioCount,
                               LLVM::LLVMFunctionType::get(i32Ty, {i64Ty, i32Ty}));
        auto call = LLVM::CallOp::create(builder, getLoc(), func, ValueRange{deadline, channel});
        return call.getResult();
    }

    /// Get the measurement channel addresses from the device_db
    static MeasurementChannelAddrs getMeasurementChannelAddresses(ModuleOp module)
    {
        MeasurementChannelAddrs out;
        auto configAttr = module->getAttrOfType<ConfigAttr>(ConfigAttr::getModuleAttrName());
        if (!configAttr) {
            return out;
        }

        static constexpr StringRef kDb = "device_db";
        static constexpr StringRef kArgs = "arguments";
        static constexpr StringRef kCh = "channel";

        int64_t ttl0Raw = device_db_detail::intAtPath(configAttr, {kDb, "ttl0", kArgs, kCh});
        int64_t ttl6Raw = device_db_detail::intAtPath(configAttr, {kDb, "ttl6", kArgs, kCh});
        int64_t ttl7Raw = device_db_detail::intAtPath(configAttr, {kDb, "ttl7", kArgs, kCh});

        out.gateRisingAddr = static_cast<int32_t>((ttl0Raw << 8) | 2);
        out.countChannel = static_cast<int32_t>(ttl0Raw);
        out.acquisitionOutputAddr = static_cast<int32_t>(ttl6Raw << 8);
        out.dmdOutputAddr = static_cast<int32_t>(ttl7Raw << 8);
        return out;
    }

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
        // Timing helpers
        ensureSecToMuFunc();
        ensureWaitUntilMuFunc();

        // Frequency setting
        ensureConfigSpiFunc();
        ensureSetFrequencyFunc();

        // Measurement helper functions
        ensureGateRisingMuFunc();
        ensureMockMeasureFunc();
        ensureCountFunc();
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
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::setFrequency,
                                             funcTy, LLVM::Linkage::Internal);

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

        int64_t spiChannel = device_db_detail::intAtPath(
            configAttr, {"device_db", "spi_urukul0", "arguments", "channel"});
        // chip_select from urukul0_ch0 is the base CS (typically 4)
        // ch0->CS=4, ch1->CS=5, ch2->CS=6, ch3->CS=7
        int64_t csBase = device_db_detail::intAtPath(
            configAttr, {"device_db", "urukul0_ch0", "arguments", "chip_select"});
        int64_t ioUpdateChannel = device_db_detail::intAtPath(
            configAttr, {"device_db", "ttl_urukul0_io_update", "arguments", "channel"});

        return {static_cast<int32_t>(spiChannel << 8), static_cast<int32_t>(csBase),
                static_cast<int32_t>(ioUpdateChannel << 8)};
    }

    /// Opens the TTL0 sensitivity gate for `duration_mu`
    void ensureGateRisingMuFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::gateRisingMu)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i64Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::gateRisingMu,
                                             funcTy, LLVM::Linkage::Internal);
        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value sensAddr = entry->getArgument(0);
        Value durationMu = entry->getArgument(1);

        rtioOutput(sensAddr, constI32(1));
        delayMu(durationMu);
        rtioOutput(sensAddr, constI32(0));
        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{});
    }

    /// Simulated photon events:
    /// ```
    /// for i in 0..<photon_count>:
    ///     fire TTL7 at start_mu + leadMu + i * periodMu
    /// ```
    void ensureMockMeasureFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::mockMeasure)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i32Ty, i64Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::mockMeasure, funcTy,
                                             LLVM::Linkage::Internal);
        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value startMu = entry->getArgument(0);
        Value ttl7Addr = entry->getArgument(1);
        Value photonCount = entry->getArgument(2);

        Value periodMu = constI64(ARTIQHardwareConfig::simPhotonPeriodMu);
        Value leadMu = constI64(ARTIQHardwareConfig::measurementPhotonLeadMu);
        Value onMu = constI64(ARTIQHardwareConfig::simPhotonPulseOnMu);

        Block *loopHead = new Block();
        Block *body = new Block();
        Block *exit = new Block();
        func.getBody().push_back(loopHead);
        func.getBody().push_back(body);
        func.getBody().push_back(exit);
        loopHead->addArgument(i64Ty, getLoc());

        Value c0 = constI64(0);
        LLVM::BrOp::create(builder, getLoc(), ValueRange{c0}, loopHead);

        builder.setInsertionPointToStart(loopHead);
        Value iv = loopHead->getArgument(0);
        Value cond =
            arith::CmpIOp::create(builder, getLoc(), arith::CmpIPredicate::slt, iv, photonCount);
        LLVM::CondBrOp::create(builder, getLoc(), cond, body, exit);

        builder.setInsertionPointToStart(body);
        Value offset = arith::MulIOp::create(builder, getLoc(), iv, periodMu);
        Value base = arith::AddIOp::create(builder, getLoc(), startMu, leadMu);
        Value tPulse = arith::AddIOp::create(builder, getLoc(), base, offset);
        atMu(tPulse);
        ttlOn(ttl7Addr);
        delayMu(onMu);
        ttlOff(ttl7Addr);
        Value ivNext = arith::AddIOp::create(builder, getLoc(), iv, constI64(1));
        LLVM::BrOp::create(builder, getLoc(), ValueRange{ivNext}, loopHead);

        builder.setInsertionPointToStart(exit);
        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{});
    }

    /// Count the number of edges on the given channel before the deadline
    /// ```
    /// count = 0
    /// while rtio_input_timestamp(deadline, channel) > -1:
    ///     count++
    /// return count
    /// ```
    void ensureCountFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::rtioCount)) {
            return;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i64Ty, i32Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::rtioCount, funcTy,
                                             LLVM::Linkage::Internal);
        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value deadline = entry->getArgument(0);
        Value channel = entry->getArgument(1);

        Block *loopHead = new Block();
        Block *done = new Block();
        func.getBody().push_back(loopHead);
        func.getBody().push_back(done);
        loopHead->addArgument(i32Ty, getLoc());

        LLVM::BrOp::create(builder, getLoc(), ValueRange{constI32(0)}, loopHead);

        builder.setInsertionPointToStart(loopHead);
        Value count = loopHead->getArgument(0);
        Value ts = rtioInputTimestamp(deadline, channel);
        Value more =
            arith::CmpIOp::create(builder, getLoc(), arith::CmpIPredicate::sgt, ts, constI64(-1));
        Value countNext = arith::AddIOp::create(builder, getLoc(), count, constI32(1));
        LLVM::CondBrOp::create(builder, getLoc(), more, loopHead, ValueRange{countNext}, done,
                               ValueRange{});

        builder.setInsertionPointToStart(done);
        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{count});
    }

    /// Busy-wait until the counter >= time
    void ensureWaitUntilMuFunc()
    {
        auto module = getModule();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::waitUntilMu))
            return;

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i64Ty});
        auto func = LLVM::LLVMFuncOp::create(builder, getLoc(), ARTIQFuncNames::waitUntilMu, funcTy,
                                             LLVM::Linkage::Internal);
        Block *entry = func.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        Value time = entry->getArgument(0);

        Block *loopHead = new Block();
        Block *exit = new Block();
        func.getBody().push_back(loopHead);
        func.getBody().push_back(exit);

        LLVM::BrOp::create(builder, getLoc(), ValueRange{}, loopHead);

        builder.setInsertionPointToStart(loopHead);
        Value counter = rtioGetCounter();
        Value cond =
            arith::CmpIOp::create(builder, getLoc(), arith::CmpIPredicate::slt, counter, time);
        LLVM::CondBrOp::create(builder, getLoc(), cond, loopHead, ValueRange{}, exit, ValueRange{});

        builder.setInsertionPointToStart(exit);
        LLVM::ReturnOp::create(builder, getLoc(), ValueRange{});
    }
};

} // namespace rtio
} // namespace catalyst
