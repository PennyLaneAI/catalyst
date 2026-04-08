// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Ion/IR/IonDialect.h"
#include "Ion/Transforms/Patterns.h"
#include "Ion/Transforms/ValueTracing.h"
#include "Quantum/IR/QuantumDialect.h"
#include "RTIO/IR/RTIODialect.h"
#include "RTIO/IR/RTIOOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace ion {

namespace {

constexpr StringLiteral kPulseGroupAttr = "_group";
constexpr StringLiteral kParallelProtocolIdAttr = "parallel_protocol_id";

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

Value awaitEvents(ArrayRef<Value> events, PatternRewriter &rewriter)
{
    if (events.size() == 1) {
        return events.front();
    }
    auto eventType = rtio::EventType::get(rewriter.getContext());
    return rtio::RTIOSyncOp::create(rewriter, rewriter.getUnknownLoc(), eventType, events);
}

/// Extract the qubit index from a memref.load value.
/// For given `memref.load @qubit_map[%cN]`, returns `N`.
static FailureOr<int64_t> getQubitIndex(Value memrefLoadValue)
{
    auto loadOp = memrefLoadValue.getDefiningOp<memref::LoadOp>();
    if (!loadOp || loadOp.getIndices().size() != 1) {
        return failure();
    }
    IntegerAttr indexAttr;
    if (!matchPattern(loadOp.getIndices()[0], m_Constant<IntegerAttr>(&indexAttr))) {
        return failure();
    }
    return indexAttr.getInt();
}

static std::optional<double> getConstF64(Value v)
{
    FloatAttr attr;
    if (matchPattern(v, m_Constant<FloatAttr>(&attr))) {
        return attr.getValueAsDouble();
    }
    return std::nullopt;
}

/// Find a pulse with the same (frequency, phase) tone.
static rtio::RTIOPulseOp findSameTonePulse(ArrayRef<rtio::RTIOPulseOp> pulses,
                                           rtio::RTIOPulseOp pulse)
{
    auto f = getConstF64(pulse.getFrequency());
    auto p = getConstF64(pulse.getPhase());
    if (!f || !p) {
        return nullptr;
    }

    auto found = llvm::find_if(pulses, [=](rtio::RTIOPulseOp target) {
        return getConstF64(target.getFrequency()) == f && getConstF64(target.getPhase()) == p;
    });
    return found != pulses.end() ? *found : nullptr;
}

/// Merge qubit qualifiers from src into dst.
static void mergeChannelQualifiers(rtio::RTIOPulseOp dst, rtio::RTIOPulseOp src,
                                   PatternRewriter &rewriter, MLIRContext *ctx, Location loc)
{
    auto dstCh = llvm::cast<rtio::ChannelType>(dst.getChannel().getType());
    auto srcCh = llvm::cast<rtio::ChannelType>(src.getChannel().getType());

    SetVector<int64_t> qubits;

    // Merge qualifiers from dst and src
    for (auto q : dstCh.getQualifiers()) {
        qubits.insert(llvm::cast<IntegerAttr>(q).getInt());
    }
    for (auto q : srcCh.getQualifiers()) {
        qubits.insert(llvm::cast<IntegerAttr>(q).getInt());
    }

    SmallVector<Attribute> quals;
    for (int64_t q : qubits) {
        quals.push_back(rewriter.getI64IntegerAttr(q));
    }

    auto mergedType = rtio::ChannelType::get(ctx, "dds", rewriter.getArrayAttr(quals),
                                             rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    Value newCh = rtio::RTIOChannelOp::create(rewriter, loc, mergedType);
    rewriter.moveOpBefore(newCh.getDefiningOp(), dst);

    Value oldCh = dst.getChannel();
    dst.getChannelMutable().assign(newCh);

    // Remove unused channel op
    if (oldCh.getDefiningOp() && oldCh.use_empty()) {
        rewriter.eraseOp(oldCh.getDefiningOp());
    }
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert ion.parallelprotocol and introduce rtio.sync to ensure the order
///
/// Example:
/// ```
/// %0, %1 = ion.parallelprotocol(%q0, %q1) {
///   ^bb0(%arg0, %arg1):
///   %p0 = rtio.pulse(...) : !rtio.event
///   %p1 = rtio.pulse(...) : !rtio.event
///   ion.yield %arg0, %arg1
/// }
/// ```
/// will be converted to:
/// ```
/// %event0 = unrealized_conversion_cast %q0 : !ion.qubit -> !rtio.event
/// %event1 = unrealized_conversion_cast %q1 : !ion.qubit -> !rtio.event
/// %p0 = rtio.pulse(..., wait = %event0) : !rtio.event
/// %p1 = rtio.pulse(..., wait = %event1) : !rtio.event
/// %sync = rtio.sync %p0, %p1 : !rtio.event
/// %0 = unrealized_conversion_cast %sync : !rtio.event -> !ion.qubit
/// %1 = unrealized_conversion_cast %sync : !rtio.event -> !ion.qubit
/// ```
/// Those unrealized conversion casts are used to establish the dependency but will be
/// resolved by the subsequent stages.
struct ParallelProtocolToRTIOPattern : public OpConversionPattern<ion::ParallelProtocolOp> {
    ParallelProtocolToRTIOPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<ion::ParallelProtocolOp>(typeConverter, ctx)
    {
    }

    LogicalResult matchAndRewrite(ion::ParallelProtocolOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = rewriter.getContext();
        Location loc = op.getLoc();

        auto protocolIdAttr = op->getAttrOfType<IntegerAttr>(kParallelProtocolIdAttr);
        assert(protocolIdAttr && "parallel protocol must have parallel protocol id");
        int64_t protocolId = protocolIdAttr.getInt();

        Block *regionBlock = &op.getBodyRegion().front();
        IRMapping irMapping;
        SmallVector<Value> inQubits;
        for (auto [blockArg, operand] :
             llvm::zip(regionBlock->getArguments(), adaptor.getOperands())) {
            irMapping.map(blockArg, operand);

            // collect qubits to trace the events
            if (isa<ion::QubitType>(operand.getType())) {
                inQubits.push_back(operand);
            }
        }

        // create events for each qubit
        auto events = llvm::map_range(inQubits, [&](Value qubit) {
            auto eventType = rtio::EventType::get(ctx);
            return UnrealizedConversionCastOp::create(rewriter, loc, eventType, qubit).getResult(0);
        });

        Value inputSyncEvent = awaitEvents(llvm::to_vector(events), rewriter);

        // Clone all operations from the region
        SmallVector<rtio::RTIOPulseOp> clonedPulses;
        for (auto &regionOp : regionBlock->without_terminator()) {
            auto *clonedOp = rewriter.clone(regionOp, irMapping);
            if (auto pulseOp = dyn_cast<rtio::RTIOPulseOp>(clonedOp))
                clonedPulses.push_back(pulseOp);
            irMapping.map(regionOp.getResults(), clonedOp->getResults());
        }

        // Merge same-tone pulses
        SmallVector<rtio::RTIOPulseOp> survivors;
        for (auto pulse : clonedPulses) {
            if (auto match = findSameTonePulse(survivors, pulse)) {
                // Merge channel qualifiers to `match`, and erase `pulse`
                mergeChannelQualifiers(match, pulse, rewriter, ctx, loc);
                Value ch = pulse.getChannel();
                rewriter.eraseOp(pulse);
                if (ch.getDefiningOp() && ch.use_empty()) {
                    rewriter.eraseOp(ch.getDefiningOp());
                }
            }
            else {
                survivors.push_back(pulse);
            }
        }

        // Set wait dependency and _group on each surviving pulse
        SmallVector<Value> pulseEvents;
        for (auto pulse : survivors) {
            pulse.setWait(inputSyncEvent);
            pulse->setAttr(kPulseGroupAttr, rewriter.getI64IntegerAttr(protocolId));
            pulseEvents.push_back(pulse.getEvent());
        }

        assert(!pulseEvents.empty() &&
               "must have at least one pulse after parallel protocol conversion");
        Value outputSyncEvent = awaitEvents(llvm::to_vector(pulseEvents), rewriter);

        SmallVector<Value> results;
        for (Value result : op.getResults()) {
            auto event = UnrealizedConversionCastOp::create(rewriter, loc, result.getType(),
                                                            outputSyncEvent);
            results.push_back(event.getResult(0));
        }

        rewriter.replaceOp(op, results);
        return success();
    }
};

/// Convert ion.pulse to rtio.pulse
///
/// Example:
/// ```
/// %pulse = ion.pulse(%duration) %qubit {
///   beam = #ion.beam<...>
/// } : !ion.pulse
/// ```
/// will be converted to:
/// ```
/// %ch = rtio.channel "dds" { channel_id = 0 } : !rtio.channel<"dds", [N : i64], 0>
/// %event = rtio.pulse %ch duration(%duration) frequency(%freq) phase(%phase)
///     : !rtio.channel<"dds", [N : i64], 0> -> !rtio.event
/// ```
struct PulseToRTIOPattern : public OpConversionPattern<ion::PulseOp> {
    IonInfo ionInfo;
    DenseMap<Value, Value> &qextractToMemrefMap;
    PulseToRTIOPattern(TypeConverter &typeConverter, MLIRContext *ctx, IonInfo ionInfo,
                       DenseMap<Value, Value> &qextractToMemrefMap)
        : OpConversionPattern<ion::PulseOp>(typeConverter, ctx), ionInfo(ionInfo),
          qextractToMemrefMap(qextractToMemrefMap)
    {
    }

    double calculateFrequency(int64_t transitionIndex, double detuning,
                              const IonInfo &ionInfo) const
    {
        // TODO: raman1_frequency can be passed as a pass option for extensibility
        double raman1_frequency = 2 * llvm::numbers::pi * 844.485e12 -
                                  2 * llvm::numbers::pi * 12.643e9 - 2 * llvm::numbers::pi * 20e6;

        auto energyDiff = ionInfo.getTransitionEnergyDiff(transitionIndex);
        assert(energyDiff.has_value() && "energyDiff must have a value");

        double reference_energy = energyDiff.value();
        double frequency =
            (reference_energy + detuning - raman1_frequency) / (2.0 * llvm::numbers::pi);
        return frequency;
    }

    LogicalResult matchAndRewrite(ion::PulseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        // Get pulse parameters
        Value duration = op.getTime();
        auto beamAttr = op.getBeam();
        auto phaseAttr = op.getPhase();

        // Extract beam parameters
        double detuning = beamAttr.getDetuning().getValueAsDouble();
        double phase = phaseAttr.getValueAsDouble();
        int64_t transitionIndex = beamAttr.getTransitionIndex().getInt();
        double frequency = calculateFrequency(transitionIndex, detuning, ionInfo);
        Value freqValue =
            arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(frequency));
        Value phaseValue =
            arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(phase));

        // Resolve qubit index and set it as the qualifier
        Value memrefLoadValue = nullptr;
        traceValueWithCallback<TraceMode::Qreg>(op.getInQubit(), [&](Value value) -> WalkResult {
            if (qextractToMemrefMap.count(value)) {
                memrefLoadValue = qextractToMemrefMap[value];
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        if (memrefLoadValue == nullptr) {
            op->emitError("Failed to trace the memref load value");
            return failure();
        }

        auto qubitIdx = getQubitIndex(memrefLoadValue);
        if (failed(qubitIdx)) {
            op->emitError("Failed to resolve qubit index from memref load");
            return failure();
        }
        ArrayAttr qualifiers = rewriter.getArrayAttr({rewriter.getI64IntegerAttr(*qubitIdx)});

        IntegerAttr channelIdAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
        auto channelType = rtio::ChannelType::get(ctx, "dds", qualifiers, channelIdAttr);
        Value channel = rtio::RTIOChannelOp::create(rewriter, loc, channelType);

        // Create rtio.pulse
        auto eventType = rtio::EventType::get(ctx);
        Value event = rtio::RTIOPulseOp::create(rewriter, loc, eventType, channel, duration,
                                                freqValue, phaseValue, nullptr);
        rewriter.replaceOp(op, event);

        return success();
    }
};

/// Propagates RTIO events from chain of operations to event types.
///
/// Steps:
/// 1. Traces backward to find all events that contribute to the current event value
/// 2. Creates a sync event from all collected events
/// 3. Replaces the cast operation with the sync event
struct PropagateEventsPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
    MLIRContext *ctx;

    PropagateEventsPattern(MLIRContext *ctx)
        : OpRewritePattern<UnrealizedConversionCastOp>(ctx), ctx(ctx)
    {
    }

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                  PatternRewriter &rewriter) const override
    {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
            return failure();

        Type srcType = op.getInputs()[0].getType();
        Type dstType = op.getResult(0).getType();

        // Only match casts from quantum/ion types to event type
        // quantum.qreg -> event, quantum.qubit -> event, ion.qubit -> event
        bool validSrcType =
            llvm::isa<quantum::QubitType, quantum::QuregType, ion::QubitType>(srcType);
        bool validDstType = llvm::isa<rtio::EventType>(dstType);
        if (!validSrcType || !validDstType)
            return failure();

        Value input = op.getInputs()[0];

        // Find associated events
        // Skip over intermediate cast/extract/insert operations to collect events
        bool reachedAllocOp = false;
        SetVector<Value> events;
        traceValueWithCallback<TraceMode::Event>(input, [&](Value value) -> WalkResult {
            auto defOp = value.getDefiningOp();
            if (defOp &&
                isa<UnrealizedConversionCastOp, quantum::ExtractOp, quantum::InsertOp>(defOp)) {
                return WalkResult::advance();
            }

            // collect event and stop tracing this path
            if (isa<rtio::EventType>(value.getType())) {
                events.insert(value);
                return WalkResult::interrupt();
            }

            if (isa<quantum::AllocOp>(defOp)) {
                reachedAllocOp = true;
                return WalkResult::interrupt();
            }

            return WalkResult::advance();
        });

        if (reachedAllocOp && events.empty()) {
            auto eventType = rtio::EventType::get(ctx);
            Value emptyEvent = rtio::RTIOEmptyOp::create(rewriter, op.getLoc(), eventType);
            rewriter.replaceOp(op, emptyEvent);
            return success();
        }

        if (events.empty()) {
            op.emitError("No events found for cast op");
            return failure();
        }

        // Create a sync event from all collected events
        // TODO: check domination, so that we can avoid creating a sync event if events are
        // already dominated by one of the events
        Value syncEvent = awaitEvents(events.getArrayRef(), rewriter);
        rewriter.replaceOp(op, syncEvent);
        return success();
    }
};

/// Convert ion.measure_pulse to rtio.pulse marked with `_measurement` for ARTIQ lowering.
struct MeasurePulseToRTIOPattern : public OpConversionPattern<ion::MeasurePulseOp> {
    DenseMap<Value, Value> &qextractToMemrefMap;

    MeasurePulseToRTIOPattern(TypeConverter &typeConverter, MLIRContext *ctx,
                              DenseMap<Value, Value> &qextractToMemrefMap)
        : OpConversionPattern<ion::MeasurePulseOp>(typeConverter, ctx),
          qextractToMemrefMap(qextractToMemrefMap)
    {
    }

    LogicalResult matchAndRewrite(ion::MeasurePulseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();

        Value duration = op.getTime();
        Value freqValue = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(0.0));
        Value phaseValue = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(0.0));

        auto beamAttr = op.getBeam();
        int64_t transitionIndex = beamAttr.getTransitionIndex().getInt();
        ArrayAttr qualifiers = rewriter.getArrayAttr({rewriter.getI64IntegerAttr(transitionIndex)});
        IntegerAttr channelIdAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0); // TTL0
        auto channelType = rtio::ChannelType::get(ctx, "ttl", qualifiers, channelIdAttr);

        Value memrefLoadValue = nullptr;
        traceValueWithCallback<TraceMode::Qreg>(op.getInQubit(), [&](Value value) -> WalkResult {
            if (qextractToMemrefMap.count(value)) {
                memrefLoadValue = qextractToMemrefMap[value];
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        if (memrefLoadValue == nullptr) {
            op->emitError("Failed to trace the memref load value for measure_pulse");
            return failure();
        }

        Value channel = rtio::RTIOChannelOp::create(rewriter, loc, channelType);

        auto eventType = rtio::EventType::get(ctx);
        Value event = rtio::RTIOPulseOp::create(rewriter, loc, eventType, channel, duration,
                                                freqValue, phaseValue, nullptr);
        event.getDefiningOp()->setAttr("_measurement", rewriter.getUnitAttr());

        rewriter.replaceOp(op, event);
        return success();
    }
};

/// Convert ion.readout_bit to rtio.readout (measurement count).
struct ReadoutBitToRTIOPattern : public OpConversionPattern<ion::ReadoutBitOp> {
    ReadoutBitToRTIOPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<ion::ReadoutBitOp>(typeConverter, ctx)
    {
    }

    LogicalResult matchAndRewrite(ion::ReadoutBitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        rtio::RTIOPulseOp measPulse;
        for (Operation *cur = op->getPrevNode(); cur; cur = cur->getPrevNode()) {
            auto pulse = dyn_cast<rtio::RTIOPulseOp>(cur);
            if (pulse && pulse->hasAttr("_measurement")) {
                measPulse = pulse;
                break;
            }
        }
        if (!measPulse) {
            return op->emitError(
                "readout_bit: no preceding rtio.pulse(_measurement) in this block (expected "
                "measure -> readout ordering after parallelprotocol lowering)");
        }

        auto readout =
            rtio::RTIOReadoutOp::create(rewriter, loc, rewriter.getI32Type(), measPulse.getEvent());

        rewriter.replaceOp(op, ValueRange{op.getInQubit(), readout.getCount()});
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Populate functions
//===----------------------------------------------------------------------===//

void populateIonPulseToRTIOPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns,
                                    const IonInfo &ionInfo,
                                    DenseMap<Value, Value> &qextractToMemrefMap)
{
    patterns.add<PulseToRTIOPattern>(typeConverter, patterns.getContext(), ionInfo,
                                     qextractToMemrefMap);
}

void populateParallelProtocolToRTIOPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns)
{
    patterns.add<ParallelProtocolToRTIOPattern>(typeConverter, patterns.getContext());
}

void populateIonToRTIOFinalizePatterns(RewritePatternSet &patterns)
{
    patterns.add<PropagateEventsPattern>(patterns.getContext());
    // patterns.add<ResolveChannelMappingPattern>(patterns.getContext());
}

void populateIonMeasurePulseToRTIOPatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           DenseMap<Value, Value> &qextractToMemrefMap)
{
    patterns.add<MeasurePulseToRTIOPattern>(typeConverter, patterns.getContext(),
                                            qextractToMemrefMap);
}

void populateIonReadoutBitToRTIOPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<ReadoutBitToRTIOPattern>(typeConverter, patterns.getContext());
}

} // namespace ion
} // namespace catalyst
