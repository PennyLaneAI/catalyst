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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Ion/IR/IonDialect.h"
#include "Ion/Transforms/Patterns.h"
#include "Ion/Transforms/ValueTracing.h"
#include "Quantum/IR/QuantumDialect.h"
#include "RTIO/IR/RTIODialect.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace ion {

namespace {

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

        // Clone operations from the region to outside
        SmallVector<Value> pulseEvents;
        DenseMap<Value, int64_t> qubitToOffset;

        // we cache the channel to index mapping to avoid multiple lookups
        DenseMap<Value, Value> cache;
        for (auto &regionOp : regionBlock->without_terminator()) {
            auto *clonedOp = rewriter.clone(regionOp, irMapping);
            if (auto pulseOp = dyn_cast<rtio::RTIOPulseOp>(clonedOp)) {
                // set wait event for the pulse operation
                pulseOp.setWait(inputSyncEvent);

                Value index = nullptr;

                SmallVector<Value> chain;
                traceValueWithCallback<TraceMode::Qreg>(
                    pulseOp.getChannel(), [&](Value value) -> WalkResult {
                        if (cache.count(value)) {
                            index = cache[value];
                            return WalkResult::interrupt();
                        }
                        chain.push_back(value);
                        if (auto loadOp =
                                llvm::dyn_cast_if_present<memref::LoadOp>(value.getDefiningOp())) {
                            index = loadOp.getIndices()[0];

                            // cache the channel to index mapping
                            cache[pulseOp.getChannel()] = index;
                            return WalkResult::interrupt();
                        }
                        return WalkResult::advance();
                    });

                if (index == nullptr) {
                    op->emitError("Failed to trace the channel index");
                    return failure();
                }

                // update cache
                for (Value value : chain) {
                    cache[value] = index;
                }
                pulseOp->setAttr("offset", rewriter.getI64IntegerAttr(qubitToOffset[index]));

                // the same qubit may appear multiple times in the parallel protocol
                // so we need to increment the offset for each appearance
                qubitToOffset[index]++;

                pulseEvents.push_back(pulseOp.getEvent());
            }
            irMapping.map(regionOp.getResults(), clonedOp->getResults());
        }

        // Create sync operation from pulse events (must have at least one after Phase 1)
        assert(pulseEvents.size() > 0 &&
               "must have at least one pulse operation after parallel protocol conversion");

        Value outputSyncEvent = awaitEvents(llvm::to_vector(pulseEvents), rewriter);

        SmallVector<Value> results;
        for (Value result : op.getResults()) {
            // unrealized conversion cast sync event to result type
            auto event =
                UnrealizedConversionCastOp::create(rewriter, loc, result.getType(), outputSyncEvent);
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
/// %ch = rtio.qubit_to_channel %qubit : !ion.qubit -> !rtio.channel<"dds", ?>
/// ... // other pulse parameters settings
/// %event = rtio.pulse %ch duration(%duration) frequency(%freq) phase(%phase)
///     : !rtio.channel<"dds", ?> -> !rtio.event
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
        Value phaseValue = arith::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(phase));

        // Convert the qubit to a channel
        ArrayAttr qualifiers = rewriter.getArrayAttr({rewriter.getI64IntegerAttr(transitionIndex)});
        auto channelType = rtio::ChannelType::get(ctx, "dds", qualifiers, nullptr);

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

        Value channel =
            rtio::RTIOQubitToChannelOp::create(rewriter, loc, channelType, memrefLoadValue);

        // Create rtio.pulse
        auto eventType = rtio::EventType::get(ctx);
        Value event = rtio::RTIOPulseOp::create(rewriter, loc, eventType, channel, duration,
                                                         freqValue, phaseValue, nullptr);
        rewriter.replaceOp(op, event);

        return success();
    }
};

/// Resolve the static channel mapping for the rtio.qubit_to_channel operation
///
/// It's expecting `qubit_to_channel` has the following def-use chain:
/// memref.global w/ constants -> memref.get_global -> memref.load -> qubit_to_channel
///
/// Example:
/// ```
/// %ch = rtio.qubit_to_channel %qubit : !ion.qubit -> !rtio.channel<"dds", ?>
/// ```
/// will be converted to:
/// ```
/// %ch = rtio.channel "dds" { channel_id = 0 } : !rtio.channel<"dds">
/// ```
struct ResolveChannelMappingPattern : public OpRewritePattern<rtio::RTIOQubitToChannelOp> {
    ResolveChannelMappingPattern(MLIRContext *ctx)
        : OpRewritePattern<rtio::RTIOQubitToChannelOp>(ctx)
    {
    }

    LogicalResult matchAndRewrite(rtio::RTIOQubitToChannelOp op,
                                  PatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        Value qubit = op.getQubit();

        auto loadOp = qubit.getDefiningOp<memref::LoadOp>();
        if (!loadOp) {
            return failure();
        }

        Value memref = loadOp.getMemRef();
        auto getGlobalOp = memref.getDefiningOp<memref::GetGlobalOp>();
        if (!getGlobalOp) {
            return failure();
        }

        StringRef globalName = getGlobalOp.getName();
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module) {
            return failure();
        }
        auto globalOp = module.lookupSymbol<memref::GlobalOp>(globalName);
        if (!globalOp) {
            return failure();
        }

        auto initialValue = globalOp.getInitialValue();
        if (!initialValue) {
            return failure();
        }

        auto denseAttr = llvm::dyn_cast<DenseIntElementsAttr>(*initialValue);
        if (!denseAttr) {
            return failure();
        }

        ValueRange indices = loadOp.getIndices();
        if (indices.size() != 1) {
            return failure();
        }

        IntegerAttr indexAttr;
        if (!matchPattern(indices[0], m_Constant<IntegerAttr>(&indexAttr))) {
            return failure();
        }

        int64_t index = indexAttr.getInt();

        size_t denseSize = denseAttr.size();
        if (index < 0 || static_cast<size_t>(index) >= denseSize) {
            return failure();
        }

        APInt channelIdValue = denseAttr.getValues<APInt>()[index];

        auto originalChannelType = llvm::dyn_cast<rtio::ChannelType>(op.getChannel().getType());
        if (!originalChannelType) {
            return failure();
        }
        StringRef kind = originalChannelType.getKind();
        ArrayAttr qualifiers = originalChannelType.getQualifiers();

        // channel should have exactly one use before lowering to channel op
        assert(op.getChannel().hasOneUse() && "channel should have exactly one use");

        auto pulseOp = cast<rtio::RTIOPulseOp>(*op.getChannel().getUsers().begin());
        int64_t offset = cast<IntegerAttr>(pulseOp->getAttr("offset")).getInt();

        IntegerAttr channelIdAttr = rewriter.getIntegerAttr(
            rewriter.getIndexType(), (channelIdValue.getSExtValue() * 2 + offset));

        auto resolvedChannelType =
            rtio::ChannelType::get(rewriter.getContext(), kind, qualifiers, channelIdAttr);

        Value channel = rtio::RTIOChannelOp::create(rewriter, loc, resolvedChannelType);

        rewriter.replaceOp(op, channel);

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
    patterns.add<ResolveChannelMappingPattern>(patterns.getContext());
}

} // namespace ion
} // namespace catalyst
