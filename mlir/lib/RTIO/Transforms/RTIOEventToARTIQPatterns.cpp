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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "RTIO/IR/RTIOOps.h"
#include "RTIO/Transforms/Patterns.h"

#include "ARTIQRuntimeBuilder.hpp"
#include "Utils.hpp"

using namespace mlir;
using namespace catalyst::rtio;

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
        rewriter.create<LLVM::CallOp>(op.getLoc(), setFreqFunc,
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
        durationMu = rewriter.create<arith::MaxSIOp>(op.getLoc(), durationMu, minDuration);

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
            maxTime = rewriter.create<arith::MaxSIOp>(op.getLoc(), maxTime, events[i]);
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
        Value syncEvent = rewriter.create<RTIOSyncOp>(
            loc, eventType, ValueRange{controlPulse.getEvent(), slackPulse.getEvent()});

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
    patterns.add<SyncOpLowering, EmptyOpLowering, ChannelOpLowering, PulseOpLowering>(
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
