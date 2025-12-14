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
#include "RTIO/Transforms/Patterns.h"

#include "ARTIQRuntimeBuilder.hpp"
#include "Utils.hpp"

using namespace mlir;
using namespace catalyst::rtio;

namespace catalyst {
namespace rtio {

#define GEN_PASS_DEF_RTIOEVENTTOARTIQPASS
#include "RTIO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

using ScheduleGroupsMap = DenseMap<int, llvm::SetVector<Operation *>>;
using GroupingPredicate =
    std::function<bool(rtio::RTIOPulseOp reference, rtio::RTIOPulseOp candidate)>;

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
        for (Operation *user : event.getUsers()) {
            auto pulse = dyn_cast<rtio::RTIOPulseOp>(user);
            if (!pulse || pulse.getWait() != event) {
                continue;
            }
            consumers.push_back(pulse);
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

            // check if event has already been processed
            // if not, process the event and insert it into the processed events
            if (!processedEvents.insert(event).second) {
                continue;
            }

            SmallVector<Value> newEvents = processEvent(event);
            llvm::append_range(worklist, newEvents);
        }
    }

    // return the next events to process
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
        SmallVector<rtio::RTIOPulseOp> boundaryConsumers;

        // Initial
        for (auto pulse : consumers) {
            if (processedPulses.contains(pulse)) {
                continue;
            }

            int32_t channel = extractChannelId(pulse.getChannel());
            if (canJoinGroup(pulse, channelPulses)) {
                channelPulses[channel].push_back(pulse);
                channelLastPulse[channel] = pulse;
            }
            else {
                if (!channelBoundary.count(channel)) {
                    channelBoundary[channel] = pulse;
                }
                boundaryConsumers.push_back(pulse);
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
                                       boundaryConsumers);
    }

    bool canJoinGroup(rtio::RTIOPulseOp cand,
                      const DenseMap<int32_t, SmallVector<rtio::RTIOPulseOp>> &channelPulses)
    {
        for (auto &[ch, pulses] : channelPulses) {
            if (!llvm::all_of(pulses, [&](auto pulse) { return groupingPredicate(pulse, cand); })) {
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
                    int32_t userChannel = extractChannelId(user.getChannel());
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
                            SmallVector<rtio::RTIOPulseOp> &boundaryConsumers)
    {
        if (channelPulses.size() > 1 && !channelBoundary.empty()) {
            return createSyncEvent(channelLastPulse, channelBoundary, boundaryConsumers);
        }
        return collectNextEvents(channelLastPulse, channelBoundary, boundaryConsumers);
    }

    SmallVector<Value> createSyncEvent(DenseMap<int32_t, rtio::RTIOPulseOp> &channelLastPulse,
                                       DenseMap<int32_t, rtio::RTIOPulseOp> &channelBoundary,
                                       SmallVector<rtio::RTIOPulseOp> &boundaryConsumers)
    {
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

        // Update boundaries and consumers
        for (auto &[_, pulse] : channelBoundary) {
            pulse.setWait(syncEvent);
        }
        for (auto pulse : boundaryConsumers) {
            pulse.setWait(syncEvent);
        }
        for (auto &entry : channelLastPulse) {
            rtio::RTIOPulseOp pulse = entry.second;
            for (auto user : pulseConsumers[pulse]) {
                auto userChannel = extractChannelId(user.getChannel());
                if (!channelBoundary.count(userChannel) || channelBoundary[userChannel] != user) {
                    if (user.getWait() == pulse.getEvent()) {
                        user.setWait(syncEvent);
                    }
                }
            }
        }

        return {syncEvent};
    }

    SmallVector<Value> collectNextEvents(DenseMap<int32_t, rtio::RTIOPulseOp> &channelLastPulse,
                                         DenseMap<int32_t, rtio::RTIOPulseOp> &channelBoundary,
                                         SmallVector<rtio::RTIOPulseOp> &boundaryConsumers)
    {
        SmallVector<Value> nextEvents;

        for (auto &entry : channelBoundary) {
            rtio::RTIOPulseOp pulse = entry.second;
            nextEvents.push_back(pulse.getWait());
        }
        if (!boundaryConsumers.empty() && !channelLastPulse.empty()) {
            rtio::RTIOPulseOp firstPulse = channelLastPulse.begin()->second;
            Value lastEvent = firstPulse.getEvent();
            for (auto pulse : boundaryConsumers) {
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

        // Simplify sync operations
        {
            RewritePatternSet patterns(&getContext());
            populateRTIOSyncSimplifyPatterns(patterns);
            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                module.emitError("Failed during sync simplification");
                return signalPassFailure();
            }
        }

        // Decompose frequency pulses
        for (auto &[funcOp, groups] : pulseGroups) {
            decomposeFrequencyPulses(groups);
        }

        // Sort blocks to fix dominance
        sortAllBlocks(module);

        // Decompose _frequency pulses into control + slack
        {
            RewritePatternSet patterns(&getContext());
            populateRTIOPulseDecomposePatterns(patterns);
            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                module.emitError("Failed during pulse decomposition");
                return signalPassFailure();
            }
        }

        // Setup device initialization
        if (failed(setupKernelDevice(module, builder))) {
            return signalPassFailure();
        }

        // Lowering to LLVM
        if (failed(lowerToLLVM(module))) {
            return signalPassFailure();
        }
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

    LogicalResult setupKernelDevice(ModuleOp module, OpBuilder &builder)
    {
        auto kernelFunc = module.lookupSymbol<func::FuncOp>(ARTIQFuncNames::kernel);
        if (!kernelFunc) {
            module.emitError("Cannot find ") << ARTIQFuncNames::kernel << " function";
            return failure();
        }

        OpBuilder::InsertionGuard guard(builder);

        // Ensure helper functions are defined in the module
        ARTIQRuntimeBuilder artiq(builder, kernelFunc);
        artiq.ensureHelperFunctions();

        builder.setInsertionPointToStart(&kernelFunc.getBody().front());
        artiq.rtioInit();

        // Set initial timeline: at_mu(rtio_get_counter() + slack)
        Value counter = artiq.rtioGetCounter();
        Value slack = artiq.constI64(ARTIQHardwareConfig::initSlackDelay);
        Value initialTime = builder.create<arith::AddIOp>(kernelFunc.getLoc(), counter, slack);
        artiq.atMu(initialTime);

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
        populateRTIOToARTIQConversionPatterns(typeConverter, patterns);

        ConversionTarget target(*ctx);
        target.addIllegalDialect<rtio::RTIODialect>();
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect, math::MathDialect,
                               scf::SCFDialect, func::FuncDialect>();

        return applyPartialConversion(module, target, std::move(patterns));
    }
};

} // namespace rtio
} // namespace catalyst
