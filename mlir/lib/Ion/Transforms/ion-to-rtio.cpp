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

#include <llvm/ADT/STLExtras.h>
#include <queue>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ion/IR/IonDialect.h"
#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "RTIO/IR/RTIODialect.h"
#include "RTIO/IR/RTIOOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace ion {

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

enum class TraceMode {
    Qreg = 0,
    Event = 1,
};

/// Traces a Value backward through the IR by tracing its dataflow dependencies
/// across control flow and specific quantum operations.
///
/// Template Parameters:
///   - ModeT: TraceMode enum (Qreg or Event) that controls how quantum.insert
///            operations are handled
///            Qreg mode: Trace to find the source qreg of the given value
///            Event mode: Trace to find all events that contribute to the given value
///   - CallbackT: Callable type that will be invoked for each visited value.
///                May optionally return WalkResult for early termination.
///
/// Supported Operations:
///   - scf.for
///   - scf.if
///   - ion.parallelprotocol
///   - unrealized_conversion_cast
///   - quantum.extract
///   - quantum.insert
template <TraceMode ModeT, typename CallbackT>
auto traceValueWithCallback(Value value, CallbackT &&callback)
{
    WalkResult walkResult = WalkResult::advance();
    std::queue<Value> visited;
    visited.push(value);

    while (!visited.empty()) {
        Value value = visited.front();
        visited.pop();

        if constexpr (std::is_same_v<std::invoke_result_t<CallbackT, Value>, WalkResult>) {
            if (callback(value).wasInterrupted()) {
                walkResult = WalkResult::interrupt();
                continue;
            }
        }
        else {
            callback(value);
        }

        if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
            Block *block = arg.getOwner();
            Operation *parentOp = block->getParentOp();

            if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                unsigned argIndex = arg.getArgNumber();
                Value iterArg = forOp.getInitArgs()[argIndex - 1];
                visited.push(iterArg);
                continue;
            }
            else if (auto parallelProtocolOp = dyn_cast<ion::ParallelProtocolOp>(parentOp)) {
                unsigned argIndex = arg.getArgNumber();
                Value inQubit = parallelProtocolOp.getInQubits()[argIndex];
                visited.push(inQubit);
                continue;
            }
            parentOp->emitError("Unsupported parent operation for block argument: ") << value;
            llvm::reportFatalInternalError("Unsupported block argument");
        }

        Operation *defOp = value.getDefiningOp();
        if (defOp == nullptr) {
            continue;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            BlockArgument iterArg = forOp.getRegionIterArg(resultIdx);
            visited.push(iterArg);
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            Value thenValue = ifOp.thenYield().getOperand(resultIdx);
            Value elseValue = ifOp.elseYield().getOperand(resultIdx);
            visited.push(thenValue);
            visited.push(elseValue);
        }
        else if (auto parallelProtocolOp = dyn_cast<ion::ParallelProtocolOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            Value inQubit = parallelProtocolOp.getInQubits()[resultIdx];
            visited.push(inQubit);
        }
        else if (auto op = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
            visited.push(op.getInputs().front());
        }
        else if (auto op = dyn_cast<quantum::ExtractOp>(defOp)) {
            visited.push(op.getQreg());
        }
        else if (auto op = dyn_cast<quantum::InsertOp>(defOp)) {
            Value inQreg = op.getInQreg();
            Value qubit = op.getQubit();
            if constexpr (ModeT == TraceMode::Qreg) {
                visited.push(inQreg);
            }
            else if constexpr (ModeT == TraceMode::Event) {
                visited.push(qubit);
                // only trace qreg if it defined op is also come from insert op
                if (llvm::isa_and_present<quantum::InsertOp>(inQreg.getDefiningOp())) {
                    visited.push(inQreg);
                }
            }
        }
    }

    if constexpr (std::is_same_v<std::invoke_result_t<CallbackT, Value>, WalkResult>) {
        return walkResult;
    }
}

Value createSyncEvent(ArrayRef<Value> events, PatternRewriter &rewriter)
{
    if (events.size() == 1) {
        return events.front();
    }
    auto eventType = rtio::EventType::get(rewriter.getContext());
    return rewriter.create<rtio::RTIOSyncOp>(rewriter.getUnknownLoc(), eventType, events);
}

// Helper class to store ion information
class IonInfo {
  private:
    llvm::StringMap<double> levelEnergyMap;

    struct TransitionInfo {
        std::string level0;
        std::string level1;
        double einstein_a;
        std::string multipole;
    };
    SmallVector<TransitionInfo> transitions;

  public:
    IonInfo(ion::IonOp op)
    {
        auto levelAttrs = op.getLevels();
        auto transitionsAttr = op.getTransitions();

        // Map from Level label to Energy value
        for (auto levelAttr : levelAttrs) {
            auto level = cast<LevelAttr>(levelAttr);
            std::string label = level.getLabel().getValue().str();
            double energy = level.getEnergy().getValueAsDouble();
            levelEnergyMap[label] = energy;
        }

        // Store transition information
        for (auto transitionAttr : transitionsAttr) {
            auto transition = cast<TransitionAttr>(transitionAttr);
            TransitionInfo info;
            info.level0 = transition.getLevel_0().getValue().str();
            info.level1 = transition.getLevel_1().getValue().str();
            info.einstein_a = transition.getEinsteinA().getValueAsDouble();
            info.multipole = transition.getMultipole().getValue().str();
            transitions.push_back(info);
        }
    }

    // Get energy of a level by label
    std::optional<double> getLevelEnergy(StringRef label) const
    {
        auto it = levelEnergyMap.find(label.str());
        if (it != levelEnergyMap.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Get level label of a transition by index
    template <int IndexT>
    std::optional<double> getTransitionLevelEnergy(size_t transitionIndex) const
    {
        static_assert(IndexT == 0 || IndexT == 1, "IndexT must be 0 or 1");

        if (transitionIndex >= transitions.size()) {
            return std::nullopt;
        }

        const auto &transition = transitions[transitionIndex];
        if constexpr (IndexT == 0) {
            return getLevelEnergy(transition.level0);
        }
        else {
            return getLevelEnergy(transition.level1);
        }
    }

    // Get energy difference of a transition (level1 energy - level0 energy)
    std::optional<double> getTransitionEnergyDiff(size_t index) const
    {
        if (index >= transitions.size()) {
            return std::nullopt;
        }

        auto energy0 = getTransitionLevelEnergy<0>(index);
        auto energy1 = getTransitionLevelEnergy<1>(index);

        if (energy0.has_value() && energy1.has_value()) {
            return energy1.value() - energy0.value();
        }

        return std::nullopt;
    }

    // Get number of transitions
    size_t getNumTransitions() const { return transitions.size(); }

    // Get transition info by index
    std::optional<TransitionInfo> getTransition(size_t index) const
    {
        if (index < transitions.size()) {
            return transitions[index];
        }
        return std::nullopt;
    }
};

} // namespace

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

        rewriter.setInsertionPointAfter(op);

        // create events for each qubit
        auto events = llvm::map_range(inQubits, [&](Value qubit) {
            auto eventType = rtio::EventType::get(ctx);
            return rewriter.create<UnrealizedConversionCastOp>(loc, eventType, qubit).getResult(0);
        });

        Value inputSyncEvent = createSyncEvent(llvm::to_vector(events), rewriter);

        // Clone operations from the region to outside
        SmallVector<Value> pulseEvents;
        for (auto &regionOp : regionBlock->without_terminator()) {
            auto *clonedOp = rewriter.clone(regionOp, irMapping);
            if (auto pulseOp = dyn_cast<rtio::RTIOPulseOp>(clonedOp)) {
                // set wait event for the pulse operation
                pulseOp.setWait(inputSyncEvent);
                pulseEvents.push_back(pulseOp.getEvent());
            }
            irMapping.map(regionOp.getResults(), clonedOp->getResults());
        }

        // Create sync operation from pulse events (must have at least one after Phase 1)
        assert(pulseEvents.size() > 0 &&
               "must have at least one pulse operation after parallel protocol conversion");

        Value outputSyncEvent = createSyncEvent(llvm::to_vector(pulseEvents), rewriter);

        SmallVector<Value> results;
        for (Value result : op.getResults()) {
            // unrealized conversion cast sync event to result type
            auto event =
                rewriter.create<UnrealizedConversionCastOp>(loc, result.getType(), outputSyncEvent);
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
            rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(frequency));
        Value phaseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(phase));

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
        assert(memrefLoadValue != nullptr && "memrefLoadValue must not be null");

        Value channel =
            rewriter.create<rtio::RTIOQubitToChannelOp>(loc, channelType, memrefLoadValue);

        // Create rtio.pulse
        auto eventType = rtio::EventType::get(ctx);
        Value event = rewriter.create<rtio::RTIOPulseOp>(loc, eventType, channel, duration,
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

        int offset = 0;
        // If the qualifiers is not empty, get the first qualifier and check if it is 0 or 1
        if (qualifiers.size() >= 1) {
            IntegerAttr qualifier0 = llvm::dyn_cast<IntegerAttr>(qualifiers[0]);
            offset = qualifier0.getInt() == 0 ? 0 : 1;
        }

        IntegerAttr channelIdAttr = rewriter.getIntegerAttr(rewriter.getIndexType(),
                                                            channelIdValue.getSExtValue() + offset);

        auto resolvedChannelType =
            rtio::ChannelType::get(rewriter.getContext(), kind, qualifiers, channelIdAttr);

        Value channel = rewriter.create<rtio::RTIOChannelOp>(loc, resolvedChannelType);

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
            Value emptyEvent = rewriter.create<rtio::RTIOEmptyOp>(op.getLoc(), eventType);
            rewriter.replaceOp(op, emptyEvent);
            return success();
        }

        if (events.empty()) {
            op.emitError("No events found for cast op");
            llvm::reportFatalInternalError("No events found for cast op");
        }

        // Create a sync event from all collected events
        // TODO: check domination, so that we can avoid creating a sync event if events are
        // already dominated by one of the events
        Value syncEvent = createSyncEvent(events.getArrayRef(), rewriter);
        rewriter.replaceOp(op, syncEvent);
        return success();
    }
};

/// Clean up quantum/ion related ops that are not needed after conversion
struct CleanQuantumOpsPattern : public RewritePattern {
    CleanQuantumOpsPattern(MLIRContext *ctx)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx)
    {
    }

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override
    {
        Dialect *dialect = op->getDialect();
        if (!dialect || !isa<quantum::QuantumDialect, ion::IonDialect, memref::MemRefDialect,
                             linalg::LinalgDialect>(dialect))
            return failure();

        if (!op->use_empty()) {
            return failure();
        }

        rewriter.eraseOp(op);
        return success();
    }
};

LogicalResult CleanQuantumOps(func::FuncOp funcOp, MLIRContext *ctx)
{
    RewritePatternSet patterns(ctx);
    patterns.add<CleanQuantumOpsPattern>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return failure();
    }
    return success();
}

LogicalResult CanonicalizeKernelFunction(func::FuncOp funcOp, MLIRContext *ctx)
{
    RewritePatternSet patterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects()) {
        dialect->getCanonicalizationPatterns(patterns);
    }
    for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
        op.getCanonicalizationPatterns(patterns, ctx);
    }
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return failure();
    }
    return success();
}

LogicalResult ResolveChannelMapping(func::FuncOp funcOp, MLIRContext *ctx)
{
    RewritePatternSet patterns(ctx);
    patterns.add<ResolveChannelMappingPattern>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return failure();
    }
    return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_IONTORTIOPASS
#include "Ion/Transforms/Passes.h.inc"

struct IonToRTIOPass : public impl::IonToRTIOPassBase<IonToRTIOPass> {
    using impl::IonToRTIOPassBase<IonToRTIOPass>::IonToRTIOPassBase;

    LogicalResult IonPulseConversion(func::FuncOp funcOp, ConversionTarget &baseTarget,
                                     TypeConverter &typeConverter, IonInfo ionInfo,
                                     DenseMap<Value, Value> &qextractToMemrefMap, MLIRContext *ctx)
    {
        ConversionTarget target(baseTarget);
        target.addIllegalOp<ion::PulseOp>();

        RewritePatternSet patterns(ctx);
        patterns.add<PulseToRTIOPattern>(typeConverter, ctx, ionInfo, qextractToMemrefMap);
        if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
            return failure();
        }
        return success();
    }

    LogicalResult ParallelProtocolConversion(func::FuncOp funcOp, ConversionTarget &baseTarget,
                                             TypeConverter &typeConverter, MLIRContext *ctx)
    {
        ConversionTarget target(baseTarget);
        target.addIllegalOp<ion::ParallelProtocolOp>();

        RewritePatternSet patterns(ctx);
        patterns.add<ParallelProtocolToRTIOPattern>(typeConverter, ctx);
        if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
            return failure();
        }
        return success();
    }

    LogicalResult SCFStructuralConversion(func::FuncOp funcOp, ConversionTarget &target,
                                          TypeConverter &typeConverter, MLIRContext *ctx)
    {
        TypeConverter scfTypeConverter(typeConverter);
        scfTypeConverter.addConversion(
            [ctx](quantum::QubitType) -> Type { return rtio::EventType::get(ctx); });
        scfTypeConverter.addConversion(
            [ctx](quantum::QuregType) -> Type { return rtio::EventType::get(ctx); });
        scfTypeConverter.addConversion(
            [ctx](ion::QubitType) -> Type { return rtio::EventType::get(ctx); });
        // Add materialization for quantum/ion -> event
        scfTypeConverter.addSourceMaterialization(
            [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
                if (inputs.size() != 1)
                    return nullptr;
                Type inputType = inputs.front().getType();
                if (inputType != resultType) {
                    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                        .getResult(0);
                }
                return inputs[0];
            });
        // Add target materialization for event -> quantum/ion
        scfTypeConverter.addTargetMaterialization(
            [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
                if (inputs.size() != 1)
                    return nullptr;
                Type inputType = inputs.front().getType();
                if (inputType != resultType) {
                    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                        .getResult(0);
                }
                return inputs[0];
            });

        ConversionTarget scfTarget(getContext());
        scfTarget.addLegalDialect<func::FuncDialect, arith::ArithDialect, rtio::RTIODialect,
                                  LLVM::LLVMDialect, memref::MemRefDialect, linalg::LinalgDialect,
                                  BuiltinDialect, quantum::QuantumDialect, ion::IonDialect>();

        // Mark SCF ops as illegal only if they use quantum/ion types
        scfTarget.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
            for (auto arg : op.getRegionIterArgs()) {
                Type type = arg.getType();
                if (llvm::isa<quantum::QubitType, quantum::QuregType, ion::QubitType>(type)) {
                    return false;
                }
            }
            for (auto result : op.getResults()) {
                Type type = result.getType();
                if (llvm::isa<quantum::QubitType, quantum::QuregType, ion::QubitType>(type)) {
                    return false;
                }
            }
            return true;
        });

        scfTarget.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
            for (auto result : op.getResults()) {
                Type type = result.getType();
                if (llvm::isa<quantum::QubitType, quantum::QuregType, ion::QubitType>(type)) {
                    return false;
                }
            }
            return true;
        });

        scfTarget.addLegalOp<UnrealizedConversionCastOp>();

        // restructure SCF Operations
        RewritePatternSet scfPatterns(&getContext());
        mlir::scf::populateSCFStructuralTypeConversionsAndLegality(scfTypeConverter, scfPatterns,
                                                                   scfTarget);

        if (failed(applyPartialConversion(funcOp, scfTarget, std::move(scfPatterns)))) {
            return failure();
        }

        return success();
    }

    LogicalResult PropagateEvents(func::FuncOp funcOp, MLIRContext *ctx)
    {
        RewritePatternSet patterns(ctx);
        patterns.add<PropagateEventsPattern>(ctx);
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
            return failure();
        }
        return success();
    }

    SmallVector<IonInfo> getIonInfos()
    {
        SmallVector<IonInfo> ionInfos;
        getOperation()->walk([&](ion::IonOp ionOp) { ionInfos.emplace_back(IonInfo(ionOp)); });
        return ionInfos;
    }

    func::FuncOp createKernelFunction(func::FuncOp qnodeFunc, std::string kernelName,
                                      OpBuilder &builder)
    {
        MLIRContext *ctx = builder.getContext();

        auto newQnodeFunc = qnodeFunc.clone();
        newQnodeFunc.setName(kernelName);
        auto oldFuncType = qnodeFunc.getFunctionType();
        // create new function type with empty results
        auto newFuncType = FunctionType::get(ctx, oldFuncType.getInputs(), {});
        newQnodeFunc.setFunctionType(newFuncType);

        // Replace all return ops with empty returns
        SmallVector<func::ReturnOp> returnsToReplace;
        newQnodeFunc.walk([&](func::ReturnOp returnOp) { returnsToReplace.push_back(returnOp); });

        for (auto returnOp : returnsToReplace) {
            builder.setInsertionPoint(returnOp);
            builder.create<func::ReturnOp>(returnOp.getLoc());
            returnOp.erase();
        }

        return newQnodeFunc;
    }

    void initializeMemrefMap(func::FuncOp funcOp, ModuleOp module,
                             DenseMap<Value, Value> &qregToMemrefMap,
                             DenseMap<Value, Value> &qextractToMemrefMap, MLIRContext *ctx)
    {
        OpBuilder builder(ctx);

        int globalCounter = 0;
        funcOp.walk([&](quantum::AllocOp allocOp) {
            size_t numQubits = allocOp.getNqubitsAttr().value();
            auto memrefType =
                MemRefType::get({static_cast<int64_t>(numQubits)}, builder.getIndexType());

            // Create a unique symbol name for this global
            std::string globalNameStr = "__qubit_map_" + std::to_string(globalCounter++);
            StringRef globalName = globalNameStr;

            // Create dense attribute with values [0, 1, 2, ..., numQubits-1] * 2
            auto tensorType =
                RankedTensorType::get({static_cast<int64_t>(numQubits)}, builder.getIndexType());
            SmallVector<APInt> values;
            // Use IndexType::kInternalStorageBitWidth for index type
            unsigned indexWidth = IndexType::kInternalStorageBitWidth;
            for (size_t i = 0; i < numQubits; i++) {
                values.push_back(APInt(indexWidth, i * 2));
            }
            auto denseAttr = DenseIntElementsAttr::get(tensorType, values);

            // Create global memref at module level
            builder.setInsertionPointToStart(module.getBody());
            auto globalOp =
                memref::GlobalOp::create(builder, allocOp.getLoc(),
                                         builder.getStringAttr(globalName), // sym_name
                                         builder.getStringAttr("private"),  // sym_visibility
                                         TypeAttr::get(memrefType),         // type
                                         denseAttr,                         // initial_value
                                         builder.getUnitAttr(),             // constant
                                         IntegerAttr());                    // alignment

            // Get the global memref in the function
            builder.setInsertionPointAfter(allocOp);
            Value qubitMap = builder.create<memref::GetGlobalOp>(allocOp.getLoc(), memrefType,
                                                                 globalOp.getSymName());

            qregToMemrefMap[allocOp.getResult()] = qubitMap;
        });

        funcOp.walk([&](quantum::ExtractOp extractOp) {
            traceValueWithCallback<TraceMode::Qreg>(
                extractOp.getQreg(), [&](Value value) -> WalkResult {
                    if (qregToMemrefMap.count(value)) {
                        builder.setInsertionPointAfter(extractOp);
                        auto memref = qregToMemrefMap[value];

                        Value memrefLoadValue = nullptr;
                        if (Value idx = extractOp.getIdx()) {
                            // idx is an operand (i64), need to cast to index
                            Value indexValue = builder.create<arith::IndexCastOp>(
                                extractOp.getLoc(), builder.getIndexType(), idx);
                            memrefLoadValue = builder.create<memref::LoadOp>(
                                extractOp.getLoc(), memref, ValueRange{indexValue});
                        }
                        else if (IntegerAttr idxAttr = extractOp.getIdxAttrAttr()) {
                            Value indexValue = builder.create<arith::ConstantIndexOp>(
                                extractOp.getLoc(), idxAttr.getInt());
                            memrefLoadValue = builder.create<memref::LoadOp>(
                                extractOp.getLoc(), memref, ValueRange{indexValue});
                        }
                        if (memrefLoadValue) {
                            qextractToMemrefMap[extractOp.getResult()] = memrefLoadValue;
                        }
                        return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                });
        });
    }

    LogicalResult updateEntryFunction(func::FuncOp entryFunc, func::FuncOp newQnodeFunc,
                                      MLIRContext *ctx)
    {
        // Update entry function return type to empty
        auto oldEntryFuncType = entryFunc.getFunctionType();
        auto newEntryFuncType = FunctionType::get(ctx, oldEntryFuncType.getInputs(), {});
        entryFunc.setFunctionType(newEntryFuncType);

        // Clear the function body
        Block *entryBlock = &entryFunc.getBody().front();
        SmallVector<Operation *> opsToErase;
        for (Operation &op : entryBlock->getOperations()) {
            opsToErase.push_back(&op);
        }
        for (auto op : opsToErase) {
            op->dropAllUses();
            op->erase();
        }

        // Create call to kernel function
        OpBuilder entryBuilder(ctx);
        entryBuilder.setInsertionPointToStart(entryBlock);

        SmallVector<Value> kernelArgs(entryFunc.getArguments().begin(),
                                      entryFunc.getArguments().end());

        // compare args type with kernel function arguments
        if (kernelArgs.size() != newQnodeFunc.getArguments().size()) {
            entryFunc->emitError("Failed to update entry function: number of arguments mismatch");
            return failure();
        }
        for (size_t i = 0; i < kernelArgs.size(); i++) {
            if (kernelArgs[i].getType() != newQnodeFunc.getArguments()[i].getType()) {
                entryFunc->emitError("Failed to update entry function: argument type mismatch");
                return failure();
            }
        }

        entryBuilder.create<func::CallOp>(entryFunc.getLoc(), newQnodeFunc.getName(), TypeRange{},
                                          kernelArgs);

        entryBuilder.setInsertionPointToEnd(entryBlock);
        entryBuilder.create<func::ReturnOp>(entryFunc.getLoc());
        return success();
    }

    void runOnOperation() override
    {
        MLIRContext *ctx = &getContext();
        auto module = cast<ModuleOp>(getOperation());

        // check if there is only one qnode function
        func::FuncOp qnodeFunc = nullptr;
        int qnodeCounts = 0;
        module.walk([&](func::FuncOp funcOp) {
            if (funcOp->hasAttr("qnode")) {
                qnodeFunc = funcOp;
                qnodeCounts++;
            }
        });
        assert(qnodeCounts == 1 && "only one qnode function is allowed");

        // collect all ion information for calculating frequency when converting ion.pulse
        SmallVector<IonInfo> ionInfos = getIonInfos();
        if (ionInfos.empty()) {
            getOperation()->emitError("Failed to get ion information");
            return signalPassFailure();
        }

        // currently, we only support one ion information
        assert(ionInfos.size() == 1 && "only one ion information is allowed");
        IonInfo &ionInfo = ionInfos.front();

        // clone qnode function as new kernel function
        OpBuilder builder(ctx);
        func::FuncOp newQnodeFunc = createKernelFunction(qnodeFunc, kernelName, builder);
        module.insert(qnodeFunc, newQnodeFunc);

        // drop one of the pulse from the certain protocol
        // the way we handle the dropped pulse will be updated in the future
        newQnodeFunc.walk([&](ion::ParallelProtocolOp parallelProtocolOp) {
            parallelProtocolOp.walk([&](ion::PulseOp pulseOp) {
                if (pulseOp.getBeamAttr().getTransitionIndex().getInt() == 0) {
                    pulseOp.erase();
                }
            });
        });

        // Construct mapping from qreg alloc and qreg extract to memref
        // In the later conversion, we use the mapping to construct the channel for rtio.pulse
        DenseMap<Value, Value> qregToMemrefMap;
        DenseMap<Value, Value> qextractToMemrefMap;
        initializeMemrefMap(newQnodeFunc, module, qregToMemrefMap, qextractToMemrefMap, ctx);

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });
        typeConverter.addConversion(
            [&](ion::PulseType type) -> Type { return rtio::EventType::get(ctx); });

        ConversionTarget target(*ctx);
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        // prepare kernel function
        if (failed(IonPulseConversion(newQnodeFunc, target, typeConverter, ionInfo,
                                      qextractToMemrefMap, ctx)) ||
            failed(ParallelProtocolConversion(newQnodeFunc, target, typeConverter, ctx)) ||
            failed(SCFStructuralConversion(newQnodeFunc, target, typeConverter, ctx)) ||
            failed(PropagateEvents(newQnodeFunc, ctx)) ||
            failed(CleanQuantumOps(newQnodeFunc, ctx)) ||
            failed(CanonicalizeKernelFunction(newQnodeFunc, ctx))) {
            newQnodeFunc->emitError("Failed to convert to rtio dialect");
            return signalPassFailure();
        }

        // Resolve the static channel, the dynamic channel will be remained as `?`
        if (failed(ResolveChannelMapping(newQnodeFunc, ctx))) {
            newQnodeFunc->emitError("Failed to resolve channel mapping");
            return signalPassFailure();
        }

        // TODO: Naive scheduling to generate the simple Timeline RTIO IR
        // To shorten the `timeline`: `list scheduling`, `graph scheduling`, ... etc. can also be
        // used to schedule the operations. Here we just linearly mapping the operation based on the
        // order in the function without caring any latency issues.
        // if (failed(NaiveScheduling(newQnodeFunc, ctx))) {
        //     newQnodeFunc->emitError("Failed to schedule");
        //     return signalPassFailure();
        // }

        // remove body of entry function and add call to kernel function
        auto entryFunc = module.lookupSymbol<func::FuncOp>("jit_circuit");
        if (!entryFunc) {
            module.emitError("Cannot find entry function 'jit_circuit'");
            return signalPassFailure();
        }

        if (failed(updateEntryFunction(entryFunc, newQnodeFunc, ctx))) {
            module.emitError("Failed to update entry function");
            return signalPassFailure();
        }

        qnodeFunc->erase();
    }
};

} // namespace ion
} // namespace catalyst
