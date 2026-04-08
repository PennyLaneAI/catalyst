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

#include "llvm/ADT/STLExtras.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

/// Schedule newly discovered pulse groups, set their wait to the current chain event and produce a
/// sync follow by the chain.
static llvm::SmallVector<Value>
schedulePulseGroups(ArrayRef<rtio::RTIOPulseOp> consumers, DenseSet<int64_t> &scheduled,
                    DenseMap<int64_t, SmallVector<rtio::RTIOPulseOp>> &pulseGroups, Value &chain,
                    OpBuilder &builder)
{
    auto eventType = rtio::EventType::get(builder.getContext());

    SmallVector<int64_t> newGroupIds;
    for (auto p : consumers) {
        int64_t gid = pulseGroupId(p);
        if (scheduled.insert(gid).second) {
            newGroupIds.push_back(gid);
        }
    }
    llvm::sort(newGroupIds);

    for (int64_t gid : newGroupIds) {
        auto &grp = pulseGroups[gid];
        for (auto p : grp) {
            p.setWait(chain);
        }

        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointAfter(grp.back());
        SmallVector<Value> evts;
        for (auto p : grp) {
            evts.push_back(p.getEvent());
        }
        chain = rtio::RTIOSyncOp::create(builder, grp.back().getLoc(), eventType, evts);
    }

    llvm::SmallVector<Value> nextEvents;
    for (auto p : consumers) {
        nextEvents.push_back(p.getEvent());
    }

    return nextEvents;
}

/// Schedule pulses for executing on ARTIQ
static void schedule(func::FuncOp funcOp, OpBuilder &builder)
{
    if (funcOp.getBody().empty()) {
        return;
    }
    Block &block = funcOp.getBody().front();

    // Collect pulses by _group
    DenseMap<int64_t, SmallVector<rtio::RTIOPulseOp>> pulseGroups;
    for (auto &op : block) {
        if (auto p = dyn_cast<rtio::RTIOPulseOp>(&op)) {
            pulseGroups[pulseGroupId(p)].push_back(p);
        }
    }
    if (pulseGroups.empty()) {
        return;
    }

    DenseSet<Value> visited;
    DenseSet<int64_t> scheduled;
    std::deque<Value> worklist;
    Value chain;

    for (auto &op : block) {
        if (auto empty = dyn_cast<rtio::RTIOEmptyOp>(&op)) {
            chain = empty.getResult();
            worklist.push_back(chain);
            break;
        }
    }
    if (!chain) {
        return;
    }

    while (!worklist.empty()) {
        Value ev = worklist.front();
        worklist.pop_front();
        if (!visited.insert(ev).second) {
            continue;
        }

        SmallVector<rtio::RTIOPulseOp> consumers;
        for (auto *user : ev.getUsers()) {
            if (auto p = dyn_cast<rtio::RTIOPulseOp>(user)) {
                if (p.getWait() == ev && p->getBlock() == &block) {
                    consumers.push_back(p);
                }
            }
            else if (auto s = dyn_cast<rtio::RTIOSyncOp>(user)) {
                if (s->getBlock() == &block) {
                    worklist.push_back(s.getSyncEvent());
                }
            }
        }

        auto nextEvents = schedulePulseGroups(consumers, scheduled, pulseGroups, chain, builder);
        llvm::append_range(worklist, nextEvents);
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

        // Schedule pulses
        module.walk([&](func::FuncOp funcOp) { schedule(funcOp, builder); });
        sortAllBlocks(module);

        // Simplify sync operations
        {
            RewritePatternSet patterns(&getContext());
            populateRTIOSyncSimplifyPatterns(patterns);
            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                module.emitError("Failed during sync simplification");
                return signalPassFailure();
            }
        }

        // Setup device initialization
        if (failed(setupKernelDevice(module, builder))) {
            return signalPassFailure();
        }

        // Assign unique RPC service IDs to all rtio.rpc ops if no predefined RPC IDs.
        (void)assignRPCIds(module);

        // Wire up measurement helpers before lowering
        if (failed(finalizeMeasurementResultHandling(module, builder))) {
            return signalPassFailure();
        }

        // Lowering to LLVM
        if (failed(lowerToLLVM(module))) {
            return signalPassFailure();
        }
    }

  private:
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

        ARTIQRuntimeBuilder artiq(builder, kernelFunc);

        // DDS/SPI helper functions are only needed when the module has pulse ops.
        // Skip them if there are no pulse ops to avoid requiring hardware device_db entries.
        bool hasPulses = false;
        module.walk([&](rtio::RTIOPulseOp) {
            hasPulses = true;
            return WalkResult::interrupt();
        });
        if (hasPulses) {
            artiq.ensureHelperFunctions();
        }

        builder.setInsertionPointToStart(&kernelFunc.getBody().front());
        artiq.rtioInit();

        // Set initial timeline: at_mu(rtio_get_counter() + slack)
        Value counter = artiq.rtioGetCounter();
        Value slack = artiq.constI64(ARTIQHardwareConfig::initSlackDelay);
        Value initialTime = arith::AddIOp::create(builder, kernelFunc.getLoc(), counter, slack);
        artiq.atMu(initialTime);

        // TTL6 scope trigger
        MeasurementChannelAddrs ch = ARTIQRuntimeBuilder::getMeasurementChannelAddresses(module);
        if (ch.acquisitionOutputAddr != 0) {
            Location loc = kernelFunc.getLoc();
            Value trigStart = arith::SubIOp::create(
                builder, loc, initialTime, artiq.constI64(ARTIQHardwareConfig::scopeTriggerLeadMu));
            artiq.atMu(trigStart);
            artiq.rtioOutput(artiq.constI32(ch.acquisitionOutputAddr), artiq.constI32(1));
            artiq.atMu(initialTime);
            artiq.rtioOutput(artiq.constI32(ch.acquisitionOutputAddr), artiq.constI32(0));
        }

        return success();
    }

    /// Wire up measurement-result helper functions created by IonToRTIO.
    /// Inserts into the kernel:
    ///   1. call @__rtio_init_dataset at the start
    ///   2. memref.store after each rtio.readout result
    ///   3. call @__rtio_transfer_measurement_results before return
    ///   4. wait_until_mu(now_mu()) after the transfer call
    static LogicalResult finalizeMeasurementResultHandling(ModuleOp module, OpBuilder &builder)
    {
        auto kernelFunc = module.lookupSymbol<func::FuncOp>(ARTIQFuncNames::kernel);
        if (!kernelFunc) {
            return success();
        }

        auto transferFunc =
            module.lookupSymbol<func::FuncOp>(ARTIQFuncNames::rtioTransferMeasurementResults);
        auto initFunc = module.lookupSymbol<func::FuncOp>(ARTIQFuncNames::rtioInitDataset);

        if (!transferFunc) {
            return success();
        }

        for (Block &block : kernelFunc.getBody()) {
            Operation *terminator = block.getTerminator();
            if (!terminator || !isa<func::ReturnOp>(terminator)) {
                continue;
            }

            Location loc = kernelFunc.getLoc();
            OpBuilder::InsertionGuard guard(builder);

            // 1. call @__rtio_init_dataset at the start
            if (initFunc) {
                builder.setInsertionPointToStart(&block);
                func::CallOp::create(builder, loc, initFunc, ValueRange{});
            }

            // Collect rtio.readout ops in block order (before they become __rtio_count).
            SmallVector<RTIOReadoutOp> readouts;
            for (Operation &op : block) {
                if (auto readout = dyn_cast<RTIOReadoutOp>(&op)) {
                    readouts.push_back(readout);
                }
            }

            if (readouts.empty()) {
                continue;
            }

            auto memrefType = cast<MemRefType>(transferFunc.getArgumentTypes()[0]);

            // Allocate the results buffer at the top of the block
            builder.setInsertionPointToStart(&block);
            Value alloc = memref::AllocaOp::create(builder, loc, memrefType);

            // 2. Store each readout count right after its defining op
            for (auto [i, readout] : llvm::enumerate(readouts)) {
                builder.setInsertionPointAfter(readout);
                Value idx = arith::ConstantIndexOp::create(builder, loc, static_cast<int64_t>(i));
                memref::StoreOp::create(builder, loc, readout.getCount(), alloc, ValueRange{idx});
            }

            // 3. call @__rtio_transfer_measurement_results + wait before return
            builder.setInsertionPoint(terminator);
            func::CallOp::create(builder, loc, transferFunc, ValueRange{alloc});

            // 4. wait_until_mu(now_mu()) so async RPCs flush before return
            ARTIQRuntimeBuilder artiq(builder, kernelFunc);
            artiq.waitUntilMu(artiq.nowMu());
        }

        return success();
    }

    // Walk every rtio.rpc in the module, assign a unique service ID to each distinct callee symbol,
    // and attach it as an IntegerAttr named "rpc_id".
    static llvm::SmallVector<std::pair<int32_t, std::string>> assignRPCIds(ModuleOp module)
    {
        llvm::DenseMap<mlir::StringAttr, int32_t> calleeToId;
        llvm::SmallVector<std::pair<int32_t, std::string>> idMap;
        int32_t nextId = 1;

        module.walk([&](rtio::RTIORPCOp rpc) {
            auto callee = mlir::StringAttr::get(module.getContext(),
                                                rpc.getCallee().getRootReference().getValue());

            auto [it, inserted] = calleeToId.try_emplace(callee, nextId);
            if (inserted) {
                idMap.push_back({nextId, callee.getValue().str()});
                nextId++;
            }
            rpc->setAttr("rpc_id",
                         mlir::IntegerAttr::get(mlir::IntegerType::get(module.getContext(), 32),
                                                it->second));
        });

        return idMap;
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
        populateRTIORPCConversionPatterns(typeConverter, patterns);

        ConversionTarget target(*ctx);
        target.addIllegalDialect<rtio::RTIODialect>();
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect, math::MathDialect,
                               scf::SCFDialect, func::FuncDialect>();

        return applyPartialConversion(module, target, std::move(patterns));
    }
};

} // namespace rtio
} // namespace catalyst
