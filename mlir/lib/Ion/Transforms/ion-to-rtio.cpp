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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Utils/JSONUtils.h"
#include "Ion/IR/IonDialect.h"
#include "Ion/IR/IonInfo.h"
#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"
#include "Ion/Transforms/ValueTracing.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "RTIO/IR/RTIODialect.h"
#include "RTIO/IR/RTIOOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace ion {

namespace {

/// Load a JSON file and convert it to an rtio.config attribute
FailureOr<rtio::ConfigAttr> loadDeviceDbAsConfig(MLIRContext *ctx, StringRef filePath)
{
    auto dictAttr = loadJsonFileAsDict(ctx, filePath);
    if (failed(dictAttr)) {
        return failure();
    }
    return rtio::ConfigAttr::get(ctx, *dictAttr);
}

/// Clean up unused quantum/ion/memref/linalg/builtin operations after conversion
/// Runs iteratively until no more ops can be erased
void cleanupUnusedOps(func::FuncOp funcOp)
{
    bool changed = true;
    while (changed) {
        changed = false;
        SmallVector<Operation *> toErase;
        funcOp.walk([&](Operation *op) {
            Dialect *dialect = op->getDialect();
            if (!dialect) {
                return;
            }

            // Include BuiltinDialect for unrealized_conversion_cast
            if (!isa<quantum::QuantumDialect, ion::IonDialect, memref::MemRefDialect,
                     linalg::LinalgDialect, BuiltinDialect>(dialect)) {
                return;
            }

            if (op->use_empty()) {
                toErase.push_back(op);
            }
        });

        for (Operation *op : toErase) {
            op->erase();
            changed = true;
        }
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_IONTORTIOPASS
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
        populateIonPulseToRTIOPatterns(typeConverter, patterns, ionInfo, qextractToMemrefMap);
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
        populateParallelProtocolToRTIOPatterns(typeConverter, patterns);
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

    LogicalResult FinalizeKernelFunction(func::FuncOp funcOp, MLIRContext *ctx)
    {
        RewritePatternSet patterns(ctx);
        for (auto *dialect : ctx->getLoadedDialects()) {
            dialect->getCanonicalizationPatterns(patterns);
        }
        for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
            op.getCanonicalizationPatterns(patterns, ctx);
        }
        populateIonToRTIOFinalizePatterns(patterns);
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
            return failure();
        }

        // Clean up unused quantum/ion/memref/linalg/builtin ops after patterns
        cleanupUnusedOps(funcOp);

        IRRewriter rewriter(ctx);
        DominanceInfo domInfo(funcOp);
        eliminateCommonSubExpressions(rewriter, domInfo, funcOp);

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

        // set public visibility and remove internal linkage for kernel function
        newQnodeFunc.setPublic();
        newQnodeFunc->removeAttr("llvm.linkage");

        // Clear operands from all return ops (make them return nothing)
        newQnodeFunc.walk([](func::ReturnOp returnOp) { returnOp.getOperandsMutable().clear(); });

        return newQnodeFunc;
    }

    void initializeMemrefMap(func::FuncOp funcOp, ModuleOp module,
                             DenseMap<Value, Value> &qregToMemrefMap,
                             DenseMap<Value, Value> &qextractToMemrefMap, MLIRContext *ctx)
    {
        OpBuilder builder(ctx);

        int globalCounter = 0;

        // create a global memref for each quantum.alloc op
        funcOp.walk([&](quantum::AllocOp allocOp) {
            size_t numQubits = allocOp.getNqubitsAttr().value();
            auto memrefType =
                MemRefType::get({static_cast<int64_t>(numQubits)}, builder.getIndexType());

            // Create a unique symbol name for this global
            std::string globalNameStr = "__qubit_map_" + std::to_string(globalCounter++);
            StringRef globalName = globalNameStr;

            // Create dense attribute with values [0, 1, 2, ..., numQubits-1]
            auto tensorType =
                RankedTensorType::get({static_cast<int64_t>(numQubits)}, builder.getIndexType());
            SmallVector<APInt> values;
            // Use IndexType::kInternalStorageBitWidth for index type
            unsigned indexWidth = IndexType::kInternalStorageBitWidth;
            for (size_t i = 0; i < numQubits; i++) {
                values.push_back(APInt(indexWidth, i));
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

    // In ARTIQ's compilation flow, we need to drop the pulse with transition 0 from the protocol
    void dropOnePulseFromProtocol(func::FuncOp funcOp)
    {
        SmallVector<ion::PulseOp> pulsesToErase;
        funcOp.walk([&](ion::PulseOp pulseOp) {
            if (pulseOp.getBeamAttr().getTransitionIndex().getInt() == 0) {
                pulsesToErase.push_back(pulseOp);
            }
        });
        for (auto pulseOp : pulsesToErase) {
            pulseOp.erase();
        }
    }

    void runOnOperation() override
    {
        MLIRContext *ctx = &getContext();
        auto module = cast<ModuleOp>(getOperation());

        // Load device_db JSON file and set rtio.config attribute on module
        if (!deviceDb.empty()) {
            auto configOrErr = loadDeviceDbAsConfig(ctx, deviceDb);
            if (failed(configOrErr)) {
                module->emitError("Failed to load device database from: ") << deviceDb;
                return signalPassFailure();
            }
            module->setAttr(rtio::ConfigAttr::getModuleAttrName(), *configOrErr);
        }

        // check if there is only one qnode function
        func::FuncOp qnodeFunc = nullptr;
        int qnodeCounts = 0;
        module.walk([&](func::FuncOp funcOp) {
            if (funcOp->hasAttr("qnode")) {
                qnodeFunc = funcOp;
                qnodeCounts++;
            }
        });

        if (qnodeCounts != 1) {
            getOperation()->emitError("only one qnode function is allowed");
            return signalPassFailure();
        }

        // collect all ion information for calculating frequency when converting ion.pulse
        SmallVector<IonInfo> ionInfos = getIonInfos();
        if (ionInfos.empty()) {
            getOperation()->emitError("Failed to get ion information");
            return signalPassFailure();
        }

        // currently, we only support one ion information
        if (ionInfos.size() != 1) {
            getOperation()->emitError("only one ion information is allowed");
            return signalPassFailure();
        }
        IonInfo &ionInfo = ionInfos.front();

        // clone qnode function as new kernel function
        OpBuilder builder(ctx);
        func::FuncOp newQnodeFunc = createKernelFunction(qnodeFunc, kernelName, builder);
        module.insert(qnodeFunc, newQnodeFunc);

        // drop one of the pulse from the certain protocol
        // the way we handle the dropped pulse will be updated in the future
        dropOnePulseFromProtocol(newQnodeFunc);

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
            failed(FinalizeKernelFunction(newQnodeFunc, ctx))) {
            newQnodeFunc->emitError("Failed to convert to rtio dialect");
            return signalPassFailure();
        }

        // remove other unused functions, only keep the kernel function
        for (auto funcOp : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
            if (funcOp.getName().str() != newQnodeFunc.getName().str()) {
                funcOp.erase();
            }
        }
    }
};

} // namespace ion
} // namespace catalyst
