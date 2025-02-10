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

#include <nlohmann/json.hpp>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;
using json = nlohmann::json;

namespace {

LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, ModuleOp mod)
{
    StringRef value = StringRef(key.data(), key.size() + 1);
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size());
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, key,
                                              rewriter.getStringAttr(value));
    }
    return rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                                        type, rewriter.create<LLVM::AddressOfOp>(loc, glb),
                                        ArrayRef<LLVM::GEPArg>{0, 0}, true);
}

LLVM::LLVMStructType createLevelStructType(MLIRContext *ctx)
{
    return LLVM::LLVMStructType::getLiteral(
        ctx, {
                 LLVM::LLVMPointerType::get(ctx), // label
                 IntegerType::get(ctx, 64),       // principal
                 Float64Type::get(ctx),           // spin
                 Float64Type::get(ctx),           // orbital
                 Float64Type::get(ctx),           // nuclear
                 Float64Type::get(ctx),           // spin_orbital
                 Float64Type::get(ctx),           // spin_orbital_nuclear
                 Float64Type::get(ctx),           // spin_orbital_nuclear_magnetization
                 Float64Type::get(ctx),           // energy
             });
}

LLVM::LLVMStructType createBeamStructType(MLIRContext *ctx, OpBuilder &rewriter, BeamAttr &beamAttr)
{
    return LLVM::LLVMStructType::getLiteral(
        ctx, {
                 IntegerType::get(ctx, 64), // transition index
                 Float64Type::get(ctx),     // rabi
                 Float64Type::get(ctx),     // detuning
                 LLVM::LLVMArrayType::get(  // polarization
                     rewriter.getIntegerType(64), beamAttr.getPolarization().size()),
                 LLVM::LLVMArrayType::get( // wavevector
                     rewriter.getIntegerType(64), beamAttr.getWavevector().size()),
             });
}

Value createLevelStruct(Location loc, OpBuilder &rewriter, MLIRContext *ctx, ModuleOp &mod,
                        LevelAttr &levelAttr, LLVM::LLVMStructType &levelStructType)
{
    Value levelStruct = rewriter.create<LLVM::UndefOp>(loc, levelStructType);
    auto label = levelAttr.getLabel().getValue().str();
    auto labelGlobal = getGlobalString(loc, rewriter, label, mod);
    std::vector<Value> fieldValues = {
        labelGlobal,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), levelAttr.getPrincipal()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getSpin()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getOrbital()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getNuclear()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getSpinOrbital()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(),
                                          levelAttr.getSpinOrbitalNuclear()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(),
                                          levelAttr.getSpinOrbitalNuclearMagnetization()),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getEnergy())};
    for (size_t i = 0; i < fieldValues.size(); i++) {
        levelStruct = rewriter.create<LLVM::InsertValueOp>(loc, levelStruct, fieldValues[i], i);
    }
    return levelStruct;
}

Value createBeamStruct(Location loc, OpBuilder &rewriter, MLIRContext *ctx, BeamAttr &beamAttr)
{
    Type beamStructType = createBeamStructType(ctx, rewriter, beamAttr);

    auto transitionIndex = beamAttr.getTransitionIndex();
    auto rabi = beamAttr.getRabi();
    auto detuning = beamAttr.getDetuning();
    auto polarization = beamAttr.getPolarization().asArrayRef();
    auto wavevector = beamAttr.getWavevector().asArrayRef();

    Value beamStruct = rewriter.create<LLVM::UndefOp>(loc, beamStructType);
    beamStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, beamStruct, rewriter.create<LLVM::ConstantOp>(loc, transitionIndex), 0);
    beamStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, beamStruct, rewriter.create<LLVM::ConstantOp>(loc, rabi), 1);
    beamStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, beamStruct, rewriter.create<LLVM::ConstantOp>(loc, detuning), 2);
    for (size_t i = 0; i < polarization.size(); i++) {
        Value polarizaitonConst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64Type(),
            rewriter.getIntegerAttr(rewriter.getI64Type(), polarization[i]));
        beamStruct = rewriter.create<LLVM::InsertValueOp>(
            loc, beamStruct, polarizaitonConst, ArrayRef<int64_t>({3, static_cast<int64_t>(i)}));
    }
    for (size_t i = 0; i < wavevector.size(); i++) {
        Value waveConst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64Type(),
            rewriter.getIntegerAttr(rewriter.getI64Type(), wavevector[i]));
        beamStruct = rewriter.create<LLVM::InsertValueOp>(
            loc, beamStruct, waveConst, ArrayRef<int64_t>({4, static_cast<int64_t>(i)}));
    }
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    Value beamStructPtr = rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                                          /*elementType=*/beamStructType, c1);
    rewriter.create<LLVM::StoreOp>(loc, beamStruct, beamStructPtr);
    return beamStructPtr;
}

struct IonOpPattern : public OpConversionPattern<catalyst::ion::IonOp> {
    using OpConversionPattern<catalyst::ion::IonOp>::OpConversionPattern;

    // Create the ion JSON and pass it into the device kwargs as a JSON string
    LogicalResult matchAndRewrite(catalyst::ion::IonOp op, catalyst::ion::IonOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();

        DeviceInitOp deviceInitOp = *funcOp.getOps<DeviceInitOp>().begin();
        StringRef deviceKwargs = deviceInitOp.getKwargs();

        auto positionAttr = op.getPosition();
        auto levelsAttr = op.getLevels();
        auto transitionsAttr = op.getTransitions();

        json ion_json = R"({
  "class_": "Ion",
  "levels": [],
  "transitions" : []
})"_json;
        ion_json["mass"] = op.getMass().getValue().convertToDouble();
        ion_json["charge"] = op.getCharge().getValueAsDouble();

        assert(positionAttr.size() == 3 && "Position must have 3 coordinates!");
        std::array<double, 3> position = {positionAttr[0], positionAttr[1], positionAttr[2]};
        ion_json["position"] = position;

        DenseMap<StringRef, size_t> LevelLabel2Index;
        for (size_t i = 0; i < levelsAttr.size(); i++) {
            auto levelAttr = cast<LevelAttr>(levelsAttr[i]);

            json this_level =
                json{{"class_", "Level"},
                     {"principal", levelAttr.getPrincipal().getInt()},
                     {"spin", levelAttr.getSpin().getValue().convertToDouble()},
                     {"orbital", levelAttr.getOrbital().getValue().convertToDouble()},
                     {"nuclear", levelAttr.getNuclear().getValue().convertToDouble()},
                     {"spin_orbital", levelAttr.getSpinOrbital().getValue().convertToDouble()},
                     {"spin_orbital_nuclear",
                      levelAttr.getSpinOrbitalNuclear().getValue().convertToDouble()},
                     {"spin_orbital_nuclear_magnetization",
                      levelAttr.getSpinOrbitalNuclearMagnetization().getValue().convertToDouble()},
                     {"energy", levelAttr.getEnergy().getValue().convertToDouble()},
                     {"label", levelAttr.getLabel().getValue().str()}};

            ion_json["levels"].push_back(this_level);
            LevelLabel2Index[levelAttr.getLabel().getValue().str()] = i;
        }

        for (size_t i = 0; i < transitionsAttr.size(); i++) {
            auto transitionAttr = cast<TransitionAttr>(transitionsAttr[i]);

            json level1 =
                ion_json["levels"][LevelLabel2Index[transitionAttr.getLevel_0().getValue().str()]];
            json level2 =
                ion_json["levels"][LevelLabel2Index[transitionAttr.getLevel_1().getValue().str()]];

            json this_transition =
                json{{"class_", "Transition"},
                     {"einsteinA", transitionAttr.getEinsteinA().getValue().convertToDouble()},
                     {"level1", level1},
                     {"level2", level2}};
            ion_json["transitions"].push_back(this_transition);
        }
        deviceInitOp.setKwargs(deviceKwargs.str() + "ION:" + std::string(ion_json.dump()));

        rewriter.eraseOp(op);
        return success();
    }
};

struct ParallelProtocolOpPattern : public OpConversionPattern<catalyst::ion::ParallelProtocolOp> {
    using OpConversionPattern<catalyst::ion::ParallelProtocolOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::ParallelProtocolOp op,
                                  catalyst::ion::ParallelProtocolOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = getTypeConverter();

        // replace region args with parallelProtocolOp args
        Block *regionBlock = &op.getBodyRegion().front();
        assert((regionBlock->getNumArguments() == op.getNumOperands()) &&
               "ParallelProtocolOp should have the same number of arguments as its region");
        for (const auto &[regionArg, opArg] :
             llvm::zip(regionBlock->getArguments(), op.getOperands())) {
            regionArg.replaceAllUsesWith(opArg);
        }

        // Clone the region operations outside ParallelProtocolOp.
        SmallVector<Value> parallelPulses;
        rewriter.setInsertionPoint(op);
        IRMapping irMapping;
        for (auto &regionOp : regionBlock->getOperations()) {
            if (auto pulseOp = dyn_cast<catalyst::ion::PulseOp>(&regionOp)) {
                auto *clonedPulseOp = rewriter.clone(regionOp, irMapping);
                irMapping.map(regionOp.getResults(), clonedPulseOp->getResults());
                // keep track of parallel Pulses for the runtime call
                parallelPulses.push_back(clonedPulseOp->getResult(0));
            }
            else if (!isa<catalyst::ion::YieldOp>(&regionOp)) {
                // Clone other operations (e.g., llvm.fdiv) that aren't YieldOp
                auto *clonedRegionOp = rewriter.clone(regionOp, irMapping);
                irMapping.map(regionOp.getResults(), clonedRegionOp->getResults());
            }
        }

        // Create an array of pulses
        Type pulseArrayType =
            LLVM::LLVMArrayType::get(conv->convertType(PulseType::get(ctx)), parallelPulses.size());
        Value pulseArray = rewriter.create<LLVM::UndefOp>(loc, pulseArrayType);
        for (size_t i = 0; i < parallelPulses.size(); i++) {
            auto convertedPulse = rewriter
                                      .create<UnrealizedConversionCastOp>(
                                          loc, LLVM::LLVMPointerType::get(ctx), parallelPulses[i])
                                      .getResult(0);
            pulseArray = rewriter.create<LLVM::InsertValueOp>(loc, pulseArray, convertedPulse, i);
        }

        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value pulseArrayPtr = rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                                              /*elementType=*/pulseArrayType, c1);
        rewriter.create<LLVM::StoreOp>(loc, pulseArray, pulseArrayPtr);

        Value pulseArraySize = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(parallelPulses.size()));
        SmallVector<Value> operands;
        operands.push_back(pulseArrayPtr);
        operands.push_back(pulseArraySize);

        // Create the parallel protocol stub function
        Type protocolResultType = conv->convertType(IonType::get(ctx));
        Type protocolFuncType =
            LLVM::LLVMFunctionType::get(protocolResultType, {ptrType, pulseArraySize.getType()});
        std::string protocolFuncName = "__catalyst_parallel_protocol";
        LLVM::LLVMFuncOp protocolFnDecl =
            ensureFunctionDeclaration(rewriter, op, protocolFuncName, protocolFuncType);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, protocolFnDecl, operands);

        return success();
    }
};

struct PulseOpPattern : public OpConversionPattern<catalyst::ion::PulseOp> {
    using OpConversionPattern<catalyst::ion::PulseOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::PulseOp op, catalyst::ion::PulseOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = getTypeConverter();

        auto time = op.getTime();
        auto phase = rewriter.create<LLVM::ConstantOp>(loc, op.getPhase());
        Type qubitTy = conv->convertType(catalyst::quantum::QubitType::get(ctx));
        auto inQubit = adaptor.getInQubit();
        auto beamAttr = op.getBeam();

        Value beamStructPtr = createBeamStruct(loc, rewriter, ctx, beamAttr);

        SmallVector<Value> operands;
        operands.push_back(inQubit);
        operands.push_back(time);
        operands.push_back(phase);
        operands.push_back(beamStructPtr);

        // Create the Ion stub function
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(PulseType::get(ctx)),
            {conv->convertType(qubitTy), time.getType(), Float64Type::get(ctx),
             LLVM::LLVMPointerType::get(rewriter.getContext())});
        std::string qirName = "__catalyst_pulse";
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace ion {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<IonOpPattern>(typeConverter, patterns.getContext());
    patterns.add<PulseOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ParallelProtocolOpPattern>(typeConverter, patterns.getContext());
}

} // namespace ion
} // namespace catalyst
