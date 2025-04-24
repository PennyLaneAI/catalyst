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

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;
using json = nlohmann::json;

namespace {

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

        std::map<std::string, size_t> LevelLabel2Index;
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

            std::string multipole = transitionAttr.getMultipole().getValue().str();
            std::string level0_label = transitionAttr.getLevel_0().getValue().str();
            std::string level1_label = transitionAttr.getLevel_1().getValue().str();

            assert(LevelLabel2Index.count(level0_label) == 1 &&
                   LevelLabel2Index.count(level1_label) == 1 &&
                   "A transition level's label must refer to an existing level in the ion!");

            const json &level1 = ion_json["levels"][LevelLabel2Index[level0_label]];
            const json &level2 = ion_json["levels"][LevelLabel2Index[level1_label]];

            json this_transition =
                json{{"class_", "Transition"},
                     {"einsteinA", transitionAttr.getEinsteinA().getValue().convertToDouble()},
                     {"level1", level1},
                     {"level2", level2},
                     {"label", level0_label + "->" + level1_label},
                     {"multipole", multipole}};
            ion_json["transitions"].push_back(this_transition);
        }
        deviceInitOp.setKwargs(deviceKwargs.str() + "ION:" + std::string(ion_json.dump()));

        deviceInitOp.setLib("oqd.qubit");

        rewriter.eraseOp(op);
        return success();
    }
};

struct ModesOpPattern : public OpConversionPattern<catalyst::ion::ModesOp> {
    using OpConversionPattern<catalyst::ion::ModesOp>::OpConversionPattern;

    // Create the modes JSON and pass it into the device kwargs as a JSON string
    LogicalResult matchAndRewrite(catalyst::ion::ModesOp op, catalyst::ion::ModesOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();

        DeviceInitOp deviceInitOp = *funcOp.getOps<DeviceInitOp>().begin();

        auto modesAttr = op.getModes();
        for (size_t i = 0; i < modesAttr.size(); i++) {
            StringRef deviceKwargs = deviceInitOp.getKwargs();
            auto phononAttr = cast<PhononAttr>(modesAttr[i]);

            json phonon_json = R"({
                            "class_": "Phonon",
                            "eigenvector" : []
                        })"_json;
            phonon_json["energy"] = phononAttr.getEnergy().getValue().convertToDouble();
            auto eigenvector = phononAttr.getEigenvector();
            for (int j = 0; j < eigenvector.size(); j++) {
                phonon_json["eigenvector"].push_back(eigenvector[j]);
            }
            deviceInitOp.setKwargs(deviceKwargs.str() +
                                   "PHONON:" + std::string(phonon_json.dump()));
        }

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
        Type protocolFuncType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                            {ptrType, pulseArraySize.getType()});
        std::string protocolFuncName = "__catalyst__oqd__ParallelProtocol";
        LLVM::LLVMFuncOp protocolFnDecl =
            catalyst::ensureFunctionDeclaration(rewriter, op, protocolFuncName, protocolFuncType);
        rewriter.create<LLVM::CallOp>(loc, protocolFnDecl, operands);

        SmallVector<Value> values;
        values.insert(values.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        rewriter.replaceOp(op, values);

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
        std::string qirName = "__catalyst__oqd__pulse";
        LLVM::LLVMFuncOp fnDecl =
            catalyst::ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
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
    patterns.add<ModesOpPattern>(typeConverter, patterns.getContext());
    patterns.add<PulseOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ParallelProtocolOpPattern>(typeConverter, patterns.getContext());
}

} // namespace ion
} // namespace catalyst
