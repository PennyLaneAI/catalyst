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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::ion;

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

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, StringRef value,
                      ModuleOp mod)
{
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
                 IntegerType::get(ctx, 64), // principal
                 Float64Type::get(ctx),     // spin
                 Float64Type::get(ctx),     // orbital
                 Float64Type::get(ctx),     // nuclear
                 Float64Type::get(ctx),     // spin_orbital
                 Float64Type::get(ctx),     // spin_orbital_nuclear
                 Float64Type::get(ctx),     // spin_orbital_nuclear_magnetization
                 Float64Type::get(ctx),     // energy
             });
}

Value createPositionStruct(Location loc, OpBuilder &rewriter, MLIRContext *ctx,
                           DenseIntElementsAttr &positionAttr)
{
    Type positionStructType =
        LLVM::LLVMStructType::getLiteral(ctx, {
                                                  IntegerType::get(ctx, 64), // x
                                                  IntegerType::get(ctx, 64), // y
                                                  IntegerType::get(ctx, 64), // z
                                              });
    Value positionStruct = rewriter.create<LLVM::UndefOp>(loc, positionStructType);
    int i = 0;
    for (auto posAttr : positionAttr) {
        mlir::Value pos = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), posAttr);
        positionStruct = rewriter.create<LLVM::InsertValueOp>(loc, positionStruct, pos, i);
        i++;
    }
    Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value positionStructPtr =
        rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                        /*elementType=*/positionStruct.getType(), c1);
    rewriter.create<LLVM::StoreOp>(loc, positionStruct, positionStructPtr);
    return positionStructPtr;
}

Value createLevelStruct(Location loc, OpBuilder &rewriter, MLIRContext *ctx, LevelAttr &levelAttr,
                        LLVM::LLVMStructType &levelStructType)
{
    Value levelStruct = rewriter.create<LLVM::UndefOp>(loc, levelStructType);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), levelAttr.getPrincipal()), 0);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getSpin()), 1);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getOrbital()), 2);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getNuclear()), 3);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getSpinOrbital()),
        4);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(),
                                          levelAttr.getSpinOrbitalNuclear()),
        5);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(),
                                          levelAttr.getSpinOrbitalNuclearMagnetization()),
        6);
    levelStruct = rewriter.create<LLVM::InsertValueOp>(
        loc, levelStruct,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), levelAttr.getEnergy()), 7);
    return levelStruct;
}
Value createLevelsArray(Location loc, OpBuilder &rewriter, MLIRContext *ctx, ArrayAttr &levelsAttr)
{
    LLVM::LLVMStructType levelStructType = createLevelStructType(ctx);
    Value levelsArray = rewriter.create<LLVM::UndefOp>(
        loc, LLVM::LLVMArrayType::get(levelStructType, levelsAttr.size()));

    for (size_t i = 0; i < levelsAttr.size(); ++i) {
        auto levelAttr = cast<LevelAttr>(levelsAttr[i]);
        Value levelStruct = createLevelStruct(loc, rewriter, ctx, levelAttr, levelStructType);
        levelsArray = rewriter.create<LLVM::InsertValueOp>(loc, levelsArray, levelStruct, i);
    }
    Value levelArraySize =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(levelsAttr.size()));
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value levelsArrayPtr =
        rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                        /*elementType=*/levelsArray.getType(), levelArraySize);
    rewriter.create<LLVM::StoreOp>(loc, levelsArray, levelsArrayPtr);
    return levelsArrayPtr;
}

Value createTransitionsArray(Location loc, OpBuilder &rewriter, MLIRContext *ctx,
                             ArrayAttr &transitionsAttr)
{
    LLVM::LLVMStructType levelStructType = createLevelStructType(ctx);
    LLVM::LLVMStructType TransitionStructType =
        LLVM::LLVMStructType::getLiteral(ctx, {
                                                  levelStructType,       // level_0
                                                  levelStructType,       // level_1
                                                  Float64Type::get(ctx), // einstein_a
                                              });
    Value TransitionsArray = rewriter.create<LLVM::UndefOp>(
        loc, LLVM::LLVMArrayType::get(TransitionStructType, transitionsAttr.size()));

    for (size_t i = 0; i < transitionsAttr.size(); ++i) {
        auto transitionAttr = cast<TransitionAttr>(transitionsAttr[i]);
        Value TransitionStruct = rewriter.create<LLVM::UndefOp>(loc, TransitionStructType);
        LevelAttr level0 = transitionAttr.getLevel_0();
        LevelAttr level1 = transitionAttr.getLevel_1();
        TransitionStruct = rewriter.create<LLVM::InsertValueOp>(
            loc, TransitionStruct, createLevelStruct(loc, rewriter, ctx, level0, levelStructType),
            0);
        TransitionStruct = rewriter.create<LLVM::InsertValueOp>(
            loc, TransitionStruct, createLevelStruct(loc, rewriter, ctx, level1, levelStructType),
            1);
        TransitionStruct = rewriter.create<LLVM::InsertValueOp>(
            loc, TransitionStruct,
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(),
                                              transitionAttr.getEinsteinA()),
            2);
        TransitionsArray =
            rewriter.create<LLVM::InsertValueOp>(loc, TransitionsArray, TransitionStruct, i);
    }

    Value TransitionsArraySize =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(transitionsAttr.size()));
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value TransitionsArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, /*resultType=*/ptrType,
        /*elementType=*/TransitionsArray.getType(), TransitionsArraySize);
    rewriter.create<LLVM::StoreOp>(loc, TransitionsArray, TransitionsArrayPtr);
    return TransitionsArrayPtr;
}

struct IonOpPattern : public OpConversionPattern<catalyst::ion::IonOp> {
    using OpConversionPattern<catalyst::ion::IonOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::IonOp op, catalyst::ion::IonOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        const TypeConverter *conv = getTypeConverter();

        Type IonTy = conv->convertType(IonType::get(ctx));
        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));

        // Extract relevant ion properties
        auto nameStr = op.getName().getValue().str();
        auto name = getGlobalString(loc, rewriter, nameStr,
                                    StringRef(nameStr.c_str(), nameStr.length() + 1), mod);
        auto mass = rewriter.create<LLVM::ConstantOp>(loc, op.getMass());
        auto charge = rewriter.create<LLVM::ConstantOp>(loc, op.getCharge());
        auto positionAttr = op.getPosition();
        auto levelsAttr = op.getLevels();
        auto transitionsAttr = op.getTransitions();

        Value positionStructPtr = createPositionStruct(loc, rewriter, ctx, positionAttr);

        Value levelsArrayPtr = createLevelsArray(loc, rewriter, ctx, levelsAttr);

        Value TransitionsArrayPtr = createTransitionsArray(loc, rewriter, ctx, transitionsAttr);

        // Define the function signature for the Ion stub
        Type ionStructType =
            LLVM::LLVMStructType::getLiteral(ctx, {
                                                      ptrType,               // name
                                                      Float64Type::get(ctx), // mass
                                                      Float64Type::get(ctx), // charge
                                                      ptrType,               // position
                                                      ptrType,               // levels
                                                      ptrType,               // Transitions
                                                  });

        // Create an instance of the Ion struct
        Value ionStruct = rewriter.create<LLVM::UndefOp>(loc, ionStructType);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, name, 0);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, mass, 1);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, charge, 2);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, positionStructPtr, 3);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, levelsArrayPtr, 4);
        ionStruct = rewriter.create<LLVM::InsertValueOp>(loc, ionStruct, TransitionsArrayPtr, 5);
        Value ionStructPtr = rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                                             /*elementType=*/ionStructType, c1);
        rewriter.create<LLVM::StoreOp>(loc, ionStruct, ionStructPtr);

        // Create the Ion stub function
        Type qirSignature = LLVM::LLVMFunctionType::get(IonTy, ptrType);
        std::string qirName = "__catalyst_ion";
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
        rewriter.create<LLVM::CallOp>(loc, fnDecl, ionStructPtr);

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
        return success();
    }
};

struct PulseOpPattern : public OpConversionPattern<catalyst::ion::PulseOp> {
    using OpConversionPattern<catalyst::ion::PulseOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(catalyst::ion::PulseOp op, catalyst::ion::PulseOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        return success();
    }
};

} // namespace

namespace catalyst {
namespace ion {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<IonOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ParallelProtocolOpPattern>(typeConverter, patterns.getContext());
    patterns.add<PulseOpPattern>(typeConverter, patterns.getContext());
}

} // namespace ion
} // namespace catalyst
