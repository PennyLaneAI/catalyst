// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

constexpr int64_t UNKNOWN = ShapedType::kDynamic;
constexpr int32_t NO_POSTSELECT = -1;

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

/**
 * @brief Initialize and fill the  `struct Modifiers` on stack return a pointer to it.
 *
 * @param loc MLIR Location object
 * @param rewriter MLIR OpBuilder object
 * @param conv MLIR TypeConverter object
 * @param adjoint The value of adjoint flag of the resulting structure
 * @param controlledQubits list of controlled qubits
 * @param controlledValues list of controlled values
 */
Value getModifiersPtr(Location loc, OpBuilder &rewriter, const TypeConverter *conv, bool adjoint,
                      ValueRange controlledQubits, ValueRange controlledValues)
{
    assert(controlledQubits.size() == controlledValues.size() &&
           "controlled qubits and controlled values have different lengths");

    MLIRContext *ctx = rewriter.getContext();

    auto boolType = IntegerType::get(ctx, 1);
    auto sizeType = IntegerType::get(ctx, 64);

    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);

    if (!adjoint && controlledQubits.empty() && controlledValues.empty()) {
        return nullPtr;
    }

    auto adjointVal = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(adjoint));
    auto numControlledVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(controlledQubits.size()))
            .getResult();
    auto structType = LLVM::LLVMStructType::getLiteral(ctx, {boolType, sizeType, ptrType, ptrType});
    auto modifiersPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, structType, c1).getResult();
    auto adjointPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, structType, modifiersPtr,
                                                   llvm::ArrayRef<LLVM::GEPArg>{0, 0}, true);
    auto numControlledPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, structType, modifiersPtr,
                                                         llvm::ArrayRef<LLVM::GEPArg>{0, 1}, true);
    auto controlledWiresPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, structType, modifiersPtr, llvm::ArrayRef<LLVM::GEPArg>{0, 2}, true);
    auto controlledValuesPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, structType, modifiersPtr, llvm::ArrayRef<LLVM::GEPArg>{0, 3}, true);

    Value ctrlPtr = nullPtr;
    Value valuePtr = nullPtr;
    if (!controlledQubits.empty()) {
        ctrlPtr =
            rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, numControlledVal).getResult();
        valuePtr =
            rewriter.create<LLVM::AllocaOp>(loc, ptrType, boolType, numControlledVal).getResult();
        for (size_t i = 0; i < controlledQubits.size(); i++) {
            {
                auto itemPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, ptrType, ctrlPtr,
                                                            llvm::ArrayRef<LLVM::GEPArg>{i}, true);
                auto qubit = controlledQubits[i];
                rewriter.create<LLVM::StoreOp>(loc, qubit, itemPtr);
            }
            {
                auto itemPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, boolType, valuePtr,
                                                            llvm::ArrayRef<LLVM::GEPArg>{i}, true);
                auto value = controlledValues[i];
                rewriter.create<LLVM::StoreOp>(loc, value, itemPtr);
            }
        }
    }

    rewriter.create<LLVM::StoreOp>(loc, adjointVal, adjointPtr);
    rewriter.create<LLVM::StoreOp>(loc, numControlledVal, numControlledPtr);
    rewriter.create<LLVM::StoreOp>(loc, ctrlPtr, controlledWiresPtr);
    rewriter.create<LLVM::StoreOp>(loc, valuePtr, controlledValuesPtr);

    return modifiersPtr;
}

////////////////////////
// Runtime Management //
////////////////////////

template <typename T> struct RTBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = this->getContext();

        StringRef qirName;
        if constexpr (std::is_same_v<T, InitializeOp>) {
            qirName = "__catalyst__rt__initialize";
            Location loc = op->getLoc();
            ModuleOp mod = op->template getParentOfType<ModuleOp>();
            Type intPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
            Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                            /* seed = */ {intPtrType});
            Value seed_val;
            if (op->hasAttr("seed")) {
                IRRewriter::InsertPoint ip = rewriter.saveInsertionPoint();
                OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
                rewriter.setInsertionPointToStart(mod.getBody());
                LLVM::GlobalOp seed_glb = rewriter.create<LLVM::GlobalOp>(
                    loc, IntegerType::get(ctx, 32), true, LLVM::Linkage::Internal, "seed",
                    cast<IntegerAttr>(op->getAttr("seed")));
                rewriter.restoreInsertionPoint(ip);
                seed_val = rewriter.create<LLVM::AddressOfOp>(loc, seed_glb);
            }
            else {
                // Set seed argument to nullptr for unseeded runs
                seed_val = rewriter.create<LLVM::ZeroOp>(loc, intPtrType);
            }
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
            SmallVector<Value> operands = {seed_val};
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);
        }
        else {
            qirName = "__catalyst__rt__finalize";
            Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, ValueRange{});
        }

        return success();
    }
};

struct DeviceInitOpPattern : public OpConversionPattern<DeviceInitOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DeviceInitOp op, DeviceInitOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        StringRef qirName = "__catalyst__rt__device_init"; // (int8_t *, int8_t *, int8_t *) -> void

        Type charPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                        {/* rtd_lib = */ charPtrType,
                                                         /* rtd_name = */ charPtrType,
                                                         /* rtd_kwargs = */ charPtrType});
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        auto rtd_lib = op.getLib().str();
        auto rtd_name = op.getName().str();
        auto rtd_kwargs = op.getKwargs().str();

        auto rtd_lib_gs = getGlobalString(loc, rewriter, rtd_lib,
                                          StringRef(rtd_lib.c_str(), rtd_lib.length() + 1), mod);
        auto rtd_name_gs = getGlobalString(loc, rewriter, rtd_name,
                                           StringRef(rtd_name.c_str(), rtd_name.length() + 1), mod);
        auto rtd_kwargs_gs = getGlobalString(
            loc, rewriter, rtd_kwargs, StringRef(rtd_kwargs.c_str(), rtd_kwargs.length() + 1), mod);

        SmallVector<Value> operands = {rtd_lib_gs, rtd_name_gs, rtd_kwargs_gs};

        rewriter.create<LLVM::CallOp>(loc, fnDecl, operands);

        rewriter.eraseOp(op);

        return success();
    }
};

struct DeviceReleaseOpPattern : public OpConversionPattern<DeviceReleaseOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DeviceReleaseOp op, DeviceReleaseOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = this->getContext();

        StringRef qirName = "__catalyst__rt__device_release";

        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, ValueRange{});

        return success();
    }
};

///////////////////////
// Memory Management //
///////////////////////

struct AllocOpPattern : public OpConversionPattern<AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AllocOp op, AllocOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__catalyst__rt__qubit_allocate_array";
        Type qirSignature = LLVM::LLVMFunctionType::get(conv->convertType(QuregType::get(ctx)),
                                                        IntegerType::get(ctx, 64));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        Value nQubits = adaptor.getNqubits();
        if (!nQubits) {
            nQubits = rewriter.create<LLVM::ConstantOp>(loc, op.getNqubitsAttrAttr());
        }

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, nQubits);

        return success();
    }
};

struct DeallocOpPattern : public OpConversionPattern<DeallocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DeallocOp op, DeallocOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__catalyst__rt__qubit_release_array";
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                        conv->convertType(QuregType::get(ctx)));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, adaptor.getOperands());

        return success();
    }
};

struct ExtractOpPattern : public OpConversionPattern<ExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ExtractOp op, ExtractOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__catalyst__rt__array_get_element_ptr_1d";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMPointerType::get(rewriter.getContext()),
            {conv->convertType(QuregType::get(ctx)), IntegerType::get(ctx, 64)});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        Value index = adaptor.getIdx();
        if (!index) {
            index = rewriter.create<LLVM::ConstantOp>(loc, op.getIdxAttrAttr());
        }
        SmallVector<Value> operands = {adaptor.getQreg(), index};

        Value elemPtr = rewriter.create<LLVM::CallOp>(loc, fnDecl, operands).getResult();
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, conv->convertType(QubitType::get(ctx)),
                                                  elemPtr);

        return success();
    }
};

struct InsertOpPattern : public OpConversionPattern<InsertOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(InsertOp op, InsertOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Unravel use-def chain of quantum register values, converting back to reference semantics.
        rewriter.replaceOp(op, adaptor.getInQreg());
        return success();
    }
};

struct SetStateOpPattern : public OpConversionPattern<SetStateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SetStateOp op, SetStateOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        bool isVarArg = false;
        MLIRContext *ctx = rewriter.getContext();
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        auto func =
            mlir::LLVM::lookupOrCreateFn(moduleOp, "__catalyst__qis__SetState",
                                         {adaptor.getInState().getType()}, voidTy, isVarArg);

        SmallVector<Value> values;
        values.insert(values.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        Location loc = op.getLoc();
        rewriter.create<LLVM::CallOp>(loc, func, ValueRange{adaptor.getInState()});
        rewriter.replaceOp(op, values);
        return success();
    }
};

///////////////////
// Quantum Gates //
///////////////////

struct CustomOpPattern : public OpConversionPattern<CustomOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomOp op, CustomOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();

        const TypeConverter *conv = getTypeConverter();
        auto modifiersPtr = getModifiersPtr(loc, rewriter, conv, op.getAdjointFlag(),
                                            adaptor.getInCtrlQubits(), adaptor.getInCtrlValues());

        std::string qirName = "__catalyst__qis__" + op.getGateName().str();
        SmallVector<Type> argTypes;
        argTypes.insert(argTypes.end(), adaptor.getParams().getTypes().begin(),
                        adaptor.getParams().getTypes().end());
        argTypes.insert(argTypes.end(), adaptor.getInQubits().getTypes().begin(),
                        adaptor.getInQubits().getTypes().end());
        argTypes.insert(argTypes.end(), modifiersPtr.getType());
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argTypes);
        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        SmallVector<Value> args;
        args.insert(args.end(), adaptor.getParams().begin(), adaptor.getParams().end());
        args.insert(args.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        args.insert(args.end(), modifiersPtr);

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        SmallVector<Value> values;
        values.insert(values.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        values.insert(values.end(), adaptor.getInCtrlQubits().begin(),
                      adaptor.getInCtrlQubits().end());
        rewriter.replaceOp(op, values);

        return success();
    }
};

struct GlobalPhaseOpPattern : public OpConversionPattern<GlobalPhaseOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GlobalPhaseOp op, GlobalPhaseOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();
        auto modifiersPtr = getModifiersPtr(loc, rewriter, conv, op.getAdjointFlag(),
                                            adaptor.getInCtrlQubits(), adaptor.getInCtrlValues());

        std::string qirName = "__catalyst__qis__GlobalPhase";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx), {Float64Type::get(ctx), modifiersPtr.getType()});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        SmallVector<Value> args;
        args.insert(args.end(), adaptor.getParams());
        args.insert(args.end(), modifiersPtr);

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        rewriter.eraseOp(op);

        return success();
    }
};

struct MultiRZOpPattern : public OpConversionPattern<MultiRZOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(MultiRZOp op, MultiRZOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();
        auto modifiersPtr = getModifiersPtr(loc, rewriter, conv, op.getAdjointFlag(),
                                            adaptor.getInCtrlQubits(), adaptor.getInCtrlValues());

        std::string qirName = "__catalyst__qis__MultiRZ";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {Float64Type::get(ctx), modifiersPtr.getType(), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = op.getOutQubits().size();
        SmallVector<Value> args;
        args.insert(args.end(), adaptor.getTheta());
        args.insert(args.end(), modifiersPtr);
        args.insert(args.end(),
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        args.insert(args.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);

        SmallVector<Value> values;
        values.insert(values.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        values.insert(values.end(), adaptor.getInCtrlQubits().begin(),
                      adaptor.getInCtrlQubits().end());
        rewriter.replaceOp(op, values);

        return success();
    }
};

struct QubitUnitaryOpPattern : public OpConversionPattern<QubitUnitaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(QubitUnitaryOp op, QubitUnitaryOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();
        auto modifiersPtr = getModifiersPtr(loc, rewriter, conv, op.getAdjointFlag(),
                                            adaptor.getInCtrlQubits(), adaptor.getInCtrlValues());

        assert(isa<MemRefType>(op.getMatrix().getType()) &&
               "unitary must take in memref before lowering");

        Type matrixType = conv->convertType(
            MemRefType::get({UNKNOWN, UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));

        std::string qirName = "__catalyst__qis__QubitUnitary";
        Type qirSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                        {LLVM::LLVMPointerType::get(rewriter.getContext()),
                                         modifiersPtr.getType(), IntegerType::get(ctx, 64)},
                                        /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = adaptor.getInQubits().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        args.insert(args.begin() + 1, modifiersPtr);
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), matrixType, c1);
        rewriter.create<LLVM::StoreOp>(loc, adaptor.getMatrix(), args[0]);

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);

        SmallVector<Value> values;
        values.insert(values.end(), adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        values.insert(values.end(), adaptor.getInCtrlQubits().begin(),
                      adaptor.getInCtrlQubits().end());
        rewriter.replaceOp(op, values);

        return success();
    }
};

/////////////////
// Observables //
/////////////////

struct ComputationalBasisOpPattern : public OpConversionPattern<ComputationalBasisOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ComputationalBasisOp op, ComputationalBasisOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            op, conv->convertType(ObservableType::get(ctx)), adaptor.getQubits());

        return success();
    }
};

struct NamedObsOpPattern : public OpConversionPattern<NamedObsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(NamedObsOp op, NamedObsOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__catalyst__qis__NamedObs";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ObservableType::get(ctx)),
            {IntegerType::get(ctx, 64), conv->convertType(QubitType::get(ctx))});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        auto obsTypeInt = static_cast<uint32_t>(op.getType());
        Value obsType =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(obsTypeInt));
        SmallVector<Value> args = {obsType, adaptor.getQubit()};

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, args);

        return success();
    }
};

struct HermitianOpPattern : public OpConversionPattern<HermitianOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(HermitianOp op, HermitianOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        assert(isa<MemRefType>(op.getMatrix().getType()) &&
               "hermitian must take in memref before lowering");

        Type matrixType = conv->convertType(
            MemRefType::get({UNKNOWN, UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));

        StringRef qirName = "__catalyst__qis__HermitianObs";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ObservableType::get(ctx)),
            {LLVM::LLVMPointerType::get(rewriter.getContext()), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = op.getQubits().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), matrixType, c1);
        rewriter.create<LLVM::StoreOp>(loc, adaptor.getMatrix(), args[0]);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, args);

        return success();
    }
};

struct TensorOpPattern : public OpConversionPattern<TensorOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorOp op, TensorOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__catalyst__qis__TensorObs";
        Type qirSignature =
            LLVM::LLVMFunctionType::get(conv->convertType(ObservableType::get(ctx)),
                                        IntegerType::get(ctx, 64), /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numTerms = op.getTerms().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin(),
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numTerms)));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, args);

        return success();
    }
};

struct HamiltonianOpPattern : public OpConversionPattern<HamiltonianOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(HamiltonianOp op, HamiltonianOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        assert(isa<MemRefType>(op.getCoeffs().getType()) &&
               "hamiltonian must take in memref before lowering");

        Type vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));

        StringRef qirName = "__catalyst__qis__HamiltonianObs";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ObservableType::get(ctx)),
            {LLVM::LLVMPointerType::get(rewriter.getContext()), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numTerms = op.getTerms().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numTerms)));
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), vectorType, c1);
        rewriter.create<LLVM::StoreOp>(loc, adaptor.getCoeffs(), args[0]);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, args);

        return success();
    }
};

//////////////////
// Measurements //
//////////////////

struct MeasureOpPattern : public OpConversionPattern<MeasureOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(MeasureOp op, MeasureOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        // Add postselect and qubit types to the function signature
        Type qubitTy = conv->convertType(QubitType::get(ctx));
        Type postselectTy = IntegerType::get(ctx, 32);
        SmallVector<Type> argSignatures = {qubitTy, postselectTy};

        StringRef qirName = "__catalyst__qis__Measure";
        Type qirSignature =
            LLVM::LLVMFunctionType::get(conv->convertType(ResultType::get(ctx)), argSignatures);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // Create the postselect value. If not given, it defaults to NO_POSTSELECT
        LLVM::ConstantOp postselect = rewriter.create<LLVM::ConstantOp>(
            loc, op.getPostselect() ? op.getPostselectAttr()
                                    : rewriter.getI32IntegerAttr(NO_POSTSELECT));

        // Add qubit and postselect values as arguments of the CallOp
        SmallVector<Value> args = {adaptor.getInQubit(), postselect};

        Value resultPtr = rewriter.create<LLVM::CallOp>(loc, fnDecl, args).getResult();
        Value mres = rewriter.create<LLVM::LoadOp>(loc, IntegerType::get(ctx, 1), resultPtr);
        rewriter.replaceOp(op, {mres, adaptor.getInQubit()});

        return success();
    }
};

template <typename T> class SampleBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

  protected:
    Value performRewrite(ConversionPatternRewriter &rewriter, Type structType, StringRef qirName,
                         T op, typename T::Adaptor adaptor) const
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();

        Type qirSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                        {LLVM::LLVMPointerType::get(rewriter.getContext()),
                                         IntegerType::get(ctx, 64), IntegerType::get(ctx, 64)},
                                        /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // We need to handle the C ABI convention of passing the result memref
        // as a struct pointer in the first argument to the C function.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value structPtr = rewriter.create<LLVM::AllocaOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), structType, c1);

        // For now obtain the qubit values from an unrealized cast created by the
        // ComputationalBasisOp lowering. Improve this once the runtime interface changes to
        // accept observables for sample.
        assert(isa<UnrealizedConversionCastOp>(adaptor.getObs().getDefiningOp()));
        ValueRange qubits = adaptor.getObs().getDefiningOp()->getOperands();

        Value numShots = rewriter.create<LLVM::ConstantOp>(loc, op.getShotsAttr());
        Value numQubits =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(qubits.size()));
        SmallVector<Value> args = {structPtr, numShots, numQubits};
        args.insert(args.end(), qubits.begin(), qubits.end());

        if constexpr (std::is_same_v<T, SampleOp>) {
            rewriter.create<LLVM::StoreOp>(loc, adaptor.getInData(), structPtr);
        }
        else if constexpr (std::is_same_v<T, CountsOp>) {
            auto aStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
            auto bStruct =
                rewriter.create<LLVM::InsertValueOp>(loc, aStruct, adaptor.getInEigvals(), 0);
            auto cStruct =
                rewriter.create<LLVM::InsertValueOp>(loc, bStruct, adaptor.getInCounts(), 1);
            rewriter.create<LLVM::StoreOp>(loc, cStruct, structPtr);
        }

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);

        return structPtr;
    };
};

struct SampleOpPattern : public SampleBasedPattern<SampleOp> {
    using SampleBasedPattern::SampleBasedPattern;

    LogicalResult matchAndRewrite(SampleOp op, SampleOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type matrixType =
            conv->convertType(MemRefType::get({UNKNOWN, UNKNOWN}, Float64Type::get(ctx)));

        StringRef qirName = "__catalyst__qis__Sample";
        performRewrite(rewriter, matrixType, qirName, op, adaptor);
        rewriter.eraseOp(op);

        return success();
    }
};

struct CountsOpPattern : public SampleBasedPattern<CountsOp> {
    using SampleBasedPattern::SampleBasedPattern;

    LogicalResult matchAndRewrite(CountsOp op, CountsOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type vector1Type = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));
        Type vector2Type = conv->convertType(MemRefType::get({UNKNOWN}, IntegerType::get(ctx, 64)));
        Type structType = LLVM::LLVMStructType::getLiteral(ctx, {vector1Type, vector2Type});

        StringRef qirName = "__catalyst__qis__Counts";
        performRewrite(rewriter, structType, qirName, op, adaptor);
        rewriter.eraseOp(op);

        return success();
    }
};

template <typename T> struct StatsBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = this->getTypeConverter();

        StringRef qirName;
        if constexpr (std::is_same_v<T, ExpvalOp>) {
            qirName = "__catalyst__qis__Expval";
        }
        else {
            qirName = "__catalyst__qis__Variance";
        }

        Type qirSignature = LLVM::LLVMFunctionType::get(
            Float64Type::get(ctx), conv->convertType(ObservableType::get(ctx)));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, adaptor.getObs());

        return success();
    }
};

template <typename T> struct StateBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = this->getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type vectorType;
        StringRef qirName;
        if constexpr (std::is_same_v<T, ProbsOp>) {
            vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));
            qirName = "__catalyst__qis__Probs";
        }
        else {
            vectorType = conv->convertType(
                MemRefType::get({UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));
            qirName = "__catalyst__qis__State";
        }

        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {LLVM::LLVMPointerType::get(rewriter.getContext()), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // We need to handle the C ABI convention of passing the result memref
        // as a struct pointer in the first argument to the C function.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value structPtr = rewriter.create<LLVM::AllocaOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), vectorType, c1);
        rewriter.create<LLVM::StoreOp>(loc, adaptor.getStateIn(), structPtr);

        // For now obtain the qubit values from an unrealized cast created by the
        // ComputationalBasisOp lowering. Improve this once the runtime interface changes to
        // accept observables for sample.
        assert(isa<UnrealizedConversionCastOp>(adaptor.getObs().getDefiningOp()));
        ValueRange qubits = adaptor.getObs().getDefiningOp()->getOperands();

        SmallVector<Value> args = {structPtr};
        if constexpr (std::is_same_v<T, ProbsOp>) {
            Value numQubits =
                rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(qubits.size()));
            args.push_back(numQubits);
            args.insert(args.end(), qubits.begin(), qubits.end());
        }
        else {
            // __catalyst__qis__State does not support individual qubit measurements yet, so it must
            // be invoked without specific specific qubits (i.e. measure the whole register).
            Value numQubits = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
            args.push_back(numQubits);
        }

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateQIRConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<RTBasedPattern<InitializeOp>>(typeConverter, patterns.getContext());
    patterns.add<RTBasedPattern<FinalizeOp>>(typeConverter, patterns.getContext());
    patterns.add<DeviceInitOpPattern>(typeConverter, patterns.getContext());
    patterns.add<DeviceReleaseOpPattern>(typeConverter, patterns.getContext());
    patterns.add<AllocOpPattern>(typeConverter, patterns.getContext());
    patterns.add<DeallocOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ExtractOpPattern>(typeConverter, patterns.getContext());
    patterns.add<InsertOpPattern>(typeConverter, patterns.getContext());
    patterns.add<SetStateOpPattern>(typeConverter, patterns.getContext());
    patterns.add<CustomOpPattern>(typeConverter, patterns.getContext());
    patterns.add<MultiRZOpPattern>(typeConverter, patterns.getContext());
    patterns.add<GlobalPhaseOpPattern>(typeConverter, patterns.getContext());
    patterns.add<QubitUnitaryOpPattern>(typeConverter, patterns.getContext());
    patterns.add<MeasureOpPattern>(typeConverter, patterns.getContext());
    patterns.add<ComputationalBasisOpPattern>(typeConverter, patterns.getContext());
    patterns.add<NamedObsOpPattern>(typeConverter, patterns.getContext());
    patterns.add<HermitianOpPattern>(typeConverter, patterns.getContext());
    patterns.add<TensorOpPattern>(typeConverter, patterns.getContext());
    patterns.add<HamiltonianOpPattern>(typeConverter, patterns.getContext());
    patterns.add<SampleOpPattern>(typeConverter, patterns.getContext());
    patterns.add<CountsOpPattern>(typeConverter, patterns.getContext());
    patterns.add<StatsBasedPattern<ExpvalOp>>(typeConverter, patterns.getContext());
    patterns.add<StatsBasedPattern<VarianceOp>>(typeConverter, patterns.getContext());
    patterns.add<StateBasedPattern<ProbsOp>>(typeConverter, patterns.getContext());
    patterns.add<StateBasedPattern<StateOp>>(typeConverter, patterns.getContext());
}

} // namespace quantum
} // namespace catalyst
