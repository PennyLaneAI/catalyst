// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

constexpr int64_t UNKNOWN = ShapedType::kDynamic;

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
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = rewriter.create<LLVM::GlobalOp>(
            loc, LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size()),
            true, LLVM::Linkage::Internal, key, rewriter.getStringAttr(value));
    }

    auto idx =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(rewriter.getContext(), 64),
                                          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    return rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)),
        rewriter.create<LLVM::AddressOfOp>(loc, glb), ArrayRef<Value>({idx, idx}));
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
            qirName = "__quantum__rt__initialize";
        }
        else {
            qirName = "__quantum__rt__finalize";
        }

        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, ValueRange{});

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

        StringRef qirName = "__quantum__rt__device_init"; // (int8_t *, int8_t *, int8_t *) -> void

        Type charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
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

        StringRef qirName = "__quantum__rt__device_release";

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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__rt__qubit_allocate_array";
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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__rt__qubit_release_array";
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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__rt__array_get_element_ptr_1d";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)),
            {conv->convertType(QuregType::get(ctx)), IntegerType::get(ctx, 64)});

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        Value index = adaptor.getIdx();
        if (!index) {
            index = rewriter.create<LLVM::ConstantOp>(loc, op.getIdxAttrAttr());
        }
        SmallVector<Value> operands = {adaptor.getQreg(), index};

        Value elemPtr = rewriter.create<LLVM::CallOp>(loc, fnDecl, operands).getResult();
        Value qubitPtr = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(conv->convertType(QubitType::get(ctx))), elemPtr);
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, qubitPtr);

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

        SmallVector<Type> argTypes(adaptor.getOperands().getTypes().begin(),
                                   adaptor.getOperands().getTypes().end());
        argTypes.insert(argTypes.end(), IntegerType::get(ctx, 1));

        std::string qirName = "__quantum__qis__" + op.getGateName().str();
        Type qirSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argTypes);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.end(), rewriter.create<LLVM::ConstantOp>(
                                    loc, rewriter.getBoolAttr(op.getAdjointFlag())));
        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        rewriter.replaceOp(op, adaptor.getInQubits());

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

        std::string qirName = "__quantum__qis__MultiRZ";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {Float64Type::get(ctx), IntegerType::get(ctx, 1), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = op.getNumResults();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        args.insert(args.begin() + 1, rewriter.create<LLVM::ConstantOp>(
                                          loc, rewriter.getBoolAttr(op.getAdjointFlag())));

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        rewriter.replaceOp(op, adaptor.getInQubits());

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
        TypeConverter *conv = getTypeConverter();

        assert(op.getMatrix().getType().isa<MemRefType>() &&
               "unitary must take in memref before lowering");

        Type matrixType = conv->convertType(
            MemRefType::get({UNKNOWN, UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));

        std::string qirName = "__quantum__qis__QubitUnitary";
        Type qirSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                        {LLVM::LLVMPointerType::get(matrixType),
                                         IntegerType::get(ctx, 1), IntegerType::get(ctx, 64)},
                                        /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = op.getNumResults();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        args.insert(args.begin() + 1, rewriter.create<LLVM::ConstantOp>(
                                          loc, rewriter.getBoolAttr(op.getAdjointFlag())));
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(matrixType), c1);
        rewriter.create<LLVM::StoreOp>(loc, adaptor.getMatrix(), args[0]);

        rewriter.create<LLVM::CallOp>(loc, fnDecl, args);
        rewriter.replaceOp(op, adaptor.getInQubits());

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
        TypeConverter *conv = getTypeConverter();

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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__qis__NamedObs";
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
        TypeConverter *conv = getTypeConverter();

        assert(op.getMatrix().getType().isa<MemRefType>() &&
               "hermitian must take in memref before lowering");

        Type matrixType = conv->convertType(
            MemRefType::get({UNKNOWN, UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));

        StringRef qirName = "__quantum__qis__HermitianObs";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ObservableType::get(ctx)),
            {LLVM::LLVMPointerType::get(matrixType), IntegerType::get(ctx, 64)}, /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numQubits = op.getQubits().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numQubits)));
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(matrixType), c1);
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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__qis__TensorObs";
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
        TypeConverter *conv = getTypeConverter();

        assert(op.getCoeffs().getType().isa<MemRefType>() &&
               "hamiltonian must take in memref before lowering");

        Type vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));

        StringRef qirName = "__quantum__qis__HamiltonianObs";
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ObservableType::get(ctx)),
            {LLVM::LLVMPointerType::get(vectorType), IntegerType::get(ctx, 64)}, /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        int64_t numTerms = op.getTerms().size();
        SmallVector<Value> args = adaptor.getOperands();
        args.insert(args.begin() + 1,
                    rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(numTerms)));
        // Replace the memref argument (LLVM struct) with a pointer to memref.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        args[0] = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
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
        TypeConverter *conv = getTypeConverter();

        StringRef qirName = "__quantum__qis__Measure";
        Type qirSignature = LLVM::LLVMFunctionType::get(conv->convertType(ResultType::get(ctx)),
                                                        conv->convertType(QubitType::get(ctx)));

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        Value resultPtr =
            rewriter.create<LLVM::CallOp>(loc, fnDecl, adaptor.getInQubit()).getResult();
        Value boolPtr = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 1)), resultPtr);
        Value mres = rewriter.create<LLVM::LoadOp>(loc, boolPtr);
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
                                        {LLVM::LLVMPointerType::get(structType),
                                         IntegerType::get(ctx, 64), IntegerType::get(ctx, 64)},
                                        /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // We need to handle the C ABI convention of passing the result memref
        // as a struct pointer in the first argument to the C function.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value structPtr =
            rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(structType), c1);

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
        TypeConverter *conv = getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type matrixType =
            conv->convertType(MemRefType::get({UNKNOWN, UNKNOWN}, Float64Type::get(ctx)));

        StringRef qirName = "__quantum__qis__Sample";
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
        TypeConverter *conv = getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type vector1Type = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));
        Type vector2Type = conv->convertType(MemRefType::get({UNKNOWN}, IntegerType::get(ctx, 64)));
        Type structType = LLVM::LLVMStructType::getLiteral(ctx, {vector1Type, vector2Type});

        StringRef qirName = "__quantum__qis__Counts";
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
        TypeConverter *conv = this->getTypeConverter();

        StringRef qirName;
        if constexpr (std::is_same_v<T, ExpvalOp>) {
            qirName = "__quantum__qis__Expval";
        }
        else {
            qirName = "__quantum__qis__Variance";
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
        TypeConverter *conv = this->getTypeConverter();

        if (!op.isBufferized())
            return op.emitOpError("op must be bufferized before lowering to LLVM");

        Type vectorType;
        StringRef qirName;
        if constexpr (std::is_same_v<T, ProbsOp>) {
            vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));
            qirName = "__quantum__qis__Probs";
        }
        else {
            vectorType = conv->convertType(
                MemRefType::get({UNKNOWN}, ComplexType::get(Float64Type::get(ctx))));
            qirName = "__quantum__qis__State";
        }

        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {LLVM::LLVMPointerType::get(vectorType), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

        // We need to handle the C ABI convention of passing the result memref
        // as a struct pointer in the first argument to the C function.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value structPtr =
            rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
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
            // __quantum__qis__State does not support individual qubit measurements yet, so it must
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
    patterns.add<CustomOpPattern>(typeConverter, patterns.getContext());
    patterns.add<MultiRZOpPattern>(typeConverter, patterns.getContext());
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
