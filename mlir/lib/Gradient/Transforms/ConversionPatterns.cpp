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

#include "iostream"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <string>
#include <vector>

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Gradient/Utils/CompDiffArgIndices.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;
using namespace catalyst::gradient;

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

struct AdjointOpPattern : public ConvertOpToLLVMPattern<AdjointOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(AdjointOp op, AdjointOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        TypeConverter *conv = getTypeConverter();

        Type vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));

        for (Type type : op.getResultTypes()) {
            if (!type.isa<MemRefType>())
                return op.emitOpError("must be bufferized before lowering");

            // Currently only expval gradients are supported by the runtime,
            // leading to tensor<?xf64> return values.
            if (type.dyn_cast<MemRefType>() != MemRefType::get({UNKNOWN}, Float64Type::get(ctx)))
                return op.emitOpError("adjoint can only return MemRef<?xf64> or tuple thereof");
        }

        // The callee of the adjoint op must return as a single result the quantum register.
        func::FuncOp callee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        assert(callee && callee.getNumResults() == 1 && "invalid qfunc symbol in adjoint op");

        StringRef cacheFnName = "__quantum__rt__toggle_recorder";
        StringRef gradFnName = "__quantum__qis__Gradient";
        Type cacheFnSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), IntegerType::get(ctx, 1));
        Type gradFnSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx), IntegerType::get(ctx, 64), /*isVarArg=*/true);

        LLVM::LLVMFuncOp cacheFnDecl =
            ensureFunctionDeclaration(rewriter, op, cacheFnName, cacheFnSignature);
        LLVM::LLVMFuncOp gradFnDecl =
            ensureFunctionDeclaration(rewriter, op, gradFnName, gradFnSignature);

        // Run the forward pass and cache the circuit.
        Value c_true = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getIntegerAttr(IntegerType::get(ctx, 1), 1));
        Value c_false = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getIntegerAttr(IntegerType::get(ctx, 1), 0));
        rewriter.create<LLVM::CallOp>(loc, cacheFnDecl, c_true);
        Value qreg = rewriter.create<func::CallOp>(loc, callee, op.getArgs()).getResult(0);
        if (!qreg.getType().isa<catalyst::quantum::QuregType>())
            return callee.emitOpError("qfunc must return quantum register");
        rewriter.create<LLVM::CallOp>(loc, cacheFnDecl, c_false);

        // We follow the C ABI convention of passing result memrefs as struct pointers in the
        // arguments to the C function, although in this case as a variadic argument list to allow
        // for a varying number of results in a single signature.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value numResults = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(op.getDataIn().size()));
        SmallVector<Value> args = {numResults};
        for (Value memref : adaptor.getDataIn()) {
            Value newArg =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, memref, newArg);
            args.push_back(newArg);
        }

        rewriter.create<LLVM::CallOp>(loc, gradFnDecl, args);
        rewriter.create<catalyst::quantum::DeallocOp>(loc, qreg);
        rewriter.eraseOp(op);

        return success();
    }
};

/// Options that configure preprocessing done on MemRefs before being passed to Enzyme.
struct EnzymeMemRefInterfaceOptions {
    /// Fill memref with zero values
    bool zeroOut = false;
    /// Mark memref as dupnoneed, allowing Enzyme to avoid computing its primal value.
    bool dupNoNeed = false;
};

static constexpr const char *enzyme_autodiff_func_name = "__enzyme_autodiff";
static constexpr const char *enzyme_allocation_key = "__enzyme_allocation_like";
static constexpr const char *enzyme_const_key = "enzyme_const";
static constexpr const char *enzyme_dupnoneed_key = "enzyme_dupnoneed";

struct BackpropOpPattern : public ConvertOpToLLVMPattern<BackpropOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(BackpropOp op, BackpropOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        if (llvm::any_of(op.getResultTypes(),
                         [](Type resultType) { return isa<TensorType>(resultType); })) {
            return op.emitOpError("must be bufferized before lowering");
        }

        // The callee of the backprop Op
        func::FuncOp callee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        assert(callee && "Expected a valid callee of type func.func");

        LowerToLLVMOptions options = getTypeConverter()->getOptions();
        if (options.useGenericFunctions) {
            LLVM::LLVMFuncOp allocFn = LLVM::lookupOrCreateGenericAllocFn(
                moduleOp, getTypeConverter()->getIndexType(), options.useOpaquePointers);
            LLVM::LLVMFuncOp freeFn =
                LLVM::lookupOrCreateGenericFreeFn(moduleOp, options.useOpaquePointers);

            // Register the previous functions as llvm globals (for Enzyme)
            // With the following piece of metadata, shadow memory is allocated with
            // _mlir_memref_to_llvm_alloc and shadow memory is freed with
            // _mlir_memref_to_llvm_free.
            insertEnzymeAllocationLike(rewriter, op->getParentOfType<ModuleOp>(), op.getLoc(),
                                       allocFn.getName(), freeFn.getName());

            // Register free
            // With the following piece of metadata, _mlir_memref_to_llvm_free's semantics are
            // stated to be equivalent to free.
            insertFunctionName(rewriter, op, "freename", StringRef("free", 5));
            insertEnzymeFunctionLike(rewriter, op, "__enzyme_function_like_free", "freename",
                                     freeFn.getName());
        }

        // Create the Enzyme function
        Type backpropFnSignature = LLVM::LLVMFunctionType::get(
            getEnzymeReturnType(ctx, op.getResultTypes()), {}, /*isVarArg=*/true);

        // We need to generate a new __enzyme_autodiff name per different function signature. One
        // way to do this is to append the number of scalar results to the name of the function.
        std::string autodiff_func_name =
            enzyme_autodiff_func_name + std::to_string(op.getNumResults());
        LLVM::LLVMFuncOp backpropFnDecl =
            ensureFunctionDeclaration(rewriter, op, autodiff_func_name, backpropFnSignature);

        // The first argument to Enzyme is a function pointer of the function to be differentiated
        Value calleePtr =
            rewriter.create<func::ConstantOp>(loc, callee.getFunctionType(), callee.getName());
        calleePtr = castToConvertedType(calleePtr, rewriter, loc);
        SmallVector<Value> callArgs = {calleePtr};

        std::vector<size_t> diffArgIndices = catalyst::compDiffArgIndices(op.getDiffArgIndices());
        getOrInsertEnzymeGlobal(rewriter, moduleOp, enzyme_const_key);
        getOrInsertEnzymeGlobal(rewriter, moduleOp, enzyme_dupnoneed_key);

        int index = 0;
        ValueRange dataIn = adaptor.getDataIn();
        Value enzymeConst = rewriter.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                                               enzyme_const_key);

        // Add the arguments and their shadow on data in
        for (Value arg : op.getArgs()) {
            std::vector<size_t>::iterator it =
                std::find(diffArgIndices.begin(), diffArgIndices.end(), index);
            if (it == diffArgIndices.end()) {
                if (isa<MemRefType>(arg.getType())) {
                    // unpackMemRef will handle the appropriate enzyme_const annotations
                    unpackMemRef(arg, /*shadow=*/nullptr, callArgs, rewriter, loc);
                }
                else {
                    callArgs.push_back(enzymeConst);
                    callArgs.push_back(castToConvertedType(arg, rewriter, loc));
                }
            }
            else if (isa<MemRefType>(arg.getType())) {
                size_t position = std::distance(diffArgIndices.begin(), it);
                unpackMemRef(arg, dataIn[position], callArgs, rewriter, loc, {.zeroOut = true});
            }
            else {
                assert(isa<FloatType>(arg.getType()) && "diff arg must be a float or float tensor");
                callArgs.push_back(arg);
            }
            index++;
        }

        for (Value qJacobian : op.getQuantumJacobian()) {
            // Enzyme requires buffers for the primal outputs, but we don't need their values.
            // We'll need to allocate space for them regardless, so marking them as dupNoNeed will
            // allow Enzyme to optimize away their computation.
            auto memrefType = cast<MemRefType>(qJacobian.getType());
            SmallVector<Value> dynamicDims;
            for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
                if (memrefType.isDynamicDim(dim)) {
                    Value dimIndex = rewriter.create<index::ConstantOp>(loc, dim);
                    dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, qJacobian, dimIndex));
                }
            }
            Value result = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicDims);

            unpackMemRef(result, qJacobian, callArgs, rewriter, loc, {.dupNoNeed = true});
        }

        // The results of backprop are in data in
        auto enzymeCall = rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, callArgs);
        SmallVector<Value> scalarResults;
        unpackScalarResults(enzymeCall, scalarResults, rewriter, loc);
        rewriter.replaceOp(op, scalarResults);
        return success();
    }

  private:
    static FlatSymbolRefAttr getOrInsertEnzymeGlobal(OpBuilder &builder, ModuleOp moduleOp,
                                                     const char *globalName)
    {
        // Copyright (C) 2023 - Jacob Mai Peng
        // https://github.com/pengmai/lagrad/blob/main/lib/LAGrad/LowerToLLVM.cpp
        auto *context = moduleOp.getContext();
        if (moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName)) {
            return SymbolRefAttr::get(context, globalName);
        }

        OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());
        auto shortTy = IntegerType::get(context, 8);
        builder.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                                       /*isConstant=*/true, LLVM::Linkage::Linkonce, globalName,
                                       IntegerAttr::get(shortTy, 0));
        return SymbolRefAttr::get(context, globalName);
    }

    Value castToConvertedType(Value value, OpBuilder &builder, Location loc) const
    {
        auto casted = builder.create<UnrealizedConversionCastOp>(
            loc, getTypeConverter()->convertType(value.getType()), value);
        return casted.getResult(0);
    }

    void unpackMemRef(Value memRefArg, Value shadowMemRef, SmallVectorImpl<Value> &callArgs,
                      OpBuilder &builder, Location loc,
                      EnzymeMemRefInterfaceOptions options = EnzymeMemRefInterfaceOptions()) const

    {
        auto llvmPtrType = LLVM::LLVMPointerType::get(builder.getContext());
        auto memRefType = cast<MemRefType>(memRefArg.getType());
        Value enzymeConst = builder.create<LLVM::AddressOfOp>(loc, llvmPtrType, enzyme_const_key);
        Value enzymeDupNoNeed =
            builder.create<LLVM::AddressOfOp>(loc, llvmPtrType, enzyme_dupnoneed_key);
        Value argStruct = castToConvertedType(memRefArg, builder, loc);
        MemRefDescriptor desc(argStruct);

        // Allocated pointer is always constant
        callArgs.push_back(enzymeConst);
        callArgs.push_back(desc.allocatedPtr(builder, loc));

        // Aligned pointer is active if a shadow is provided
        if (shadowMemRef) {
            if (options.dupNoNeed) {
                callArgs.push_back(enzymeDupNoNeed);
            }
            callArgs.push_back(desc.alignedPtr(builder, loc));
            Value shadowStruct = castToConvertedType(shadowMemRef, builder, loc);
            MemRefDescriptor shadowDesc(shadowStruct);
            Value shadowPtr = shadowDesc.alignedPtr(builder, loc);

            if (options.zeroOut) {
                Value bufferSizeBytes =
                    computeMemRefSizeInBytes(memRefType, shadowDesc, builder, loc);
                Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI8Type(), 0);
                builder.create<LLVM::MemsetOp>(loc, shadowPtr, zero, bufferSizeBytes,
                                               /*isVolatile=*/false);
            }
            callArgs.push_back(shadowPtr);
        }
        else {
            callArgs.push_back(enzymeConst);
            callArgs.push_back(desc.alignedPtr(builder, loc));
        }

        // Offsets, sizes, and strides
        callArgs.push_back(desc.offset(builder, loc));
        for (int64_t dim = 0; dim < memRefType.getRank(); dim++) {
            callArgs.push_back(desc.size(builder, loc, dim));
        }
        for (int64_t dim = 0; dim < memRefType.getRank(); dim++) {
            callArgs.push_back(desc.stride(builder, loc, dim));
        }
    }

    /// Determine the return type of the __enzyme_autodiff function based on the expected number of
    /// scalar returns.
    static Type getEnzymeReturnType(MLIRContext *ctx, TypeRange scalarReturns)
    {
        if (scalarReturns.empty()) {
            return LLVM::LLVMVoidType::get(ctx);
        }
        if (scalarReturns.size() == 1) {
            return scalarReturns.front();
        }
        return LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(scalarReturns));
    }

    static void unpackScalarResults(LLVM::CallOp enzymeCall, SmallVectorImpl<Value> &results,
                                    OpBuilder &builder, Location loc)
    {
        if (enzymeCall.getNumResults() == 0) {
            return;
        }

        Value result = enzymeCall.getResult();
        if (isa<FloatType>(result.getType())) {
            results.push_back(result);
        }
        if (auto structType = dyn_cast<LLVM::LLVMStructType>(result.getType())) {
            size_t numResults = structType.getBody().size();
            for (size_t i = 0; i < numResults; i++) {
                results.push_back(builder.create<LLVM::ExtractValueOp>(loc, result, i));
            }
        }
    }

    /// Compute the number of bytes required to store the array data of a general ranked MemRef.
    /// This is computed using the formula `element_size * (offset + sizes[0] * strides[0])`.
    /// For example, a rank-3 MemRef with shape [M, N, K] has sizes [M, N, K] and strides [N * K, K,
    /// 1]. The overall number of elements is M * N * K = sizes[0] * strides[0].
    Value computeMemRefSizeInBytes(MemRefType type, MemRefDescriptor descriptor, OpBuilder &builder,
                                   Location loc) const
    {
        Value bufferSize;
        Type indexType = getTypeConverter()->getIndexType();
        if (type.getRank() == 0) {
            bufferSize = builder.create<LLVM::ConstantOp>(loc, indexType, builder.getIndexAttr(1));
        }
        else {
            bufferSize = builder.create<LLVM::MulOp>(loc, descriptor.size(builder, loc, 0),
                                                     descriptor.stride(builder, loc, 0));
            bufferSize =
                builder.create<LLVM::AddOp>(loc, descriptor.offset(builder, loc), bufferSize);
        }
        Value elementByteSize = builder.create<LLVM::ConstantOp>(
            loc, indexType, builder.getIndexAttr(type.getElementTypeBitWidth() / 8));
        Value bufferSizeBytes = builder.create<LLVM::MulOp>(loc, elementByteSize, bufferSize);
        return bufferSizeBytes;
    }

    /// This registers a custom allocation and deallocation functions with Enzyme. It creates a
    /// global LLVM array that Enzyme will convert to the appropriate metadata using the
    /// `preserve-nvvm` pass.
    ///
    /// This functionality is described at:
    /// https://github.com/EnzymeAD/Enzyme/issues/930#issuecomment-1334502012
    LLVM::GlobalOp insertEnzymeAllocationLike(OpBuilder &builder, ModuleOp moduleOp, Location loc,
                                              StringRef allocFuncName, StringRef freeFuncName) const
    {
        MLIRContext *context = moduleOp.getContext();
        Type indexType = getTypeConverter()->getIndexType();
        OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto allocationLike = moduleOp.lookupSymbol<LLVM::GlobalOp>(enzyme_allocation_key);
        if (allocationLike) {
            return allocationLike;
        }

        auto ptrType = LLVM::LLVMPointerType::get(context);
        auto resultType = LLVM::LLVMArrayType::get(ptrType, 4);

        builder.create<LLVM::GlobalOp>(loc, LLVM::LLVMArrayType::get(builder.getI8Type(), 3), true,
                                       LLVM::Linkage::Linkonce, "dealloc_indices",
                                       builder.getStringAttr(StringRef("-1", 3)));
        allocationLike =
            builder.create<LLVM::GlobalOp>(loc, resultType,
                                           /*isConstant=*/false, LLVM::Linkage::External,
                                           enzyme_allocation_key, /*address space=*/nullptr);
        builder.createBlock(&allocationLike.getInitializerRegion());
        Value allocFn = builder.create<LLVM::AddressOfOp>(loc, ptrType, allocFuncName);
        Value sizeArgIndex =
            builder.create<LLVM::ConstantOp>(loc, indexType, builder.getIndexAttr(0));
        Value sizeArgIndexPtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, sizeArgIndex);
        Value deallocIndicesPtr =
            builder.create<LLVM::AddressOfOp>(loc, ptrType, "dealloc_indices");
        Value freeFn = builder.create<LLVM::AddressOfOp>(loc, ptrType, freeFuncName);

        Value result = builder.create<LLVM::UndefOp>(loc, resultType);
        result = builder.create<LLVM::InsertValueOp>(loc, result, allocFn, 0);
        result = builder.create<LLVM::InsertValueOp>(loc, result, sizeArgIndexPtr, 1);
        result = builder.create<LLVM::InsertValueOp>(loc, result, deallocIndicesPtr, 2);
        result = builder.create<LLVM::InsertValueOp>(loc, result, freeFn, 3);
        builder.create<LLVM::ReturnOp>(loc, result);

        return allocationLike;
    }

    static void insertFunctionName(PatternRewriter &rewriter, Operation *op, StringRef key,
                                   StringRef value)
    {
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        LLVM::GlobalOp glb = moduleOp.lookupSymbol<LLVM::GlobalOp>(key);
        if (!glb) {
            glb = rewriter.create<LLVM::GlobalOp>(
                moduleOp.getLoc(),
                LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size()),
                true, LLVM::Linkage::Linkonce, key, rewriter.getStringAttr(value));
        }
    }

    static LLVM::GlobalOp insertEnzymeFunctionLike(PatternRewriter &rewriter, Operation *op,
                                                   StringRef key, StringRef name,
                                                   StringRef originalName)
    {
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        auto *context = moduleOp.getContext();
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp glb = moduleOp.lookupSymbol<LLVM::GlobalOp>(key);
        if (glb) {
            return glb;
        }

        auto ptrType = LLVM::LLVMPointerType::get(context);
        glb = rewriter.create<LLVM::GlobalOp>(
            moduleOp.getLoc(), LLVM::LLVMArrayType::get(ptrType, 2), /*isConstant=*/false,
            LLVM::Linkage::External, key, nullptr);

        // Create the block and push it back in the global
        auto *contextGlb = glb.getContext();
        Block *block = new Block();
        glb.getInitializerRegion().push_back(block);
        rewriter.setInsertionPointToStart(block);

        auto llvmPtr = LLVM::LLVMPointerType::get(contextGlb);

        // Get original global name
        auto originalNameRefAttr = SymbolRefAttr::get(contextGlb, originalName);
        auto originalGlobal =
            rewriter.create<LLVM::AddressOfOp>(glb.getLoc(), llvmPtr, originalNameRefAttr);

        // Get global name
        auto nameRefAttr = SymbolRefAttr::get(contextGlb, name);
        auto enzymeGlobal = rewriter.create<LLVM::AddressOfOp>(glb.getLoc(), llvmPtr, nameRefAttr);

        auto undefArray =
            rewriter.create<LLVM::UndefOp>(glb.getLoc(), LLVM::LLVMArrayType::get(ptrType, 2));
        Value llvmInsert0 =
            rewriter.create<LLVM::InsertValueOp>(glb.getLoc(), undefArray, originalGlobal, 0);
        Value llvmInsert1 =
            rewriter.create<LLVM::InsertValueOp>(glb.getLoc(), llvmInsert0, enzymeGlobal, 1);
        rewriter.create<LLVM::ReturnOp>(glb.getLoc(), llvmInsert1);
        return glb;
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<AdjointOpPattern>(typeConverter);
    patterns.add<BackpropOpPattern>(typeConverter);
}

} // namespace gradient
} // namespace catalyst
