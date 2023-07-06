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
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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

struct AdjointOpPattern : public OpConversionPattern<AdjointOp> {
    using OpConversionPattern::OpConversionPattern;

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
            auto newArg =
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

func::FuncOp genEnzymeWrapperFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);

    // Define the properties of the enzyme wrapper function.
    std::string fnName = callee.getName().str() + ".enzyme_wrapper";
    SmallVector<Type> argResTypes(callee.getArgumentTypes().begin(),
                                  callee.getArgumentTypes().end());
    argResTypes.insert(argResTypes.end(), callee.getResultTypes().begin(),
                       callee.getResultTypes().end());
    SmallVector<Type> originalArgTypes(callee.getArgumentTypes().begin(),
                                       callee.getArgumentTypes().end());
    SmallVector<Type> convertedTypes;

    // Create the wrapped operation
    FunctionType fnType = rewriter.getFunctionType(argResTypes, {});

    func::FuncOp wrappedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!wrappedCallee) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        wrappedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType);
        wrappedCallee.setPrivate();
        Block *entryBlock = wrappedCallee.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Get the arguments
        SmallVector<Value> callArgs(wrappedCallee.getArguments().begin(),
                                    wrappedCallee.getArguments().end() - callee.getNumResults());

        // Call the callee
        ValueRange results = rewriter.create<func::CallOp>(loc, callee, callArgs).getResults();
        ValueRange dpsOutputs = wrappedCallee.getArguments().drop_front(callee.getNumArguments());

        // Store the results
        for (auto [result, dpsOut] : llvm::zip(results, dpsOutputs)) {
            assert(result.getType() == dpsOut.getType() &&
                   "unexpected type mismatch between enzyme_wrapper destination-passing style "
                   "output and callee result");
            rewriter.create<memref::CopyOp>(loc, /*source=*/result, /*target=*/dpsOut);
        }

        // Add return op
        rewriter.create<func::ReturnOp>(loc);
    }

    return wrappedCallee;
}

struct BackpropOpPattern : public OpConversionPattern<BackpropOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(BackpropOp op, BackpropOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        LLVMTypeConverter llvmTypeConverter(ctx);
        auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);

        for (Type type : op.getResultTypes()) {
            if (!type.isa<MemRefType>())
                return op.emitOpError("must be bufferized before lowering");
        }

        // The callee of the backprop Op
        func::FuncOp callee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        assert(callee && "Expected a valid callee of type func.func");

        // Create mlir memref to llvm alloc
        StringRef allocFnName = "_mlir_memref_to_llvm_alloc";
        Type allocFnSignature = LLVM::LLVMFunctionType::get(llvmPtrType, {rewriter.getI64Type()});
        ensureFunctionDeclaration(rewriter, op, allocFnName, allocFnSignature);

        // Create mlir memref to llvm free
        StringRef freeFnName = "_mlir_memref_to_llvm_free";
        Type freeFnSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {llvmPtrType});
        ensureFunctionDeclaration(rewriter, op, freeFnName, freeFnSignature);

        // Register the previous functions as llvm globals (for Enzyme)
        insertEnzymeAllocationLike(rewriter, op->getParentOfType<ModuleOp>(), op.getLoc(),
                                   allocFnName, freeFnName);

        // Create the Enzyme function
        StringRef backpropFnName = "__enzyme_autodiff";
        Type backpropFnSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {}, /*isVarArg=*/true);

        LLVM::LLVMFuncOp backpropFnDecl =
            ensureFunctionDeclaration(rewriter, op, backpropFnName, backpropFnSignature);

        // Generate the wrapper function for argmapfn
        func::FuncOp wrapper = genEnzymeWrapperFunction(rewriter, loc, callee);

        // Set up the first argument and transforms the wrapper value in ptr
        FunctionType wrapperType = wrapper.getFunctionType();
        Value wrapperValue =
            rewriter.create<func::ConstantOp>(loc, wrapperType, wrapper.getName()).getResult();
        Type wrapperPtrType = llvmTypeConverter.convertType(wrapperType);
        Value wrapperPtr =
            rewriter.create<UnrealizedConversionCastOp>(loc, wrapperPtrType, wrapperValue)
                .getResult(0);
        // Add the pointer to the wrapped callee
        SmallVector<Value> callArgs = {wrapperPtr};

        std::vector<size_t> diffArgIndices = catalyst::compDiffArgIndices(op.getDiffArgIndices());
        auto const_global = getOrInsertEnzymeGlobal(rewriter, op, "enzyme_const");
        auto enzymeConst =
            rewriter.create<LLVM::AddressOfOp>(op->getLoc(), llvmPtrType, const_global);
        auto const_dupnoneed = getOrInsertEnzymeGlobal(rewriter, op, "enzyme_dupnoneed");
        auto enzymeDupNoNeed =
            rewriter.create<LLVM::AddressOfOp>(op->getLoc(), llvmPtrType, const_dupnoneed);

        int index = 0;
        ValueRange dataIn = adaptor.getDataIn();
        auto unpackMemRef = [&](Value memrefArg, Value shadowMemRef,
                                SmallVectorImpl<Value> &callArgs, OpBuilder &builder, Location loc,
                                bool zeroOut = false, bool dupNoNeed = false) {
            auto memrefType = cast<MemRefType>(memrefArg.getType());
            Value argStruct =
                builder
                    .create<UnrealizedConversionCastOp>(
                        loc, llvmTypeConverter.convertType(memrefArg.getType()), memrefArg)
                    .getResult(0);
            MemRefDescriptor desc(argStruct);

            // Allocated pointer is always constant
            callArgs.push_back(enzymeConst);
            callArgs.push_back(desc.allocatedPtr(builder, loc));

            // Aligned pointer is active if a shadow is provided
            if (shadowMemRef) {
                if (dupNoNeed) {
                    callArgs.push_back(enzymeDupNoNeed);
                }
                callArgs.push_back(desc.alignedPtr(builder, loc));
                Value shadowStruct =
                    builder
                        .create<UnrealizedConversionCastOp>(
                            loc, llvmTypeConverter.convertType(shadowMemRef.getType()),
                            shadowMemRef)
                        .getResult(0);
                MemRefDescriptor shadowDesc(shadowStruct);
                Value shadowPtr = shadowDesc.alignedPtr(builder, loc);

                if (zeroOut) {
                    Value bufferSizeBytes =
                        this->computeMemRefSizeInBytes(memrefType, shadowDesc, builder, loc);
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
            for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
                callArgs.push_back(desc.size(builder, loc, dim));
            }
            for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
                callArgs.push_back(desc.stride(builder, loc, dim));
            }
        };

        // Add the arguments and their shadow on data in
        for (auto [arg, llvmMemrefArg] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
            auto it = std::find(diffArgIndices.begin(), diffArgIndices.end(), index);
            if (it == diffArgIndices.end()) {
                callArgs.push_back(enzymeConst);
                if (isa<MemRefType>(arg.getType())) {
                    unpackMemRef(arg, nullptr, callArgs, rewriter, loc);
                }
                else {
                    Value casted = rewriter
                                       .create<UnrealizedConversionCastOp>(
                                           loc, llvmTypeConverter.convertType(arg.getType()), arg)
                                       .getResult(0);
                    callArgs.push_back(casted);
                }
            }
            else {
                auto position = std::distance(diffArgIndices.begin(), it);
                unpackMemRef(arg, dataIn[position], callArgs, rewriter, loc, /*zeroOut=*/true);
            }
            index++;
        }

        for (Value qJacobian : op.getQuantumJacobian()) {
            // Enzyme requires buffers for the primal outputs, but we don't need them.
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

            unpackMemRef(result, qJacobian, callArgs, rewriter, loc, /*zeroOut=*/false,
                         /*dupNoNeed=*/true);
        }

        // The results of backprop are in data in
        rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, callArgs);
        rewriter.eraseOp(op);
        return success();
    }

  private:
    static FlatSymbolRefAttr getOrInsertEnzymeGlobal(PatternRewriter &rewriter, Operation *op,
                                                     const char *globalName)
    {
        // Copyright (C) 2023 - Jacob Mai Peng
        // https://github.com/pengmai/lagrad/blob/main/lib/LAGrad/LowerToLLVM.cpp
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        auto *context = moduleOp.getContext();
        if (moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName)) {
            return SymbolRefAttr::get(context, globalName);
        }

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto shortTy = IntegerType::get(context, 8);
        rewriter.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                                        /*isConstant=*/true, LLVM::Linkage::Linkonce, globalName,
                                        IntegerAttr::get(shortTy, 0));
        return SymbolRefAttr::get(context, globalName);
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

    static Value computeMemRefSizeInBytes(MemRefType type, MemRefDescriptor descriptor,
                                          OpBuilder &builder, Location loc)
    {
        // element_size * (offset + sizes[0] * strides[0])
        Value bufferSize;
        if (type.getRank() == 0) {
            bufferSize = builder.create<LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(1));
        }
        else {
            bufferSize = builder.create<LLVM::MulOp>(loc, descriptor.size(builder, loc, 0),
                                                     descriptor.stride(builder, loc, 0));
            bufferSize =
                builder.create<LLVM::AddOp>(loc, descriptor.offset(builder, loc), bufferSize);
        }
        Value elementByteSize = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                                 type.getElementTypeBitWidth() / 8);
        Value bufferSizeBytes = builder.create<LLVM::MulOp>(loc, elementByteSize, bufferSize);
        return bufferSizeBytes;
    }

    static LLVM::GlobalOp insertEnzymeAllocationLike(OpBuilder &builder, ModuleOp moduleOp,
                                                     Location loc, StringRef allocFuncName,
                                                     StringRef freeFuncName)
    {
        MLIRContext *context = moduleOp.getContext();
        OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());
        const char *key = "__enzyme_allocation_like";

        auto allocationLike = moduleOp.lookupSymbol<LLVM::GlobalOp>(key);
        if (allocationLike) {
            return allocationLike;
        }

        auto ptrType = LLVM::LLVMPointerType::get(context);
        auto resultType = LLVM::LLVMArrayType::get(ptrType, 4);

        // TODO(jacob): document the weirdness of __enzyme_allocation_like
        // TODO(jacob): the dealloc_indices and size index should probably be configurable
        builder.create<LLVM::GlobalOp>(loc, LLVM::LLVMArrayType::get(builder.getI8Type(), 3), true,
                                       LLVM::Linkage::Linkonce, "dealloc_indices",
                                       builder.getStringAttr(StringRef("-1", 3)));
        allocationLike = builder.create<LLVM::GlobalOp>(
            loc, resultType,
            /*isConstant=*/false, LLVM::Linkage::External, key, /*address space=*/nullptr);
        builder.createBlock(&allocationLike.getInitializerRegion());
        auto allocFn = builder.create<LLVM::AddressOfOp>(loc, ptrType, allocFuncName);
        auto sizeArgIndex =
            builder.create<LLVM::ConstantOp>(loc, builder.getIntegerAttr(builder.getI64Type(), 0));
        auto sizeArgIndexPtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, sizeArgIndex);
        auto deallocIndicesPtr = builder.create<LLVM::AddressOfOp>(loc, ptrType, "dealloc_indices");
        auto freeFn = builder.create<LLVM::AddressOfOp>(loc, ptrType, freeFuncName);

        Value result = builder.create<LLVM::UndefOp>(loc, resultType);
        result = builder.create<LLVM::InsertValueOp>(loc, result, allocFn, 0);
        result = builder.create<LLVM::InsertValueOp>(loc, result, sizeArgIndexPtr, 1);
        result = builder.create<LLVM::InsertValueOp>(loc, result, deallocIndicesPtr, 2);
        result = builder.create<LLVM::InsertValueOp>(loc, result, freeFn, 3);
        builder.create<LLVM::ReturnOp>(loc, result);

        return allocationLike;
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populateConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<AdjointOpPattern>(typeConverter, patterns.getContext());
    patterns.add<BackpropOpPattern>(typeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
