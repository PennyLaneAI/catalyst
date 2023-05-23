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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
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
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Define the properties of the enzyme wrapper function.
    std::string fnName = callee.getName().str() + ".enzyme_wrapper";
    SmallVector<Type> argResTypes(callee.getArgumentTypes().begin(),
                                  callee.getArgumentTypes().end());
    SmallVector<Type> originalArgTypes(callee.getArgumentTypes().begin(),
                                       callee.getArgumentTypes().end());

    argResTypes.insert(argResTypes.end(), callee.getResultTypes().begin(),
                       callee.getResultTypes().end());

    // Lower the results
    for (auto resTypeIt = argResTypes.begin() + callee.getNumArguments();
         resTypeIt < argResTypes.end(); resTypeIt++) {
        Type buffResType = buffTypeConverter.convertType(*resTypeIt);
        Type llvmResType = llvmTypeConverter.convertType(buffResType);
        if (!llvmResType)
            emitError(loc, "Could not convert argmap result to LLVM type: ") << buffResType;
        *resTypeIt = LLVM::LLVMPointerType::get(llvmResType);
    }

    ArrayRef convertedResTypes(argResTypes.begin() + callee.getNumArguments(), argResTypes.end());

    // Create the wrapped operation
    FunctionType fnType = rewriter.getFunctionType(argResTypes, {});
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp wrappedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!wrappedCallee) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        wrappedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = wrappedCallee.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Get the arguments
        SmallVector<Value> callArgs(wrappedCallee.getArguments().begin(),
                                    wrappedCallee.getArguments().end() - callee.getNumResults());

        // Call the callee
        ValueRange results = rewriter.create<func::CallOp>(loc, callee, callArgs).getResults();
        results = rewriter.create<UnrealizedConversionCastOp>(loc, convertedResTypes, results)
                      .getResults();
        ValueRange resArgs = wrappedCallee.getArguments().drop_front(callee.getNumArguments());

        // Store the results
        for (auto [result, resArg] : llvm::zip(results, resArgs)) {
            rewriter.create<LLVM::StoreOp>(loc, result, resArg);
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
        TypeConverter *conv = getTypeConverter();
        LLVMTypeConverter llvmTypeConverter(ctx);

        Type vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));

        for (Type type : op.getResultTypes()) {
            if (!type.isa<MemRefType>())
                return op.emitOpError("must be bufferized before lowering");
        }

        // The callee of the backprop Op
        func::FuncOp callee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        assert(callee);

        // Creat constants
        Value c0 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));

        // Create mlir memref to llvm
        StringRef allocFnName = "_mlir_memref_to_llvm_alloc";
        Type allocFnSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx),
                                                            {IntegerType::get(ctx, 64)});

        LLVM::LLVMFuncOp allocFnDecl =
            ensureFunctionDeclaration(rewriter, op, allocFnName, allocFnSignature);
        // Create the Enzyme function
        StringRef backpropFnName = "__enzyme_autodiff";
        Type backpropFnSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {}, /*isVarArg=*/true);

        LLVM::LLVMFuncOp backpropFnDecl =
            ensureFunctionDeclaration(rewriter, op, backpropFnName, backpropFnSignature);

        // Create the memset function
        StringRef memsetFnName = "memset";
        Type memsetFnSignature =
            LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx),
                                        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 32),
                                         IntegerType::get(ctx, 64)});

        LLVM::LLVMFuncOp memsetFnDecl =
            ensureFunctionDeclaration(rewriter, op, memsetFnName, memsetFnSignature);

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

        // Add the arguments and their shadow
        for (auto [arg, llvmarg] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
            auto newArg =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, llvmarg, newArg);
            callArgs.push_back(newArg);

            if (!llvmarg.getType().isa<LLVM::LLVMStructType>()) {
                Value shadowPtr = rewriter.create<LLVM::AllocaOp>(
                    loc, LLVM::LLVMPointerType::get(llvmarg.getType()), c1);
                Value zeroArg = rewriter.create<LLVM::ConstantOp>(
                    loc, llvmarg.getType(), rewriter.getZeroAttr(llvmarg.getType()));
                rewriter.create<LLVM::StoreOp>(loc, zeroArg, shadowPtr);
                callArgs.push_back(shadowPtr);
            }
            else {
                // Add shadow for memref
                Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, llvmarg, 2);

                SmallVector<int64_t> vectorIndices{3, 0};

                Value sizeArray =
                    rewriter.create<LLVM::ExtractValueOp>(loc, llvmarg, vectorIndices);

                vectorIndices[0] = 4;
                Value strideArray =
                    rewriter.create<LLVM::ExtractValueOp>(loc, llvmarg, vectorIndices);

                Value bufferSize = rewriter.create<LLVM::MulOp>(loc, sizeArray, strideArray);
                bufferSize = rewriter.create<LLVM::AddOp>(loc, offset, bufferSize);

                MemRefType memrefType = arg.getType().cast<MemRefType>();
                size_t value = memrefType.getElementTypeBitWidth() / 8;

                auto memrefSize =
                    rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64), value);
                Value bufferMemSize = rewriter.create<LLVM::MulOp>(loc, bufferSize, memrefSize);

                Value buffer =
                    rewriter.create<LLVM::CallOp>(loc, allocFnDecl, bufferMemSize).getResult();

                // Set value to 0
                rewriter.create<LLVM::CallOp>(loc, memsetFnDecl,
                                              ArrayRef<Value>{buffer, c0, bufferMemSize});

                Type llvmBaseType = conv->convertType(memrefType.getElementType());
                Value bufferCast = rewriter.create<LLVM::BitcastOp>(
                    loc, LLVM::LLVMPointerType::get(llvmBaseType), buffer);

                llvmarg = rewriter.create<LLVM::InsertValueOp>(loc, llvmarg, bufferCast, 0);
                llvmarg = rewriter.create<LLVM::InsertValueOp>(loc, llvmarg, bufferCast, 1);

                Value shadowPtr = rewriter.create<LLVM::AllocaOp>(
                    loc, LLVM::LLVMPointerType::get(vectorType), c1);
                rewriter.create<LLVM::StoreOp>(loc, llvmarg, shadowPtr);

                callArgs.push_back(shadowPtr);
            }
        }

        // Add the results and their shadows;

        for (auto [memref, llvmmemref] : llvm::zip(op.getDataIn(), adaptor.getDataIn())) {
            // Add the results
            auto newArg =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, llvmmemref, newArg);
            callArgs.push_back(newArg);

            // Add shadow for memref

            Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, llvmmemref, 2);

            SmallVector<int64_t> vectorIndices{3, 0};

            Value sizeArray = rewriter.create<LLVM::ExtractValueOp>(loc, llvmmemref, vectorIndices);

            vectorIndices[0] = 4;
            Value strideArray =
                rewriter.create<LLVM::ExtractValueOp>(loc, llvmmemref, vectorIndices);

            Value bufferSize = rewriter.create<LLVM::MulOp>(loc, sizeArray, strideArray);
            bufferSize = rewriter.create<LLVM::AddOp>(loc, offset, bufferSize);

            MemRefType memrefType = memref.getType().cast<MemRefType>();
            size_t value = memrefType.getElementTypeBitWidth() / 8;

            auto memrefSize =
                rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64), value);
            Value bufferMemSize = rewriter.create<LLVM::MulOp>(loc, bufferSize, memrefSize);

            Value buffer =
                rewriter.create<LLVM::CallOp>(loc, allocFnDecl, bufferMemSize).getResult();

            // Set value to 0
            rewriter.create<LLVM::CallOp>(loc, memsetFnDecl,
                                          ArrayRef<Value>{buffer, c0, bufferMemSize});

            Type llvmBaseType = conv->convertType(memrefType.getElementType());
            Value bufferCast = rewriter.create<LLVM::BitcastOp>(
                loc, LLVM::LLVMPointerType::get(llvmBaseType), buffer);

            llvmmemref = rewriter.create<LLVM::InsertValueOp>(loc, llvmmemref, bufferCast, 0);
            llvmmemref = rewriter.create<LLVM::InsertValueOp>(loc, llvmmemref, bufferCast, 1);

            Value shadowPtr =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, llvmmemref, shadowPtr);

            callArgs.push_back(shadowPtr);
        }

        auto callop = rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, callArgs);
        
        auto results = callop.getResults();
        rewriter.eraseOp(op);
        return success();
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
