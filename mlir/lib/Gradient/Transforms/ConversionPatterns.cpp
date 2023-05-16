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
#include <deque>
#include <string>
#include <vector>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

func::FuncOp genEnzymeWrapperFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Define the properties of the enzyme wrapper function.
    std::string fnName = callee.getName().str() + ".enzyme_wrapper";
    SmallVector<Type> argResTypes(callee.getArgumentTypes().begin(),
                               callee.getArgumentTypes().end());
    argResTypes.insert(argResTypes.end(), callee.getResultTypes().begin(),
                    callee.getResultTypes().end());
    

    // Bufferize and lower the arguments
    SmallVector<Type> originalArgTypes, bufferizedArgTypes;
    for (auto argTypeIt = argResTypes.begin(); argTypeIt < argResTypes.end() - callee.getNumResults();
         argTypeIt++) {
        originalArgTypes.push_back(*argTypeIt);
        if (argTypeIt->isa<TensorType>()) {
            Type buffArgType = buffTypeConverter.convertType(*argTypeIt);
            bufferizedArgTypes.push_back(buffArgType);
            Type llvmArgType = llvmTypeConverter.convertType(buffArgType);
            if (!llvmArgType)
                emitError(loc, "Could not convert argmap argument to LLVM type: ") << buffArgType;
            *argTypeIt = LLVM::LLVMPointerType::get(llvmArgType);
        }
        else {
            bufferizedArgTypes.push_back(*argTypeIt);
        }
    }

    // Bufferize and lower the results
    SmallVector<Type> bufferizedResultTypes, llvmResultTypes;
    for (auto resTypeIt = argResTypes.begin() + callee.getNumArguments(); resTypeIt < argResTypes.end();
         resTypeIt++) {
        Type buffResType = buffTypeConverter.convertType(*resTypeIt);
        bufferizedResultTypes.push_back(buffResType);
        Type llvmResType = llvmTypeConverter.convertType(buffResType);
        if (!llvmResType)
            emitError(loc, "Could not convert argmap result to LLVM type: ") << buffResType;
        llvmResultTypes.push_back(llvmResType);
        *resTypeIt = LLVM::LLVMPointerType::get(llvmResType);
    }


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

        SmallVector<Value> callArgs(wrappedCallee.getArguments().begin(),
                                    wrappedCallee.getArguments().end() - callee.getNumResults());
        for (auto [arg, buffType] : llvm::zip(callArgs, bufferizedArgTypes)) {
            if (arg.getType().isa<LLVM::LLVMPointerType>()) {
                Value memrefStruct = rewriter.create<LLVM::LoadOp>(loc, arg);
                Value memref =
                    rewriter.create<UnrealizedConversionCastOp>(loc, buffType, memrefStruct)
                        .getResult(0);
                arg = rewriter.create<bufferization::ToTensorOp>(loc, memref);
            }
        }
        // Call the callee
        ValueRange results = rewriter.create<func::CallOp>(loc, callee, callArgs).getResults();

        ValueRange resArgs = wrappedCallee.getArguments().drop_front(callee.getNumArguments());

        //Bufferize the results
        SmallVector<Value> tensorFreeResults;
        for (auto [result, memrefType] : llvm::zip(results, bufferizedResultTypes)) {
            if (result.getType().isa<TensorType>())
                result = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, result);
            tensorFreeResults.push_back(result);
        }

        ValueRange llvmResults =
            rewriter.create<UnrealizedConversionCastOp>(loc, llvmResultTypes, tensorFreeResults)
                .getResults();

        // Store the results
        for (auto [result, resArg] : llvm::zip(llvmResults, resArgs)) {
            rewriter.create<LLVM::StoreOp>(loc, result, resArg);
        }

        // Add return op
        rewriter.create<func::ReturnOp>(loc);
    }

    return wrappedCallee;
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

        StringRef backpropFnName = "__enzyme_autodiff";
        Type backpropFnSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx), {}, /*isVarArg=*/true);

        LLVM::LLVMFuncOp backpropFnDecl = ensureFunctionDeclaration(rewriter, op, backpropFnName, backpropFnSignature);

        SmallVector<Type> argTypes(op.getArgs().getTypes().begin(), op.getArgs().getTypes().end());

        // We follow the C ABI convention of passing result memrefs as struct pointers in the
        // arguments to the C function, although in this case as a variadic argument list to allow
        // for a varying number of results in a single signature.
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        Value numResults = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(op.getDataIn().size()));
        SmallVector<Value> argsRes = {numResults};
        for (Value memref : adaptor.getDataIn()) {
            auto newArg =
                rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
            rewriter.create<LLVM::StoreOp>(loc, memref, newArg);
            argsRes.push_back(newArg);
        }

        genEnzymeWrapperFunction(rewriter, loc, callee);

        rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, argsRes);
        //rewriter.create<func::ReturnOp>(loc, returnValues);

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
