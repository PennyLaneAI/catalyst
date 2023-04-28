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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

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


/// Generate an mlir function to wrap an existing function into a return-by-pointer style function.
///
/// .
///
func::FuncOp genEnzymeWrapperFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                      func::FuncOp argMapFn)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Define the properties of the enzyme wrapper function.
    std::string fnName = gradOp.getCallee().str() + ".enzyme_wrapper";
    SmallVector<Type> argTypes(argMapFn.getArgumentTypes().begin(),
                               argMapFn.getArgumentTypes().end());
    argTypes.insert(argTypes.end(), argMapFn.getResultTypes().begin(),
                    argMapFn.getResultTypes().end());

    SmallVector<Type> originalArgTypes, bufferizedArgTypes;
    for (auto argTypeIt = argTypes.begin(); argTypeIt < argTypes.end() - argMapFn.getNumResults();
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
    SmallVector<Type> bufferizedResultTypes, llvmResultTypes;
    for (auto resTypeIt = argTypes.begin() + argMapFn.getNumArguments(); resTypeIt < argTypes.end();
         resTypeIt++) {
        Type buffResType = buffTypeConverter.convertType(*resTypeIt);
        bufferizedResultTypes.push_back(buffResType);
        Type llvmResType = llvmTypeConverter.convertType(buffResType);
        if (!llvmResType)
            emitError(loc, "Could not convert argmap result to LLVM type: ") << buffResType;
        llvmResultTypes.push_back(llvmResType);
        *resTypeIt = LLVM::LLVMPointerType::get(llvmResType);
    }

    FunctionType fnType = rewriter.getFunctionType(argTypes, {});
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp enzymeFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, rewriter.getStringAttr(fnName));
    if (!enzymeFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(argMapFn);

        enzymeFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = enzymeFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        SmallVector<Value> callArgs(enzymeFn.getArguments().begin(),
                                    enzymeFn.getArguments().end() - argMapFn.getNumResults());
        for (auto [arg, buffType] : llvm::zip(callArgs, bufferizedArgTypes)) {
            if (arg.getType().isa<LLVM::LLVMPointerType>()) {
                Value memrefStruct = rewriter.create<LLVM::LoadOp>(loc, arg);
                Value memref =
                    rewriter.create<UnrealizedConversionCastOp>(loc, buffType, memrefStruct)
                        .getResult(0);
                arg = rewriter.create<bufferization::ToTensorOp>(loc, memref);
            }
        }
        ValueRange results = rewriter.create<func::CallOp>(loc, argMapFn, callArgs).getResults();

        ValueRange resArgs = enzymeFn.getArguments().drop_front(argMapFn.getNumArguments());

        SmallVector<Value> tensorFreeResults;
        for (auto [result, memrefType] : llvm::zip(results, bufferizedResultTypes)) {
            if (result.getType().isa<TensorType>())
                result = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, result);
            tensorFreeResults.push_back(result);
        }

        ValueRange llvmResults =
            rewriter.create<UnrealizedConversionCastOp>(loc, llvmResultTypes, tensorFreeResults)
                .getResults();
        for (auto [result, resArg] : llvm::zip(llvmResults, resArgs)) {
            rewriter.create<LLVM::StoreOp>(loc, result, resArg);
        }

        rewriter.create<func::ReturnOp>(loc);
    }

    return enzymeFn;
}

/// Generate an mlir function to compute the classical Jacobian via Enzyme.
///
/// .
///
func::FuncOp genBackpropFunction(PatternRewriter &rewriter, Location loc, gradient::GradOp gradOp,
                                 func::FuncOp callee, func::FuncOp wrapper)
{
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    bufferization::BufferizeTypeConverter buffTypeConverter;

    // Declare the special Enzyme autodiff function.
    std::string autodiffFnName = "__enzymne_autodiff";
    LLVM::LLVMFuncOp autodiffFn = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        wrapper, rewriter.getStringAttr(autodiffFnName));
    if (!autodiffFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(wrapper->getParentOfType<mlir::ModuleOp>().getBody());

        LLVM::LLVMFunctionType fnType =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {}, /*isVarArg=*/true);

        autodiffFn = rewriter.create<LLVM::LLVMFuncOp>(loc, autodiffFnName, fnType);
    }

    // Define the properties of the classical Jacobian function.
    std::string fnName = wrapper.getName().str() + ".backprop";
    TypeRange argTypes = callee.getArgumentTypes();
    std::vector<Type> resTypes = computeResultTypes(callee, gradOp.compDiffArgIndices());
    // Drop the last dimension as we only compute the gradient (not Jacobian) for now.
    for (Type &type : resTypes) {
        if (auto tensorType = type.dyn_cast<TensorType>()) {
            type = RankedTensorType::get(tensorType.getShape().drop_back(),
                                         tensorType.getElementType());
        }
    }
    FunctionType fnType = rewriter.getFunctionType(argTypes, resTypes);
    StringAttr visibility = rewriter.getStringAttr("private");

    size_t numResultArgs = wrapper.getNumArguments() - callee.getNumArguments();

    func::FuncOp gradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(wrapper, rewriter.getStringAttr(fnName));
    if (!gradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(wrapper);

        gradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = gradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        FunctionType wrapperType = wrapper.getFunctionType();
        Value wrapperValue =
            rewriter.create<func::ConstantOp>(loc, wrapperType, wrapper.getName()).getResult();
        Type wrapperPtrType = llvmTypeConverter.convertType(wrapperType);
        Value wrapperPtr =
            rewriter.create<UnrealizedConversionCastOp>(loc, wrapperPtrType, wrapperValue)
                .getResult(0);

        SmallVector<Value> callArgs = {wrapperPtr};
        SmallVector<Value> gradients;
        // SmallVector<Type> gradientTypes;

        // Handle callee arguments and their shadows.
        ValueRange gradFnArgs = gradFn.getArguments();
        for (auto [arg, targetType] :
             llvm::zip(gradFnArgs, wrapper.getArgumentTypes().drop_back(numResultArgs))) {

            Value shadow;
            if (auto tensorType = arg.getType().dyn_cast<TensorType>()) {
                // Assume we're dealing with our own converted pointer types for now.
                assert(targetType.isa<LLVM::LLVMPointerType>());
                Type structType = targetType.cast<LLVM::LLVMPointerType>().getElementType();

                Value memref = rewriter.create<bufferization::ToMemrefOp>(
                    loc, buffTypeConverter.convertType(tensorType), arg);
                Value memrefStruct =
                    rewriter.create<UnrealizedConversionCastOp>(loc, structType, memref)
                        .getResult(0);
                Value c1 = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
                Value structPtr = rewriter.create<LLVM::AllocaOp>(loc, targetType, c1);
                rewriter.create<LLVM::StoreOp>(loc, memrefStruct, structPtr);
                arg = structPtr;

                // Also generate it's shadow.
                Value shadowMemref =
                    rewriter.create<memref::AllocOp>(loc, memref.getType().cast<MemRefType>());
                Value shadowStruct =
                    rewriter.create<UnrealizedConversionCastOp>(loc, structType, shadowMemref)
                        .getResult(0);
                Value shadowPtr = rewriter.create<LLVM::AllocaOp>(loc, targetType, c1);
                rewriter.create<LLVM::StoreOp>(loc, shadowStruct, shadowPtr);
                shadow = shadowPtr;
                gradients.push_back(shadow);
                // gradientTypes.push_back(shadowMemref.getType());
            }
            else {
                Type llvmArgType = llvmTypeConverter.convertType(arg.getType());
                if (!llvmArgType)
                    emitError(loc, "Could not convert argmap argument to LLVM type: ")
                        << arg.getType();
                if (llvmArgType != arg.getType()) {
                    arg = rewriter.create<UnrealizedConversionCastOp>(loc, llvmArgType, arg)
                              .getResult(0);
                }
            }

            callArgs.push_back(arg);
            if (shadow) {
                callArgs.push_back(shadow);
            }
        }

        // Handle callee results and their shadows.
        Value memrefSize = gradFn.getArguments().back();
        TypeRange calleeResTypes = callee.getResultTypes();
        for (auto [resType, targetType] :
             llvm::zip(calleeResTypes, wrapper.getArgumentTypes().take_back(numResultArgs))) {

            Value c1 = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
            Value resArg = rewriter.create<LLVM::AllocaOp>(loc, targetType, c1);
            Value shadow;
            if (auto tensorType = resType.dyn_cast<TensorType>()) {
                // Result tensors are always converted into pointer struct types.
                assert(targetType.isa<LLVM::LLVMPointerType>());
                Type structType = targetType.cast<LLVM::LLVMPointerType>().getElementType();

                // Also generate it's shadow.
                MemRefType memrefType =
                    buffTypeConverter.convertType(tensorType).cast<MemRefType>();
                Value shadowMemref = rewriter.create<memref::AllocOp>(loc, memrefType, memrefSize);

                // One-hot initialize the tangent vector.
                Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                SmallVector<Value> indices(memrefType.getRank(), c0);
                Value c1_f = rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getFloatAttr(memrefType.getElementType(), 1.0));
                rewriter.create<memref::StoreOp>(loc, c1_f, shadowMemref, indices);

                Value shadowStruct =
                    rewriter.create<UnrealizedConversionCastOp>(loc, structType, shadowMemref)
                        .getResult(0);
                Value shadowPtr = rewriter.create<LLVM::AllocaOp>(loc, targetType, c1);
                rewriter.create<LLVM::StoreOp>(loc, shadowStruct, shadowPtr);
                shadow = shadowPtr;
            }

            callArgs.push_back(resArg);
            if (shadow) {
                callArgs.push_back(shadow);
            }
        }

        rewriter.create<LLVM::CallOp>(loc, autodiffFn, callArgs);

        SmallVector<Value> returnValues;
        for (auto [structPtr, targetType] : llvm::zip(gradients, gradFn.getResultTypes())) {
            // Assume result gradients are always tensors for now.
            assert(targetType.isa<TensorType>());
            Type targetMemrefType = buffTypeConverter.convertType(targetType);

            Value memrefStruct = rewriter.create<LLVM::LoadOp>(loc, structPtr);
            Value memref =
                rewriter.create<UnrealizedConversionCastOp>(loc, targetMemrefType, memrefStruct)
                    .getResult(0);
            // Value castedMemref = rewriter.create<memref::CastOp>(loc, targetMemrefType, memref);
            Value tensor = rewriter.create<bufferization::ToTensorOp>(loc, memref);
            returnValues.push_back(tensor);
        }
        rewriter.create<func::ReturnOp>(loc, returnValues);
    }

    return gradFn;
}

// struct BackpropOpPattern : public OpConversionPattern<BackpropOp> {
//     using OpConversionPattern::OpConversionPattern;

//     LogicalResult matchAndRewrite(BackpropOp op, BackpropOpAdaptor adaptor,
//                                   ConversionPatternRewriter &rewriter) const override
//     {
//         Location loc = op.getLoc();
//         MLIRContext *ctx = getContext();
//         TypeConverter *conv = getTypeConverter();

//         Type vectorType = conv->convertType(MemRefType::get({UNKNOWN}, Float64Type::get(ctx)));

//         for (Type type : op.getResultTypes()) {
//             if (!type.isa<MemRefType>())
//                 return op.emitOpError("must be bufferized before lowering");

//             // Currently only expval gradients are supported by the runtime,
//             // leading to tensor<?xf64> return values.
//             if (type.dyn_cast<MemRefType>() != MemRefType::get({UNKNOWN}, Float64Type::get(ctx)))
//                 return op.emitOpError("adjoint can only return MemRef<?xf64> or tuple thereof");
//         }

//         // The callee of the adjoint op must return as a single result the quantum register.
//         func::FuncOp callee =
//             SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
//         assert(callee && callee.getNumResults() == 1 && "invalid qfunc symbol in adjoint op");

//         StringRef cacheFnName = "__quantum__rt__toggle_recorder";
//         StringRef gradFnName = "__quantum__qis__Gradient";
//         Type cacheFnSignature =
//             LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), IntegerType::get(ctx, 1));
//         Type gradFnSignature = LLVM::LLVMFunctionType::get(
//             LLVM::LLVMVoidType::get(ctx), IntegerType::get(ctx, 64), /*isVarArg=*/true);

//         LLVM::LLVMFuncOp cacheFnDecl =
//             ensureFunctionDeclaration(rewriter, op, cacheFnName, cacheFnSignature);
//         LLVM::LLVMFuncOp gradFnDecl =
//             ensureFunctionDeclaration(rewriter, op, gradFnName, gradFnSignature);

//         // Run the forward pass and cache the circuit.
//         Value c_true = rewriter.create<LLVM::ConstantOp>(
//             loc, rewriter.getIntegerAttr(IntegerType::get(ctx, 1), 1));
//         Value c_false = rewriter.create<LLVM::ConstantOp>(
//             loc, rewriter.getIntegerAttr(IntegerType::get(ctx, 1), 0));
//         rewriter.create<LLVM::CallOp>(loc, cacheFnDecl, c_true);
//         Value qreg = rewriter.create<func::CallOp>(loc, callee, op.getArgs()).getResult(0);
//         if (!qreg.getType().isa<catalyst::quantum::QuregType>())
//             return callee.emitOpError("qfunc must return quantum register");
//         rewriter.create<LLVM::CallOp>(loc, cacheFnDecl, c_false);

//         // We follow the C ABI convention of passing result memrefs as struct pointers in the
//         // arguments to the C function, although in this case as a variadic argument list to allow
//         // for a varying number of results in a single signature.
//         Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
//         Value numResults = rewriter.create<LLVM::ConstantOp>(
//             loc, rewriter.getI64IntegerAttr(op.getDataIn().size()));
//         SmallVector<Value> args = {numResults};
//         for (Value memref : adaptor.getDataIn()) {
//             auto newArg =
//                 rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(vectorType), c1);
//             rewriter.create<LLVM::StoreOp>(loc, memref, newArg);
//             args.push_back(newArg);
//         }

//         rewriter.create<LLVM::CallOp>(loc, gradFnDecl, args);
//         rewriter.create<catalyst::quantum::DeallocOp>(loc, qreg);
//         rewriter.eraseOp(op);

//         return success();
//     }
// };

} // namespace

namespace catalyst {
namespace gradient {

void populateConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    // patterns.add<BackpropOpPattern>(typeConverter, patterns.getContext());
    patterns.add<AdjointOpPattern>(typeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
