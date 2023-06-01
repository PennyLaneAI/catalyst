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
#include "mlir/Dialect/SCF/IR/SCF.h"

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

    // Define the properties of the enzyme wrapper function.
    std::string fnName = callee.getName().str() + ".enzyme_wrapper";
    SmallVector<Type> argResTypes(callee.getArgumentTypes().begin(),
                                  callee.getArgumentTypes().end());
    argResTypes.insert(argResTypes.end(), callee.getResultTypes().begin(),
                       callee.getResultTypes().end());
    SmallVector<Type> originalArgTypes(callee.getArgumentTypes().begin(),
                                       callee.getArgumentTypes().end());
    SmallVector<Type> convertedTypes;
    // Lower the args and results
    for (auto resTypeIt = argResTypes.begin(); resTypeIt < argResTypes.end(); resTypeIt++) {
        Type llvmResType = llvmTypeConverter.convertType(*resTypeIt);
        if (!llvmResType)
            emitError(loc, "Could not convert argmap result to LLVM type: ") << *resTypeIt;
        convertedTypes.push_back(llvmResType);
        *resTypeIt = LLVM::LLVMPointerType::get(llvmResType);
    }

    SmallVector<Type> convertedResTypes(convertedTypes.begin() + callee.getNumArguments(),
                                        convertedTypes.end());

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

        for (auto [arg, originalType] : llvm::zip(callArgs, originalArgTypes)) {
            if (arg.getType().dyn_cast<LLVM::LLVMPointerType>()) {
                Value memrefStruct = rewriter.create<LLVM::LoadOp>(loc, arg);
                arg = rewriter.create<UnrealizedConversionCastOp>(loc, originalType, memrefStruct)
                          .getResult(0);
            }
        }
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
        LLVMTypeConverter llvmTypeConverter(ctx);

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
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));

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

        // Add the arguments and their shadow on data in
        for (auto [memrefArg, llvmMemrefArg, memrefDataIn, llvmDataIn] :
             llvm::zip(op.getArgs(), adaptor.getArgs(), op.getDataIn(), adaptor.getDataIn())) {

            // Get information about the memref arg
            MemRefType memrefArgType = memrefArg.getType().cast<MemRefType>();
            size_t sizeShape = memrefArgType.getShape().size();
            size_t value = memrefArgType.getElementTypeBitWidth() / 8;
            auto memrefArgSize =
                rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64), value);
            int64_t argRank = memrefArgType.getRank();

            // Prepare the offset size and stride for taking the subview of the memref from data in
            // (gradients)
            Value llvmMemref = llvmMemrefArg;
            Value memrefData = memrefDataIn;
            MemRefType memrefDataType = memrefData.getType().cast<MemRefType>();
            int64_t rank = memrefDataType.getRank();
            std::vector<int64_t> sizes = memrefDataType.getShape();

            sizes[0] = 1;
            std::vector<Value> dynSizes;
            for (int64_t dim = 1; dim < rank; dim++) {
                if (sizes[dim] == ShapedType::kDynamic) {
                    Value idx = rewriter.create<index::ConstantOp>(loc, dim);
                    Value dimSize = rewriter.create<memref::DimOp>(loc, memrefData, idx);
                    dynSizes.push_back(dimSize);
                }
            }

            std::vector<int64_t> offsets(rank, 0);
            offsets[0] = ShapedType::kDynamic;

            std::vector<int64_t> strides(rank, 1);
            std::vector<Value> dynStrides = {};

            // Loop over the grad size and take subviews of the memref data in in order to match the
            // memref arg
            auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            rewriter.create<scf::ForOp>(
                loc, lowerBound, op.getGradSize(), step, std::nullopt,
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    // Add the pointer to the wrapped callee
                    SmallVector<Value> callArgs = {wrapperPtr};
                    // Add the argument to call arg as a pointer
                    auto newMemrefArg = rewriter.create<LLVM::AllocaOp>(
                        loc, LLVM::LLVMPointerType::get(llvmMemref.getType()), c1);
                    rewriter.create<LLVM::StoreOp>(loc, llvmMemref, newMemrefArg);
                    callArgs.push_back(newMemrefArg);
                    std::vector<Value> dynOffsets = {iv};

                    // Get the subview type (match the memref arg)
                    Type subViewType;
                    if (argRank == 0) {
                        subViewType = memref::SubViewOp::inferRankReducedResultType(
                            memrefDataType.getShape().drop_front(), memrefDataType, offsets, sizes,
                            strides);
                    }
                    else {
                        subViewType = memref::SubViewOp::inferRankReducedResultType(
                            memrefDataType.getShape(), memrefDataType, offsets, sizes, strides);
                    }
                    Value shadowMemref = builder.create<memref::SubViewOp>(
                        loc, subViewType, memrefData, dynOffsets, dynSizes, dynStrides, offsets,
                        sizes, strides);
                    Type llvmResType = llvmTypeConverter.convertType(shadowMemref.getType());
                    Value shadowLlvm =
                        builder.create<UnrealizedConversionCastOp>(loc, llvmResType, shadowMemref)
                            .getResult(0);

                    Value bufferSize;

                    if (sizeShape == 0) {
                        bufferSize =
                            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
                    }
                    else {
                        Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemref, 2);

                        SmallVector<int64_t> vectorIndices{3, 0};

                        Value sizeArray =
                            rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemref, vectorIndices);

                        vectorIndices[0] = 4;
                        Value strideArray =
                            rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemref, vectorIndices);

                        bufferSize = rewriter.create<LLVM::MulOp>(loc, sizeArray, strideArray);
                        bufferSize = rewriter.create<LLVM::AddOp>(loc, offset, bufferSize);
                    }

                    Value bufferMemSize =
                        rewriter.create<LLVM::MulOp>(loc, bufferSize, memrefArgSize);

                    Value buffer =
                        rewriter.create<LLVM::CallOp>(loc, allocFnDecl, bufferMemSize).getResult();
                    // Set value to 0 (gradients)
                    rewriter.create<LLVM::CallOp>(loc, memsetFnDecl,
                                                  ArrayRef<Value>{buffer, c0, bufferMemSize});

                    Type llvmBaseType =
                        llvmTypeConverter.convertType(memrefArgType.getElementType());
                    Value bufferCast = rewriter.create<LLVM::BitcastOp>(
                        loc, LLVM::LLVMPointerType::get(llvmBaseType), buffer);

                    auto shadowLlvmInsert0 =
                        rewriter.create<LLVM::InsertValueOp>(loc, shadowLlvm, bufferCast, 0);
                    auto shadowLlvmInsert1 =
                        rewriter.create<LLVM::InsertValueOp>(loc, shadowLlvmInsert0, bufferCast, 1);

                    Value shadowPtr = rewriter.create<LLVM::AllocaOp>(
                        loc, LLVM::LLVMPointerType::get(shadowLlvm.getType()), c1);
                    rewriter.create<LLVM::StoreOp>(loc, shadowLlvmInsert1, shadowPtr);
                    callArgs.push_back(shadowPtr);

                    // Results of callee and their shadow
                    SmallVector<Value> calleeArgs(op.getArgs().begin(), op.getArgs().end());
                    ValueRange results =
                        rewriter.create<func::CallOp>(loc, callee, calleeArgs).getResults();

                    // Lower callee results to to llvm
                    SmallVector<Type> resCalleeTypes(callee.getResultTypes().begin(),
                                                     callee.getResultTypes().end());
                    for (auto resTypeIt = resCalleeTypes.begin(); resTypeIt < resCalleeTypes.end();
                         resTypeIt++) {
                        Type llvmResType = llvmTypeConverter.convertType(*resTypeIt);
                        if (!llvmResType)
                            emitError(loc, "Could not convert argmap result to LLVM type: ")
                                << *resTypeIt;
                        *resTypeIt = llvmResType;
                    }

                    ValueRange llvmresults =
                        rewriter.create<UnrealizedConversionCastOp>(loc, resCalleeTypes, results)
                            .getResults();

                    // Store the results
                    for (auto [result, llvmresult, llvmshadowresult] :
                         llvm::zip(results, llvmresults, adaptor.getQuantumJacobians())) {
                        auto newArg = rewriter.create<LLVM::AllocaOp>(
                            loc, LLVM::LLVMPointerType::get(llvmresult.getType()), c1);
                        rewriter.create<LLVM::StoreOp>(loc, llvmresult, newArg);
                        callArgs.push_back(newArg);

                        // Add shadow for memref
                        Value shadowPtr = rewriter.create<LLVM::AllocaOp>(
                            loc, LLVM::LLVMPointerType::get(llvmshadowresult.getType()), c1);
                        rewriter.create<LLVM::StoreOp>(loc, llvmshadowresult, shadowPtr);
                        callArgs.push_back(shadowPtr);
                    }

                    // The results of backprop are in data in
                    rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, callArgs);

                    rewriter.create<scf::YieldOp>(loc);
                });
        }
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
