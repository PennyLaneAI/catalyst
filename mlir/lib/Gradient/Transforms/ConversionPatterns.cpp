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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

#include "Catalyst/Utils/CallGraph.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Gradient/Utils/DestinationPassingStyle.h"
#include "Gradient/Utils/EinsumLinalgGeneric.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;
using namespace catalyst::gradient;

using llvm::errs;

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
static constexpr const char *enzyme_custom_gradient_key = "__enzyme_register_gradient_";
static constexpr const char *enzyme_like_free_key = "__enzyme_function_like_free";
static constexpr const char *enzyme_const_key = "enzyme_const";
static constexpr const char *enzyme_dupnoneed_key = "enzyme_dupnoneed";

void wrapMemRefArgs(func::FuncOp func, LLVMTypeConverter &typeConverter, PatternRewriter &rewriter,
                    Location loc)
{
    if (llvm::none_of(func.getArgumentTypes(),
                      [](Type argType) { return isa<MemRefType>(argType); })) {
        // The memref arguments are already wrapped
        return;
    }

    ModuleOp moduleOp = func->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
    for (const auto [idx, argType] : llvm::enumerate(func.getArgumentTypes())) {
        if (auto memrefType = dyn_cast<MemRefType>(argType)) {
            BlockArgument memrefArg = func.getArgument(idx);
            func.insertArgument(idx, ptrType, DictionaryAttr::get(ctx), loc);
            Value wrappedMemref = func.getArgument(idx);

            Type convertedType = typeConverter.convertType(memrefType);
            Value replacedMemref = rewriter.create<LLVM::LoadOp>(loc, convertedType, wrappedMemref);
            replacedMemref =
                rewriter.create<UnrealizedConversionCastOp>(loc, argType, replacedMemref)
                    .getResult(0);
            memrefArg.replaceAllUsesWith(replacedMemref);
            func.eraseArgument(memrefArg.getArgNumber());
        }
    }

    std::optional<SymbolTable::UseRange> uses = func.getSymbolUses(moduleOp);
    if (uses.has_value()) {
        for (auto use : *uses) {
            if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(callOp);

                Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));

                SmallVector<Value> operands;
                SmallVector<Value> outputs;
                auto wrapMemref = [&](Value memref) {
                    Type convertedType = typeConverter.convertType(memref.getType());
                    Value space =
                        rewriter.create<LLVM::AllocaOp>(loc, /*resultType=*/ptrType,
                                                        /*elementType=*/convertedType, c1);
                    Value convertedValue =
                        rewriter.create<UnrealizedConversionCastOp>(loc, convertedType, memref)
                            .getResult(0);
                    rewriter.create<LLVM::StoreOp>(loc, convertedValue, space);
                    return space;
                };
                for (Value oldOperand : callOp.getOperands()) {
                    if (isa<MemRefType>(oldOperand.getType())) {
                        operands.push_back(wrapMemref(oldOperand));
                    }
                }
                for (Type resultType : callOp.getResultTypes()) {
                    if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
                        assert(memrefType.hasStaticShape());
                        Value memref = rewriter.create<memref::AllocOp>(loc, memrefType);
                        outputs.push_back(memref);

                        memref = wrapMemref(memref);
                        operands.push_back(memref);
                    }
                }

                rewriter.create<func::CallOp>(callOp.getLoc(), func, operands);
                rewriter.replaceOp(callOp, outputs);
            }
        }
    }
}

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

        catalyst::convertToDestinationPassingStyle(callee, rewriter);
        SymbolTableCollection symbolTable;
        catalyst::traverseCallGraph(callee, symbolTable, [&](func::FuncOp func) {
            // Register custom gradients of quantum functions
            if (func->hasAttrOfType<FlatSymbolRefAttr>("gradient.qgrad")) {
                auto qgradFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                    func, func->getAttrOfType<FlatSymbolRefAttr>("gradient.qgrad"));

                // When lowering multiple backprop ops, the callee type will be mutated by the
                // wrapped op. Save the original unwrapped function type so that later backprop
                // ops can read it.
                if (!func->hasAttr("unwrapped_type")) {
                    func->setAttr("unwrapped_type", TypeAttr::get(func.getFunctionType()));
                }
                catalyst::convertToDestinationPassingStyle(func, rewriter);

                wrapMemRefArgs(func, *getTypeConverter(), rewriter, loc);

                func::FuncOp augFwd = genAugmentedForward(func, rewriter);
                func::FuncOp customQGrad =
                    genCustomQGradient(func, func.getLoc(), qgradFn, rewriter);
                insertEnzymeCustomGradient(rewriter, func->getParentOfType<ModuleOp>(),
                                           func.getLoc(), func, augFwd, customQGrad);
            }
        });

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
            insertGlobalSymbol(rewriter, moduleOp, "freename", StringRef("free", 5));
            insertEnzymeFunctionLike(rewriter, moduleOp, enzyme_like_free_key, "freename",
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

        const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
        insertGlobalSymbol(rewriter, moduleOp, enzyme_const_key, std::nullopt);
        insertGlobalSymbol(rewriter, moduleOp, enzyme_dupnoneed_key, std::nullopt);

        ValueRange argShadows = adaptor.getDiffArgShadows();
        Value enzymeConst = rewriter.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                                               enzyme_const_key);
        ValueRange convArgs = adaptor.getArgs();
        // Add the arguments and the argument shadows of memrefs
        for (auto [index, arg] : llvm::enumerate(op.getArgs())) {
            auto it = std::find(diffArgIndices.begin(), diffArgIndices.end(), index);
            if (it == diffArgIndices.end()) {
                if (isa<MemRefType>(arg.getType())) {
                    // unpackMemRefAndAppend will handle the appropriate enzyme_const annotations
                    unpackMemRefAndAppend(arg, /*shadow=*/nullptr, callArgs, rewriter, loc);
                }
                else {
                    callArgs.push_back(enzymeConst);
                    callArgs.push_back(convArgs[index]);
                }
            }
            else {
                assert(isDifferentiable(arg.getType()));
                if (isa<MemRefType>(arg.getType())) {
                    size_t position = std::distance(diffArgIndices.begin(), it);
                    unpackMemRefAndAppend(arg, argShadows[position], callArgs, rewriter, loc,
                                          {.zeroOut = true});
                }
                else {
                    callArgs.push_back(arg);
                }
            }
        }

        for (auto [result, cotangent] :
             llvm::zip_equal(op.getCalleeResults(), op.getCotangents())) {
            unpackMemRefAndAppend(result, cotangent, callArgs, rewriter, loc, {.dupNoNeed = true});
        }

        // The results of backprop are in argShadows, except scalar derivatives which are in the
        // results of the enzyme call.
        auto enzymeCall = rewriter.create<LLVM::CallOp>(loc, backpropFnDecl, callArgs);
        SmallVector<Value> scalarResults;
        unpackScalarResults(enzymeCall, scalarResults, rewriter, loc);
        rewriter.replaceOp(op, scalarResults);
        return success();
    }

  private:
    Value castToConvertedType(Value value, OpBuilder &builder, Location loc) const
    {
        auto casted = builder.create<UnrealizedConversionCastOp>(
            loc, getTypeConverter()->convertType(value.getType()), value);
        return casted.getResult(0);
    }

    void unpackMemRefAndAppend(
        Value memRefArg, Value shadowMemRef, SmallVectorImpl<Value> &callArgs, OpBuilder &builder,
        Location loc, EnzymeMemRefInterfaceOptions options = EnzymeMemRefInterfaceOptions()) const

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

        // LLVM Functions can only return up to one result. If one scalar is being differentiated,
        // it will be the sole result. If there are multiple scalars being differentiated, Enzyme
        // will return a struct of all the derivatives with respect to those scalars.
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

    void convertCustomGradArgumentTypes(TypeRange argTypes,
                                        SmallVectorImpl<Type> &llvmArgTypes) const
    {
        for (Type argType : argTypes) {
            llvmArgTypes.append(2, argType);
        }
    }

    func::FuncOp genAugmentedForward(func::FuncOp qnode, OpBuilder &builder) const
    {
        std::string augmentedName = (qnode.getName() + ".augfwd").str();
        auto augmentedForward = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            qnode, builder.getStringAttr(augmentedName));
        if (augmentedForward) {
            return augmentedForward;
        }
        assert(qnode.getNumResults() == 0 && "Expected QNode to be in destination-passing style");
        MLIRContext *ctx = builder.getContext();
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointAfter(qnode);
        // The tape type is a null pointer because we don't need to pass any data from the forward
        // pass to the reverse pass.
        auto tapeType = LLVM::LLVMPointerType::get(ctx);
        SmallVector<Type> argTypes;
        convertCustomGradArgumentTypes(qnode.getArgumentTypes(), argTypes);
        augmentedForward = builder.create<func::FuncOp>(
            qnode.getLoc(), augmentedName, FunctionType::get(ctx, argTypes, {tapeType}));
        augmentedForward.setPrivate();
        Location loc = qnode.getLoc();

        Block *entry = augmentedForward.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        // Every other argument is a shadow
        SmallVector<Value> arguments;
        for (unsigned i = 0; i < qnode.getNumArguments(); i++) {
            arguments.push_back(augmentedForward.getArgument(i * 2));
        }

        builder.create<func::CallOp>(loc, qnode, arguments);
        Value tape = builder.create<LLVM::NullOp>(loc, tapeType);
        builder.create<func::ReturnOp>(loc, tape);
        return augmentedForward;
    }

    // `qnodeType` is the original type of the function prior to conversion to destination-passing
    // style and wrapping memrefs into pointers.
    func::FuncOp genCustomQGradient(func::FuncOp qnode, Location loc, func::FuncOp qgradFn,
                                    OpBuilder &builder) const
    {
        std::string customQGradName = (qnode.getName() + ".customqgrad").str();
        auto customQGrad = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            qnode, builder.getStringAttr(customQGradName));
        if (customQGrad) {
            return customQGrad;
        }

        SmallVector<Type> customArgTypes;
        MLIRContext *ctx = builder.getContext();
        auto tapeType = LLVM::LLVMPointerType::get(ctx);
        SmallVector<Type> argTypes;
        convertCustomGradArgumentTypes(qnode.getArgumentTypes(), argTypes);
        argTypes.push_back(tapeType);
        auto qnodeType =
            cast<FunctionType>(qnode->getAttrOfType<TypeAttr>("unwrapped_type").getValue());

        OpBuilder::InsertionGuard insertionGuard(builder);
        // make volatile load to argument
        {
            for (unsigned i = 0; i < qnodeType.getNumInputs(); i++) {
                auto structType = getTypeConverter()->convertType(qnodeType.getInput(i));
                builder.setInsertionPointToStart(&qnode.getFunctionBody().front());
                Value loaded = builder.create<LLVM::LoadOp>(loc, structType, qnode.getArgument(i),
                                                            /*alignment=*/0,
                                                            /*isVolatile=*/true);
                builder.create<LLVM::StoreOp>(loc, loaded, qnode.getArgument(i));
            }
        }
        //
        builder.setInsertionPoint(qnode);
        auto funcType = FunctionType::get(ctx, argTypes, {});
        customQGrad = builder.create<func::FuncOp>(qnode.getLoc(), customQGradName, funcType);
        customQGrad.setPrivate();
        Block *block = customQGrad.addEntryBlock();
        builder.setInsertionPointToStart(block);

        SmallVector<Value> primalArgs;
        SmallVector<Value> shadowArgs;
        for (unsigned i = 0; i < qnode.getNumArguments(); i++) {
            primalArgs.push_back(block->getArgument(i * 2));
            shadowArgs.push_back(block->getArgument(i * 2 + 1));
        }

        SmallVector<Value> unwrappedInputs;
        SmallVector<Value> unwrappedShadows;
        unsigned idx = 0;

        auto unwrapMemRef = [&](Value wrapped, Type unwrappedType) {
            auto structType = getTypeConverter()->convertType(unwrappedType);
            Value unwrapped = builder.create<LLVM::LoadOp>(loc, structType, wrapped);
            unwrapped = builder.create<UnrealizedConversionCastOp>(loc, unwrappedType, unwrapped)
                            .getResult(0);
            return unwrapped;
        };

        for (const auto &[unwrappedType, arg, shadow] :
             llvm::zip(qnodeType.getInputs(), primalArgs, shadowArgs)) {
            if (isa<MemRefType>(unwrappedType)) {
                unwrappedInputs.push_back(unwrapMemRef(arg, unwrappedType));
                unwrappedShadows.push_back(unwrapMemRef(shadow, unwrappedType));
            }
            else {
                assert(false && "non memref inputs not yet supported");
                unwrappedInputs.push_back(arg);
            }
            idx++;
        }

        SmallVector<Value> primalInputs{
            ValueRange{unwrappedInputs}.take_front(qgradFn.getNumArguments() - 1)};
        Value gateParamShadow = unwrappedShadows.back();
        // The gate param list is always 1-d
        Value pcount = builder.create<memref::DimOp>(loc, gateParamShadow, 0);
        primalInputs.push_back(pcount);

        auto qgrad = builder.create<func::CallOp>(loc, qgradFn, primalInputs);

        for (unsigned i = 0; i < qnodeType.getNumResults(); i++) {
            // The QNode has n inputs and m outputs (in destination-passing style).
            // The customQGrad arguments are: [
            //   inprimal_0, inshadow_0,
            //   ...,
            //   inprimal_n, inshadow_n,
            //   outprimal_0, outshadow_0,
            //   ...,
            //   outprimal_m, outshadow_m
            // ]
            // This indexing extracts [outshadow_0, ..., outshadow_m]
            Value resultShadow = unwrapMemRef(
                block->getArgument((i + qnodeType.getNumInputs()) * 2 + 1), qnodeType.getResult(i));

            // If G is the number of gate params and [...result] is the shape of the result with
            // rank R:
            //   - The result shadow always has shape [...result]
            //   - The quantum gradient always has shape [G, ...result]
            //   - The gate param shadow always has shape [G]
            // Since einsumLinalgGeneric uses integers to represent dimensions, the resulting
            // einsum to propagate the chain rule should thus be:
            //   [1,...,R], [0,1,...,R] -> [0]
            SmallVector<int64_t> qgradDims;
            int64_t qgradRank = cast<ShapedType>(qgrad.getType(i)).getRank();
            for (int64_t i = 0; i < qgradRank; i++) {
                qgradDims.push_back(i);
            }
            ArrayRef<int64_t> resultDims(qgradDims.begin() + 1, qgradDims.end());

            // The gate param shadow is shared, meaning it accumulates additions from einsums from
            // all results (this is due to the multivariate chain rule saying that derivatives
            // combine additively).
            catalyst::einsumLinalgGeneric(builder, loc, resultDims, qgradDims, {0}, resultShadow,
                                          qgrad.getResult(i), gateParamShadow);
        }
        builder.create<func::ReturnOp>(loc);

        return customQGrad;
    }

    /// This registers custom allocation and deallocation functions with Enzyme. It creates a
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

    /// This function inserts a llvm global (symbol) and associates it to a value if provided
    /// (optional).
    ///
    /// It can be used to add Enzyme globals.
    static void insertGlobalSymbol(PatternRewriter &rewriter, ModuleOp op, StringRef key,
                                   std::optional<StringRef> value)
    {
        auto *context = op.getContext();
        LLVM::GlobalOp glb = op.lookupSymbol<LLVM::GlobalOp>(key);

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(op.getBody());
        auto shortTy = IntegerType::get(context, 8);
        if (!glb) {
            if (!value) {
                rewriter.create<LLVM::GlobalOp>(op.getLoc(), shortTy,
                                                /*isConstant=*/true, LLVM::Linkage::Linkonce, key,
                                                IntegerAttr::get(shortTy, 0));
            }
            else {
                rewriter.create<LLVM::GlobalOp>(
                    op.getLoc(), LLVM::LLVMArrayType::get(shortTy, value->size()), true,
                    LLVM::Linkage::Linkonce, key, rewriter.getStringAttr(*value));
            }
        }
    }

    /// This functions inserts a llvm global (with a block), it is used to tell Enzyme
    /// how to deal with function with custom definition like mlir allocation and free.
    static LLVM::GlobalOp insertEnzymeFunctionLike(PatternRewriter &rewriter, ModuleOp op,
                                                   StringRef key, StringRef name,
                                                   StringRef originalName)
    {
        auto *context = op.getContext();
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(op.getBody());

        LLVM::GlobalOp glb = op.lookupSymbol<LLVM::GlobalOp>(key);

        auto ptrType = LLVM::LLVMPointerType::get(context);
        if (glb) {
            return glb;
        }
        glb = rewriter.create<LLVM::GlobalOp>(op.getLoc(), LLVM::LLVMArrayType::get(ptrType, 2),
                                              /*isConstant=*/false, LLVM::Linkage::External, key,
                                              nullptr);

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

    static LLVM::GlobalOp insertEnzymeCustomGradient(OpBuilder &builder, ModuleOp moduleOp,
                                                     Location loc, func::FuncOp originalFunc,
                                                     func::FuncOp augmentedPrimal,
                                                     func::FuncOp gradient)
    {
        MLIRContext *context = moduleOp.getContext();
        OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        std::string key = (enzyme_custom_gradient_key + originalFunc.getName()).str();
        auto customGradient = moduleOp.lookupSymbol<LLVM::GlobalOp>(key);
        if (customGradient) {
            return customGradient;
        }

        auto ptrType = LLVM::LLVMPointerType::get(context);
        auto resultType = LLVM::LLVMArrayType::get(ptrType, 3);
        customGradient = builder.create<LLVM::GlobalOp>(
            loc, resultType,
            /*isConstant=*/false, LLVM::Linkage::External, key, /*address space=*/nullptr);
        builder.createBlock(&customGradient.getInitializerRegion());
        Value origFnPtr = builder.create<func::ConstantOp>(loc, originalFunc.getFunctionType(),
                                                           originalFunc.getName());
        Value augFnPtr = builder.create<func::ConstantOp>(loc, augmentedPrimal.getFunctionType(),
                                                          augmentedPrimal.getName());
        Value gradFnPtr =
            builder.create<func::ConstantOp>(loc, gradient.getFunctionType(), gradient.getName());
        SmallVector<Value> fnPtrs{origFnPtr, augFnPtr, gradFnPtr};
        Value result = builder.create<LLVM::UndefOp>(loc, resultType);
        for (const auto &[idx, fnPtr] : llvm::enumerate(fnPtrs)) {
            Value casted =
                builder.create<UnrealizedConversionCastOp>(loc, ptrType, fnPtr).getResult(0);
            result = builder.create<LLVM::InsertValueOp>(loc, result, casted, idx);
        }

        builder.create<LLVM::ReturnOp>(loc, result);

        return customGradient;
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
