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

        wrappedCallee =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        Block *entryBlock = wrappedCallee.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Get the arguments
        SmallVector<Value> callArgs(wrappedCallee.getArguments().begin(),
                                    wrappedCallee.getArguments().end() - callee.getNumResults());

        for (auto [arg, originalType] : llvm::zip(callArgs, originalArgTypes)) {
            if (isa<LLVM::LLVMPointerType>(arg.getType())) {
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
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));

        // Create mlir memref to llvm
        StringRef allocFnName = "_mlir_memref_to_llvm_alloc";
        Type allocFnSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx),
                                                            {IntegerType::get(ctx, 64)});
        LLVM::LLVMFuncOp allocFnDecl =
            ensureFunctionDeclaration(rewriter, op, allocFnName, allocFnSignature);
        
        // Create mlir memref to free
        StringRef freeFnName = "_mlir_memref_to_llvm_free";
        Type freeFnSignature = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx), {});
        ensureFunctionDeclaration(rewriter, op, freeFnName, freeFnSignature);
                
        // Defyne some Enzyme Globals
        // malloc
        insertFunctionName(rewriter, op, "mallocname", StringRef("malloc", 7));
        insertEnzymeFunctionLike(rewriter, op, "__enzyme_function_like_malloc", "mallocname", "_mlir_memref_to_llvm_alloc");

        // free
        insertFunctionName(rewriter, op, "freename", StringRef("free", 5));
        insertEnzymeFunctionLike(rewriter, op, "__enzyme_function_like_free", "freename", "_mlir_memref_to_llvm_free");

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

        // Get the diff arg indices
        std::vector<size_t> diffArgIndices{0};
        if (op.getDiffArgIndices().has_value()) {
            auto range = op.getDiffArgIndices().value().getValues<size_t>();
            diffArgIndices = std::vector<size_t>(range.begin(), range.end());
        }

        int index = 0;
        ValueRange dataIn = adaptor.getDataIn();
        // Add the arguments and their shadow on data in
        for (auto [memrefArg, llvmMemrefArg] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
            auto it = std::find(diffArgIndices.begin(), diffArgIndices.end(), index);
            if (it == diffArgIndices.end()) {
                auto const_global = getOrInsertEnzymeConstDecl(rewriter, op);
                auto llvmI8PtrTy =
                    LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
                auto enzymeConst =
                    rewriter.create<LLVM::AddressOfOp>(op->getLoc(), llvmI8PtrTy, const_global);
                callArgs.push_back(enzymeConst);
                // Add the argument to call arg as a pointer
                auto newMemrefArg = rewriter.create<LLVM::AllocaOp>(
                    loc, LLVM::LLVMPointerType::get(llvmMemrefArg.getType()), c1);
                rewriter.create<LLVM::StoreOp>(loc, llvmMemrefArg, newMemrefArg);
                callArgs.push_back(newMemrefArg);
            }
            else {
                auto position = std::distance(diffArgIndices.begin(), it);
                auto llvmDataIn = dataIn[position];
                // Get information about the memref arg
                MemRefType memrefArgType = memrefArg.getType().cast<MemRefType>();
                size_t sizeShape = memrefArgType.getShape().size();
                size_t value = memrefArgType.getElementTypeBitWidth() / 8;
                auto memrefArgSize =
                    rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64), value);

                // Add the argument to call arg as a pointer
                auto newMemrefArg = rewriter.create<LLVM::AllocaOp>(
                    loc, LLVM::LLVMPointerType::get(llvmMemrefArg.getType()), c1);
                rewriter.create<LLVM::StoreOp>(loc, llvmMemrefArg, newMemrefArg);
                callArgs.push_back(newMemrefArg);

                // Shadow of args
                Value bufferSize;

                if (sizeShape == 0) {
                    bufferSize =
                        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
                }
                else {
                    Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemrefArg, 2);

                    SmallVector<int64_t> vectorIndices{3, 0};

                    Value sizeArray =
                        rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemrefArg, vectorIndices);

                    vectorIndices[0] = 4;
                    Value strideArray =
                        rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemrefArg, vectorIndices);

                    bufferSize = rewriter.create<LLVM::MulOp>(loc, sizeArray, strideArray);
                    bufferSize = rewriter.create<LLVM::AddOp>(loc, offset, bufferSize);
                }

                Value bufferMemSize = rewriter.create<LLVM::MulOp>(loc, bufferSize, memrefArgSize);

                Value buffer =
                    rewriter.create<LLVM::CallOp>(loc, allocFnDecl, bufferMemSize).getResult();
                // Set value to 0 (gradients)
                Value c0 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
                // Value c0float = rewriter.create<LLVM::ConstantOp>(loc,
                // rewriter.getF64FloatAttr(0.0));
                rewriter.create<LLVM::CallOp>(loc, memsetFnDecl,
                                              ArrayRef<Value>{buffer, c0, bufferMemSize});

                auto llvmInsert0 = rewriter.create<LLVM::InsertValueOp>(loc, llvmDataIn, buffer, 0);
                auto llvmInsert1 =
                    rewriter.create<LLVM::InsertValueOp>(loc, llvmInsert0, buffer, 1);

                Value shadowPtr = rewriter.create<LLVM::AllocaOp>(
                    loc, LLVM::LLVMPointerType::get(llvmDataIn.getType()), c1);
                rewriter.create<LLVM::StoreOp>(loc, llvmInsert1, shadowPtr);
                callArgs.push_back(shadowPtr);
            }
            index++;
        }

        // Results of callee and their shadow
        SmallVector<Value> calleeArgs(op.getArgs().begin(), op.getArgs().end());
        ValueRange results = rewriter.create<func::CallOp>(loc, callee, calleeArgs).getResults();

        // Lower callee results to to llvm
        SmallVector<Type> resCalleeTypes(callee.getResultTypes().begin(),
                                         callee.getResultTypes().end());
        for (auto resTypeIt = resCalleeTypes.begin(); resTypeIt < resCalleeTypes.end();
             resTypeIt++) {
            Type llvmResType = llvmTypeConverter.convertType(*resTypeIt);
            if (!llvmResType)
                emitError(loc, "Could not convert argmap result to LLVM type: ") << *resTypeIt;
            *resTypeIt = llvmResType;
        }

        ValueRange llvmresults =
            rewriter.create<UnrealizedConversionCastOp>(loc, resCalleeTypes, results).getResults();

        // Store the results
        for (auto [result, llvmresult, llvmshadowresult] :
             llvm::zip(results, llvmresults, adaptor.getQuantumJacobian())) {
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
        rewriter.eraseOp(op);
        return success();
    }

  private:
    static FlatSymbolRefAttr getOrInsertEnzymeConstDecl(PatternRewriter &rewriter, Operation *op)
    {
        // Copyright (C) 2023 - Jacob Mai Peng
        // https://github.com/pengmai/lagrad/blob/main/lib/LAGrad/LowerToLLVM.cpp
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        auto *context = moduleOp.getContext();
        if (moduleOp.lookupSymbol<LLVM::GlobalOp>("enzyme_const")) {
            return SymbolRefAttr::get(context, "enzyme_const");
        }

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto shortTy = IntegerType::get(context, 8);
        rewriter.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                                        /*isConstant=*/true, LLVM::Linkage::Linkonce,
                                        "enzyme_const", IntegerAttr::get(shortTy, 0));
        return SymbolRefAttr::get(context, "enzyme_const");
    }

  private:
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
private:
    static LLVM::GlobalOp insertEnzymeFunctionLike(PatternRewriter &rewriter, Operation *op, StringRef key, StringRef name, StringRef originalName)
    {
        ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
        auto *context = moduleOp.getContext();
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        LLVM::GlobalOp glb = moduleOp.lookupSymbol<LLVM::GlobalOp>(key);
        auto ptrType = LLVM::LLVMPointerType::get(context);
        if (!glb) {
            glb = rewriter.create<LLVM::GlobalOp>(moduleOp.getLoc(), LLVM::LLVMArrayType::get(ptrType, 2), /*isConstant=*/false, LLVM::Linkage::External,
                key, nullptr);
        }
        auto *contextGlb = glb.getContext();
        Block *block = new Block();
        glb.getInitializerRegion().push_back(block);
        rewriter.setInsertionPointToStart(block);

        auto llvmPtr = LLVM::LLVMPointerType::get(contextGlb);

        // Get original global name
        auto originalNameRefAttr = SymbolRefAttr::get(contextGlb, originalName);
        auto originalGlobal = rewriter.create<LLVM::AddressOfOp>(glb.getLoc(), llvmPtr, originalNameRefAttr);

                // Get global name
        auto nameRefAttr = SymbolRefAttr::get(contextGlb, name);
        auto enzymeGlobal = rewriter.create<LLVM::AddressOfOp>(glb.getLoc(), llvmPtr, nameRefAttr);

        auto undefArray = rewriter.create<LLVM::UndefOp>(glb.getLoc(), LLVM::LLVMArrayType::get(ptrType, 2));
        Value llvmInsert0 = rewriter.create<LLVM::InsertValueOp>(glb.getLoc(), undefArray, originalGlobal, 0);
        Value llvmInsert1 = rewriter.create<LLVM::InsertValueOp>(glb.getLoc(), llvmInsert0, enzymeGlobal, 1);
        rewriter.create<LLVM::ReturnOp>(glb.getLoc(), llvmInsert1);
        return glb;
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
