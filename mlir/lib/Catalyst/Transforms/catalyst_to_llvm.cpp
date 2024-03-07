// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace {

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

enum NumericType : int8_t {
    index = 0,
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    c64,
    c128,
};

std::optional<int8_t> encodeNumericType(Type elemType)
{
    int8_t typeEncoding;
    if (isa<IndexType>(elemType)) {
        typeEncoding = NumericType::index;
    }
    else if (auto intType = dyn_cast<IntegerType>(elemType)) {
        switch (intType.getWidth()) {
        case 1:
            typeEncoding = NumericType::i1;
            break;
        case 8:
            typeEncoding = NumericType::i8;
            break;
        case 16:
            typeEncoding = NumericType::i16;
            break;
        case 32:
            typeEncoding = NumericType::i32;
            break;
        case 64:
            typeEncoding = NumericType::i64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto floatType = dyn_cast<FloatType>(elemType)) {
        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::f32;
            break;
        case 64:
            typeEncoding = NumericType::f64;
            break;
        default:
            return std::nullopt;
        }
    }
    else if (auto cmplxType = dyn_cast<ComplexType>(elemType)) {
        auto floatType = dyn_cast<FloatType>(cmplxType.getElementType());
        if (!floatType)
            return std::nullopt;

        switch (floatType.getWidth()) {
        case 32:
            typeEncoding = NumericType::c64;
            break;
        case 64:
            typeEncoding = NumericType::c128;
            break;
        default:
            return std::nullopt;
        }
    }
    else {
        return std::nullopt;
    }
    return typeEncoding;
}

// Encodes memref as LLVM struct value:
//
// {
//    i64 rank,
//    void* memref_descriptor,
//    i8 type_encoding
// }
Value EncodeOpaqueMemRef(Location loc, PatternRewriter &rewriter, MemRefType memrefType,
                         Type llvmMemrefType, Value memrefLlvm)
{
    auto ctx = rewriter.getContext();

    // Encoded memref type: !llvm.struct<(i64, ptr<i8>, i8)>.
    Type i8 = rewriter.getI8Type();
    Type i64 = rewriter.getI64Type();
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    auto type = LLVM::LLVMStructType::getLiteral(ctx, {i64, ptr, i8});

    std::optional<int8_t> elementDtype = encodeNumericType(memrefType.getElementType());

    // Create values for filling encoded memref struct.
    Value dtype =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8IntegerAttr(elementDtype.value()));
    Value rank =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(memrefType.getRank()));

    // Construct encoded memref value.
    Value memref = rewriter.create<LLVM::UndefOp>(loc, type);
    // Rank
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, rank, 0);

    // Memref
    Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    Value memrefPtr = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), llvmMemrefType, c1);
    rewriter.create<LLVM::StoreOp>(loc, memrefLlvm, memrefPtr);
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, memrefPtr, 1);

    // Dtype
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, dtype, 2);

    return memref;
}

struct PrintOpPattern : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {

        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type voidPtrType = LLVM::LLVMPointerType::get(ctx);

        if (op.getConstVal().has_value()) {
            ModuleOp mod = op->getParentOfType<ModuleOp>();

            StringRef qirName = "__catalyst__rt__print_string";

            Type charPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, charPtrType);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            StringRef stringValue = op.getConstVal().value();
            std::string symbolName = std::to_string(std::hash<std::string>()(stringValue.str()));
            Value global = getGlobalString(loc, rewriter, symbolName, stringValue, mod);
            rewriter.create<LLVM::CallOp>(loc, fnDecl, global);
            rewriter.eraseOp(op);
        }
        else {
            StringRef qirName = "__catalyst__rt__print_tensor";

            // C interface for the print function is an unranked & opaque memref descriptor:
            // {
            //    i64 rank,
            //    void* memref_descriptor,
            //    i8 type_encoding
            // }
            // Where the type_encoding is a simple enum for all supported numeric types:
            //   i1, i16, i32, i64, f32, f64, c64, c128 (see runtime Types.h)
            Type structType = LLVM::LLVMStructType::getLiteral(
                ctx, {IntegerType::get(ctx, 64), voidPtrType, IntegerType::get(ctx, 8)});
            Type structPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
            SmallVector<Type> argTypes{structPtrType, IntegerType::get(ctx, 1)};
            Type qirSignature = LLVM::LLVMFunctionType::get(voidType, argTypes);
            LLVM::LLVMFuncOp fnDecl =
                ensureFunctionDeclaration(rewriter, op, qirName, qirSignature);

            Value memref = op.getVal();
            MemRefType memrefType = cast<MemRefType>(memref.getType());
            Value llvmMemref = adaptor.getVal();
            Type llvmMemrefType = llvmMemref.getType();
            Value structValue =
                EncodeOpaqueMemRef(loc, rewriter, memrefType, llvmMemrefType, llvmMemref);

            Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));

            Value structPtr = rewriter.create<LLVM::AllocaOp>(
                loc, LLVM::LLVMPointerType::get(rewriter.getContext()), structType, c1);
            rewriter.create<LLVM::StoreOp>(loc, structValue, structPtr);

            Value printDescriptor = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(),
                                                                      op.getPrintDescriptor());
            SmallVector<Value> callArgs{structPtr, printDescriptor};
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, callArgs);
        }

        return success();
    }
};

// Encodes memref as LLVM struct value:
//
// {
//    i64 rank,
//    void* data,
//    i8 type_encoding
// }
Value EncodeDataMemRef(Location loc, PatternRewriter &rewriter, MemRefType memrefType,
                       Type llvmMemrefType, Value memrefLlvm)
{
    auto ctx = rewriter.getContext();

    // Encoded memref type: !llvm.struct<(i64, ptr<i8>, i8)>.
    Type i8 = rewriter.getI8Type();
    Type i64 = rewriter.getI64Type();
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    auto type = LLVM::LLVMStructType::getLiteral(ctx, {i64, ptr, i8});

    std::optional<int8_t> elementDtype = encodeNumericType(memrefType.getElementType());

    // Create values for filling encoded memref struct.
    Value dtype =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8IntegerAttr(elementDtype.value()));
    Value rank =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(memrefType.getRank()));

    // Construct encoded memref value.
    Value memref = rewriter.create<LLVM::UndefOp>(loc, type);
    // Rank
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, rank, 0);

    // Memref data
    MemRefDescriptor desc = MemRefDescriptor(memrefLlvm);
    Value c0 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    Value data = rewriter.create<LLVM::GEPOp>(loc, ptr, memrefType.getElementType(),
                                              desc.alignedPtr(rewriter, loc), c0, true);
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, data, 1);

    // Dtype
    memref = rewriter.create<LLVM::InsertValueOp>(loc, memref, dtype, 2);

    return memref;
}

struct CustomCallOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = op.getContext();
        Location loc = op.getLoc();
        // Create function
        Type ptr = LLVM::LLVMPointerType::get(ctx);

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto point = rewriter.saveInsertionPoint();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        LLVM::LLVMFuncOp customCallFnOp = mlir::LLVM::lookupOrCreateFn(
            mod, op.getCallTargetName(), {/*args=*/ptr, /*rets=*/ptr}, /*ret_type=*/voidType);
        customCallFnOp.setPrivate();
        rewriter.restoreInsertionPoint(point);

        // Setup args and res
        int32_t numberArg = op.getNumberOriginalArgAttr()[0];
        SmallVector<Value> operands = op.getOperands();
        SmallVector<Value> args = {operands.begin(), operands.begin() + numberArg};
        SmallVector<Value> res = {operands.begin() + numberArg, operands.end()};

        SmallVector<Value> operandsConverted = adaptor.getOperands();
        SmallVector<Value> argsConverted = {operandsConverted.begin(),
                                            operandsConverted.begin() + numberArg};
        SmallVector<Value> resConverted = {operandsConverted.begin() + numberArg,
                                           operandsConverted.end()};

        // Encode args
        Value c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
        SmallVector<LLVM::AllocaOp> encodedArgs;
        for (auto tuple : llvm::zip(args, argsConverted)) {
            auto memref_type = std::get<0>(tuple).getType().cast<MemRefType>();
            Type llvmMemrefType = std::get<1>(tuple).getType();
            auto encodedArg =
                EncodeDataMemRef(loc, rewriter, memref_type, llvmMemrefType, std::get<1>(tuple));
            LLVM::AllocaOp alloca =
                rewriter.create<LLVM::AllocaOp>(loc, ptr, encodedArg.getType(), c1, 0);
            rewriter.create<LLVM::StoreOp>(loc, encodedArg, alloca);
            encodedArgs.push_back(alloca);
        }

        // We store encoded arguments as `!llvm.array<ptr x len>`.
        size_t len = encodedArgs.size();
        Type typeArgs = LLVM::LLVMArrayType::get(ptr, len);

        // Prepare an array for encoded arguments.
        Value arrArgs = rewriter.create<LLVM::UndefOp>(loc, typeArgs);
        auto insertValueArgs = [&](Value value, int64_t offset) {
            arrArgs = rewriter.create<LLVM::InsertValueOp>(loc, arrArgs, value, offset);
        };
        // Store pointer to encoded arguments into the allocated storage.
        for (const auto &pair : llvm::enumerate(encodedArgs)) {
            int64_t offset = pair.index();
            insertValueArgs(pair.value(), offset);
        }
        // Get allocation for packed arguments pointers.
        LLVM::AllocaOp alloca = rewriter.create<LLVM::AllocaOp>(loc, ptr, arrArgs.getType(), c1, 0);

        // Store constructed arguments pointers array into the alloca.
        rewriter.create<LLVM::StoreOp>(loc, arrArgs, alloca);

        // Alloca that encodes the custom call arguments.
        auto encodedArguments = alloca.getResult();

        // Results: ########

        // Encode all returns as a set of pointers
        SmallVector<LLVM::AllocaOp> encodedRess;
        for (auto tuple : llvm::zip(res, resConverted)) {
            auto memref_type = std::get<0>(tuple).getType().cast<MemRefType>();
            Type llvmMemrefType = std::get<1>(tuple).getType();
            auto encodedRes =
                EncodeDataMemRef(loc, rewriter, memref_type, llvmMemrefType, std::get<1>(tuple));
            LLVM::AllocaOp alloca =
                rewriter.create<LLVM::AllocaOp>(loc, ptr, encodedRes.getType(), c1, 0);
            rewriter.create<LLVM::StoreOp>(loc, encodedRes, alloca);
            encodedRess.push_back(alloca);
        }

        // We store encoded results as `!llvm.array<ptr x len>`.
        size_t lenRes = encodedRess.size();
        Type typeRes = LLVM::LLVMArrayType::get(ptr, lenRes);

        // Prepare an array for encoding results.
        Value arrRes = rewriter.create<LLVM::UndefOp>(loc, typeRes);
        auto insertValueRes = [&](Value value, int64_t offset) {
            arrRes = rewriter.create<LLVM::InsertValueOp>(loc, arrRes, value, offset);
        };

        // Store encoded results into the allocated storage.
        for (const auto &pair : llvm::enumerate(encodedRess)) {
            int64_t offset = pair.index();
            insertValueRes(pair.value(), offset);
        }

        // Get allocation for packed results pointers.
        LLVM::AllocaOp allocaRes =
            rewriter.create<LLVM::AllocaOp>(loc, ptr, arrRes.getType(), c1, 0);

        // Store constructed results pointers array on the stack.
        rewriter.create<LLVM::StoreOp>(loc, arrRes, allocaRes);

        // Alloca that encodes the custom call returns.
        auto encodedResults = allocaRes.getResult();
        // Call op
        SmallVector<Value> callArgs{encodedArguments, encodedResults};
        rewriter.create<LLVM::CallOp>(loc, customCallFnOp, callArgs);
        rewriter.eraseOp(op);
        return success();
    }
};

struct PythonCallOpPattern : public OpConversionPattern<PythonCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PythonCallOp op, PythonCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MLIRContext *ctx = op.getContext();
        Location loc = op.getLoc();

        // Create function
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto point = rewriter.saveInsertionPoint();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        Type i64 = rewriter.getI64Type();
        LLVM::LLVMFuncOp customCallFnOp = mlir::LLVM::lookupOrCreateFn(
            mod, "pyregistry", {/*args=*/i64}, /*ret_type=*/voidType);
        customCallFnOp.setPrivate();
        rewriter.restoreInsertionPoint(point);

        auto identAttr = op.getIdentifier();
        auto ident = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(identAttr));
        SmallVector<Value> callArgs{ident};
        rewriter.create<LLVM::CallOp>(loc, customCallFnOp, callArgs);
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_CATALYSTCONVERSIONPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct CatalystConversionPass : impl::CatalystConversionPassBase<CatalystConversionPass> {
    using CatalystConversionPassBase::CatalystConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LLVMTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        patterns.add<CustomCallOpPattern>(typeConverter, context);
        patterns.add<PythonCallOpPattern>(typeConverter, context);
        patterns.add<PrintOpPattern>(typeConverter, context);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<CatalystDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createCatalystConversionPass()
{
    return std::make_unique<catalyst::CatalystConversionPass>();
}

} // namespace catalyst
