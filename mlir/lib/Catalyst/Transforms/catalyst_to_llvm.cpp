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
#include "Catalyst/Transforms/Patterns.h"
#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "Catalyst/Utils/StaticAllocas.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Utils.h"

using namespace mlir;
using namespace catalyst;

namespace {

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, StringRef value,
                      ModuleOp mod)
{
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size());
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter); // to reset the insertion point
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = LLVM::GlobalOp::create(rewriter, loc, type, true, LLVM::Linkage::Internal, key,
                                     rewriter.getStringAttr(value));
    }
    return LLVM::GEPOp::create(rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                               type, LLVM::AddressOfOp::create(rewriter, loc, glb),
                               ArrayRef<LLVM::GEPArg>{0, 0}, LLVM::GEPNoWrapFlags::inbounds);
}

// Get or create a `internal constant !llvm.array<N x i64>` global.
// And return a `!llvm.ptr` to its first element.
//
// llvm.mlir.global internal constant @<key>(dense<[v0, v1, ...]> : tensor<NxI64>)
//     : !llvm.array<N x i64>
//
// Call site:
//
// %addr = llvm.mlir.addressof @<key> : !llvm.ptr
// %ptr  = llvm.getelementptr inbounds %addr[0, 0]
//             : (!llvm.ptr) -> !llvm.ptr, !llvm.array<N x i64>
//
Value getGlobalI64Array(Location loc, OpBuilder &rewriter, StringRef key, ArrayRef<int64_t> values,
                        ModuleOp mod)
{
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    // Skip and return a null pointer if the array is empty.
    if (values.empty()) {
        return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }
    Type i64Ty = rewriter.getI64Type();
    auto arrTy = LLVM::LLVMArrayType::get(i64Ty, values.size());
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    // Create a new global if it doesn't exist.
    if (!glb) {
        auto tensorTy = RankedTensorType::get({static_cast<int64_t>(values.size())}, i64Ty);
        auto valuesAttr = DenseElementsAttr::get(tensorTy, values);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        glb = LLVM::GlobalOp::create(rewriter, loc, arrTy, /*isConstant=*/true,
                                     LLVM::Linkage::Internal, key, valuesAttr);
    }
    return LLVM::GEPOp::create(rewriter, loc, ptrTy, arrTy,
                               LLVM::AddressOfOp::create(rewriter, loc, glb),
                               ArrayRef<LLVM::GEPArg>{0, 0}, LLVM::GEPNoWrapFlags::inbounds);
}

// Allocate a stack buffer of `!llvm.array<N x ptr>`.
//
// Outputs:
//
//   %slot = llvm.alloca ... : !llvm.array<N x ptr>
//   %a0   = llvm.undef : !llvm.array<N x ptr>
//   %a1   = llvm.insertvalue %ptr0, %a0[0] : !llvm.array<N x ptr>
//   %a2   = llvm.insertvalue %ptr1, %a1[1] : !llvm.array<N x ptr>
//   ...
//   llvm.store %aN, %slot : !llvm.array<N x ptr>, !llvm.ptr
//
Value buildStackPtrArray(Location loc, RewriterBase &rewriter, ArrayRef<Value> ptrs)
{
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    if (ptrs.empty()) {
        return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    }
    auto arrTy = LLVM::LLVMArrayType::get(ptrTy, ptrs.size());
    Value alloca = getStaticAlloca(loc, rewriter, arrTy, 1);
    Value arr = LLVM::UndefOp::create(rewriter, loc, arrTy);
    for (auto [i, p] : llvm::enumerate(ptrs)) {
        arr = LLVM::InsertValueOp::create(rewriter, loc, arr, p, SmallVector<int64_t>{(int64_t)i});
    }
    LLVM::StoreOp::create(rewriter, loc, arr, alloca);
    return alloca;
}

// Calculate the byte size of one memref element.
// For complex<T>, the size is 2 * sizeof(T). 1 for real, 1 for imaginary.
int64_t memrefElemSizeBytes(MemRefType ty)
{
    Type elem = ty.getElementType();
    if (auto cplx = dyn_cast<ComplexType>(elem)) {
        return 2 * ((cplx.getElementType().getIntOrFloatBitWidth() + 7) / 8);
    }
    return (elem.getIntOrFloatBitWidth() + 7) / 8;
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
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8IntegerAttr(elementDtype.value()));
    Value rank =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(memrefType.getRank()));

    // Construct encoded memref value.
    Value memref = LLVM::UndefOp::create(rewriter, loc, type);
    // Rank
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, rank, SmallVector<int64_t>{0});

    // Memref
    Value memrefPtr = getStaticAlloca(loc, rewriter, llvmMemrefType, 1);
    LLVM::StoreOp::create(rewriter, loc, memrefLlvm, memrefPtr);
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, memrefPtr, SmallVector<int64_t>{1});

    // Dtype
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, dtype, SmallVector<int64_t>{2});

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
            LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
                rewriter, op, qirName, qirSignature);

            StringRef stringValue = op.getConstVal().value();
            std::string symbolName = std::to_string(std::hash<std::string>()(stringValue.str()));
            Value global = getGlobalString(loc, rewriter, symbolName, stringValue, mod);
            LLVM::CallOp::create(rewriter, loc, fnDecl, global);
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
            LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
                rewriter, op, qirName, qirSignature);

            Value memref = op.getVal();
            MemRefType memrefType = cast<MemRefType>(memref.getType());
            Value llvmMemref = adaptor.getVal();
            Type llvmMemrefType = llvmMemref.getType();
            Value structValue =
                EncodeOpaqueMemRef(loc, rewriter, memrefType, llvmMemrefType, llvmMemref);

            Value structPtr = getStaticAlloca(loc, rewriter, structType, 1);
            LLVM::StoreOp::create(rewriter, loc, structValue, structPtr);

            Value printDescriptor = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                                             op.getPrintDescriptor());
            SmallVector<Value> callArgs{structPtr, printDescriptor};
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, callArgs);
        }

        return success();
    }
};

struct AssertionOpPattern : public OpConversionPattern<AssertionOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AssertionOp op, AssertionOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        StringRef qirName = "__catalyst__rt__assert_bool";

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type int1Type = IntegerType::get(ctx, 1);
        Type charPtrType = LLVM::LLVMPointerType::get(ctx);

        SmallVector<Type> argTypes{int1Type, charPtrType};
        Type assertSignature = LLVM::LLVMFunctionType::get(voidType, argTypes);

        ModuleOp mod = op->getParentOfType<ModuleOp>();
        LLVM::LLVMFuncOp assertFunc = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, qirName, assertSignature);

        Value assertionDescriptor = adaptor.getAssertion();

        StringRef errorMessage = op.getError();
        std::string symbolName = std::to_string(std::hash<std::string>()(errorMessage.str()));
        Value globalString = getGlobalString(loc, rewriter, symbolName, errorMessage, mod);

        SmallVector<Value> callArgs{assertionDescriptor, globalString};
        LLVM::CallOp::create(rewriter, loc, assertFunc, callArgs);

        rewriter.eraseOp(op);

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
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI8IntegerAttr(elementDtype.value()));
    Value rank =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(memrefType.getRank()));

    // Construct encoded memref value.
    Value memref = LLVM::UndefOp::create(rewriter, loc, type);
    // Rank
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, rank, SmallVector<int64_t>{0});

    // Memref data
    MemRefDescriptor desc = MemRefDescriptor(memrefLlvm);
    Value c0 = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));
    Value data =
        LLVM::GEPOp::create(rewriter, loc, ptr, memrefType.getElementType(),
                            desc.alignedPtr(rewriter, loc), c0, LLVM::GEPNoWrapFlags::inbounds);
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, data, SmallVector<int64_t>{1});

    // Dtype
    memref = LLVM::InsertValueOp::create(rewriter, loc, memref, dtype, SmallVector<int64_t>{2});

    return memref;
}

struct CustomCallOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Remote-dispatch custom_calls are lowered by their own dedicated patterns below.
        StringRef name = op.getCallTargetName();
        if (name == "remote_call" || name == "remote_lib_call" || name == "remote_open" ||
            name == "remote_send_binary") {
            return failure();
        }
        MLIRContext *ctx = op.getContext();
        Location loc = op.getLoc();
        // Create function
        Type ptr = LLVM::LLVMPointerType::get(ctx);

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        auto point = rewriter.saveInsertionPoint();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        LLVM::LLVMFuncOp customCallFnOp =
            mlir::LLVM::lookupOrCreateFn(rewriter, mod, op.getCallTargetName(),
                                         {/*args=*/ptr, /*rets=*/ptr},
                                         /*ret_type=*/voidType)
                .value();
        customCallFnOp.setPrivate();
        rewriter.restoreInsertionPoint(point);

        // Setup args and res
        int32_t numberArg = op.getNumberOriginalArg().value_or(0);
        SmallVector<Value> operands = op.getOperands();
        SmallVector<Value> args = {operands.begin(), operands.begin() + numberArg};
        SmallVector<Value> res = {operands.begin() + numberArg, operands.end()};

        SmallVector<Value> operandsConverted = adaptor.getOperands();
        SmallVector<Value> argsConverted = {operandsConverted.begin(),
                                            operandsConverted.begin() + numberArg};
        SmallVector<Value> resConverted = {operandsConverted.begin() + numberArg,
                                           operandsConverted.end()};

        // Encode args
        SmallVector<LLVM::AllocaOp> encodedArgs;
        for (auto tuple : llvm::zip(args, argsConverted)) {
            auto memref_type = cast<MemRefType>(std::get<0>(tuple).getType());
            Type llvmMemrefType = std::get<1>(tuple).getType();
            auto encodedArg =
                EncodeDataMemRef(loc, rewriter, memref_type, llvmMemrefType, std::get<1>(tuple));
            LLVM::AllocaOp alloca = getStaticAlloca(loc, rewriter, encodedArg.getType(), 1);
            LLVM::StoreOp::create(rewriter, loc, encodedArg, alloca);
            encodedArgs.push_back(alloca);
        }

        // We store encoded arguments as `!llvm.array<ptr x len>`.
        size_t len = encodedArgs.size();
        Type typeArgs = LLVM::LLVMArrayType::get(ptr, len);

        // Prepare an array for encoded arguments.
        Value arrArgs = LLVM::UndefOp::create(rewriter, loc, typeArgs);
        auto insertValueArgs = [&](Value value, int64_t offset) {
            arrArgs = LLVM::InsertValueOp::create(rewriter, loc, arrArgs, value, offset);
        };
        // Store pointer to encoded arguments into the allocated storage.
        for (const auto &pair : llvm::enumerate(encodedArgs)) {
            int64_t offset = pair.index();
            insertValueArgs(pair.value(), offset);
        }
        // Get allocation for packed arguments pointers.
        LLVM::AllocaOp alloca = getStaticAlloca(loc, rewriter, arrArgs.getType(), 1);

        // Store constructed arguments pointers array into the alloca.
        LLVM::StoreOp::create(rewriter, loc, arrArgs, alloca);

        // Alloca that encodes the custom call arguments.
        auto encodedArguments = alloca.getResult();

        // Results: ########

        // Encode all returns as a set of pointers
        SmallVector<LLVM::AllocaOp> encodedRess;
        for (auto tuple : llvm::zip(res, resConverted)) {
            auto memref_type = cast<MemRefType>(std::get<0>(tuple).getType());
            Type llvmMemrefType = std::get<1>(tuple).getType();
            auto encodedRes =
                EncodeDataMemRef(loc, rewriter, memref_type, llvmMemrefType, std::get<1>(tuple));
            LLVM::AllocaOp alloca = getStaticAlloca(loc, rewriter, encodedRes.getType(), 1);
            LLVM::StoreOp::create(rewriter, loc, encodedRes, alloca);
            encodedRess.push_back(alloca);
        }

        // We store encoded results as `!llvm.array<ptr x len>`.
        size_t lenRes = encodedRess.size();
        Type typeRes = LLVM::LLVMArrayType::get(ptr, lenRes);

        // Prepare an array for encoding results.
        Value arrRes = LLVM::UndefOp::create(rewriter, loc, typeRes);
        auto insertValueRes = [&](Value value, int64_t offset) {
            arrRes = LLVM::InsertValueOp::create(rewriter, loc, arrRes, value, offset);
        };

        // Store encoded results into the allocated storage.
        for (const auto &pair : llvm::enumerate(encodedRess)) {
            int64_t offset = pair.index();
            insertValueRes(pair.value(), offset);
        }

        // Get allocation for packed results pointers.
        LLVM::AllocaOp allocaRes = getStaticAlloca(loc, rewriter, arrRes.getType(), 1);

        // Store constructed results pointers array on the stack.
        LLVM::StoreOp::create(rewriter, loc, arrRes, allocaRes);

        // Alloca that encodes the custom call returns.
        auto encodedResults = allocaRes.getResult();
        // Call op
        SmallVector<Value> callArgs{encodedArguments, encodedResults};
        LLVM::CallOp::create(rewriter, loc, customCallFnOp, callArgs);
        rewriter.eraseOp(op);
        return success();
    }
};

// Rewrite `catalyst.custom_call fn("remote_call") -> ...`
// to three runtime calls:
//
//   __catalyst__remote__open(addr)
//   __catalyst__remote__send_binary(addr,p)
//   __catalyst__remote__launch(addr, "_catalyst_pyface_<callee>",
//                              num_in,  in_descs,  in_ranks,  in_sizes,
//                              num_out, out_descs, out_ranks, out_sizes);
//
struct RemoteCustomCallOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getCallTargetName() != "remote_call") {
            return failure();
        }

        auto addrAttr = op->getAttrOfType<StringAttr>("catalyst.remote_address");
        auto calleeAttr = op->getAttrOfType<StringAttr>("catalyst.remote_kernel_callee");
        if (!addrAttr) {
            llvm::errs() << "remote_call custom_call is missing `catalyst.remote_address`\n";
            return failure();
        }
        if (!calleeAttr) {
            llvm::errs() << "remote_call custom_call is missing `catalyst.remote_kernel_callee`\n";
            return failure();
        }

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i64Ty = rewriter.getI64Type();
        Type voidTy = LLVM::LLVMVoidType::get(ctx);
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        // Declare extern runtime entry points.
        Type launchSig = LLVM::LLVMFunctionType::get(
            voidTy, {ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy});
        LLVM::LLVMFuncOp launchFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__launch", launchSig);

        // We need the address string globally so the launch knows which session to dispatch to.
        std::string callee = calleeAttr.getValue().str();
        Value addrPtr = getGlobalString(loc, rewriter, "remote_addr_" + callee,
                                        addrAttr.getValue().str() + '\0', mod);

        // Get a global string for the symbol name "_catalyst_pyface_<callee>"
        std::string symbolName = "_catalyst_pyface_" + callee;
        std::string symbolKey = "remote_sym_" + callee;
        Value symbolPtr = getGlobalString(loc, rewriter, symbolKey, symbolName + '\0', mod);

        // Spill input descriptor structs to stack allocas
        SmallVector<Value> inputDescPtrs;
        SmallVector<int64_t> inputRanks, inputElemSizes;
        for (auto [origInput, llvmInput] : llvm::zip(op.getOperands(), adaptor.getOperands())) {
            auto memrefTy = cast<MemRefType>(origInput.getType());
            inputRanks.push_back(memrefTy.getRank());
            inputElemSizes.push_back(memrefElemSizeBytes(memrefTy));
            Value alloca = getStaticAlloca(loc, rewriter, llvmInput.getType(), 1);
            LLVM::StoreOp::create(rewriter, loc, llvmInput, alloca);
            inputDescPtrs.push_back(alloca);
        }

        // Allocate stack buffers holding input/output descriptor pointers.
        SmallVector<Value> outputDescPtrs;
        SmallVector<int64_t> outputRanks, outputElemSizes;
        for (Type resultTy : op.getResultTypes()) {
            auto memrefTy = cast<MemRefType>(resultTy);
            outputRanks.push_back(memrefTy.getRank());
            outputElemSizes.push_back(memrefElemSizeBytes(memrefTy));
            Type llvmDescTy = getTypeConverter()->convertType(resultTy);
            Value alloca = getStaticAlloca(loc, rewriter, llvmDescTy, 1);
            outputDescPtrs.push_back(alloca);
        }

        Value inputDescsArr = buildStackPtrArray(loc, rewriter, inputDescPtrs);
        Value outputDescsArr = buildStackPtrArray(loc, rewriter, outputDescPtrs);

        // Get global arrays for ranks / elem-sizes.
        Value inputRanksArr =
            getGlobalI64Array(loc, rewriter, "remote_in_ranks_" + callee, inputRanks, mod);
        Value inputSizesArr =
            getGlobalI64Array(loc, rewriter, "remote_in_sizes_" + callee, inputElemSizes, mod);
        Value outputRanksArr =
            getGlobalI64Array(loc, rewriter, "remote_out_ranks_" + callee, outputRanks, mod);
        Value outputSizesArr =
            getGlobalI64Array(loc, rewriter, "remote_out_sizes_" + callee, outputElemSizes, mod);

        Value numInputs = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(inputDescPtrs.size()));
        Value numOutputs = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(outputDescPtrs.size()));

        LLVM::CallOp::create(rewriter, loc, launchFn,
                             ValueRange{addrPtr, symbolPtr, numInputs, inputDescsArr, inputRanksArr,
                                        inputSizesArr, numOutputs, outputDescsArr, outputRanksArr,
                                        outputSizesArr});

        // Load the runtime-filled output descriptors and replace the op with them.
        SmallVector<Value> results;
        for (auto [descPtr, resultTy] : llvm::zip(outputDescPtrs, op.getResultTypes())) {
            Type llvmDescTy = getTypeConverter()->convertType(resultTy);
            Value loaded = LLVM::LoadOp::create(rewriter, loc, llvmDescTy, descPtr);
            results.push_back(loaded);
        }

        rewriter.replaceOp(op, results);
        return success();
    }
};

// Rewrite the `catalyst.custom_call fn("remote_open")` op to `__catalyst__remote__open(addr)`.
struct RemoteOpenOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getCallTargetName() != "remote_open") {
            return failure();
        }
        auto addrAttr = op->getAttrOfType<StringAttr>("catalyst.remote_address");
        if (!addrAttr) {
            return op->emitOpError("remote_open call is missing `catalyst.remote_address`");
        }

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i64Ty = rewriter.getI64Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        Type openSig = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy});
        LLVM::LLVMFuncOp openFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__open", openSig);

        Value addrPtr = getGlobalString(loc, rewriter, "remote_setup_addr",
                                        addrAttr.getValue().str() + '\0', mod);

        LLVM::CallOp::create(rewriter, loc, openFn, ValueRange{addrPtr});
        rewriter.eraseOp(op);
        return success();
    }
};

// Rewrite the `catalyst.custom_call fn("remote_send_binary")` op to
// `__catalyst__remote__send_binary(addr, path)`.
struct RemoteSendBinaryOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getCallTargetName() != "remote_send_binary") {
            return failure();
        }
        auto addrAttr = op->getAttrOfType<StringAttr>("catalyst.remote_address");
        auto pathAttr = op->getAttrOfType<StringAttr>("catalyst.remote_kernel_path");
        auto calleeAttr = op->getAttrOfType<StringAttr>("catalyst.remote_kernel_callee");
        if (!addrAttr) {
            return op->emitOpError("remote_send_binary call is missing `catalyst.remote_address`");
        }
        if (!pathAttr) {
            return op->emitOpError(
                "remote_send_binary call is missing `catalyst.remote_kernel_path`");
        }
        if (!calleeAttr) {
            return op->emitOpError(
                "remote_send_binary call is missing `catalyst.remote_kernel_callee`");
        }

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = rewriter.getI32Type();
        Type i64Ty = rewriter.getI64Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        Type sendBinSig = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, ptrTy, i32Ty});
        LLVM::LLVMFuncOp sendBinFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__send_binary", sendBinSig);

        std::string callee = calleeAttr.getValue().str();
        Value addrPtr = getGlobalString(loc, rewriter, "remote_addr_" + callee,
                                        addrAttr.getValue().str() + '\0', mod);
        Value pathPtr = getGlobalString(loc, rewriter, "remote_path_" + callee,
                                        pathAttr.getValue().str() + '\0', mod);

        // TODO: Hardcoded format tag for now. (0 as object)
        Value formatTag = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));

        LLVM::CallOp::create(rewriter, loc, sendBinFn, ValueRange{addrPtr, pathPtr, formatTag});
        rewriter.eraseOp(op);
        return success();
    }
};

// Rewrite the `catalyst.custom_call fn("remote_lib_call")` op to
// `__catalyst__remote__call_wrapper(addr, sym, args_buf, args_size, &out, &out_size)`.
struct RemoteLibCallOpPattern : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, CustomCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getCallTargetName() != "remote_lib_call") {
            return failure();
        }

        auto addrAttr = op->getAttrOfType<StringAttr>("catalyst.remote_address");
        auto symAttr = op->getAttrOfType<StringAttr>("catalyst.remote_lib_symbol");
        if (!addrAttr) {
            return op->emitOpError("remote_lib_call is missing `catalyst.remote_address`");
        }
        if (!symAttr) {
            return op->emitOpError("remote_lib_call is missing `catalyst.remote_lib_symbol`");
        }
        std::string sym = symAttr.getValue().str();

        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = rewriter.getI32Type();
        Type i64Ty = rewriter.getI64Type();
        Type voidTy = LLVM::LLVMVoidType::get(ctx);
        Type i8Ty = rewriter.getI8Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        // Declare extern runtime entry points.
        Type callSig =
            LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy});
        LLVM::LLVMFuncOp callFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__call_wrapper", callSig);
        Type freeSig = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
        LLVM::LLVMFuncOp freeFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__free_result", freeSig);

        SmallVector<int64_t> offsets;
        int64_t totalSize = 0;
        for (Type ty : op.getOperandTypes()) {
            int64_t n = primitiveByteSize(ty);
            if (n < 0) {
                return op->emitOpError("unsupported arg type for remote_lib_call: ")
                       << ty << " (supports int/float/index/complex only)";
            }
            offsets.push_back(totalSize);
            totalSize += n;
        }

        Type bufTy = LLVM::LLVMArrayType::get(i8Ty, totalSize > 0 ? totalSize : 1);

        // Symbols
        Value addrPtr = getGlobalString(loc, rewriter, "remote_lib_addr_" + sym,
                                        addrAttr.getValue().str() + '\0', mod);
        Value symPtr = getGlobalString(loc, rewriter, "remote_lib_sym_" + sym, sym + '\0', mod);

        // Alloca args buffer + store each arg.
        Value argsBuf = getStaticAlloca(loc, rewriter, bufTy, 1);
        for (auto [llvmVal, off] : llvm::zip(adaptor.getOperands(), offsets)) {
            Value slot = LLVM::GEPOp::create(rewriter, loc, ptrTy, bufTy, argsBuf,
                                             ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(off)},
                                             LLVM::GEPNoWrapFlags::inbounds);
            LLVM::StoreOp::create(rewriter, loc, llvmVal, slot);
        }
        Value argsSize =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(totalSize));

        // Alloca result buffer + size.
        Value outBufSlot = getStaticAlloca(loc, rewriter, ptrTy, 1);
        Value outSizeSlot = getStaticAlloca(loc, rewriter, i64Ty, 1);

        // Call the runtime.
        LLVM::CallOp::create(
            rewriter, loc, callFn,
            ValueRange{addrPtr, symPtr, argsBuf, argsSize, outBufSlot, outSizeSlot});

        // Decode return value (if any).
        SmallVector<Value> returns;
        Value outBuf;
        if (!op.getResultTypes().empty()) {
            if (op.getResultTypes().size() != 1) {
                return op->emitOpError("remote_lib_call supports at most one result");
            }
            Type retTy = op.getResultTypes().front();
            if (primitiveByteSize(retTy) < 0) {
                return op->emitOpError("unsupported return type for remote_lib_call: ") << retTy;
            }
            Type retLLVMTy = getTypeConverter()->convertType(retTy);
            outBuf = LLVM::LoadOp::create(rewriter, loc, ptrTy, outBufSlot);
            Value rv = LLVM::LoadOp::create(rewriter, loc, retLLVMTy, outBuf);
            returns.push_back(rv);
        }
        else {
            outBuf = LLVM::LoadOp::create(rewriter, loc, ptrTy, outBufSlot);
        }

        // Release the runtime-allocated result buffer.
        LLVM::CallOp::create(rewriter, loc, freeFn, ValueRange{outBuf});

        rewriter.replaceOp(op, returns);
        return success();
    }

  private:
    // Supported scalar byte sizes. Returns -1 for unsupported types.
    static int64_t primitiveByteSize(Type ty)
    {
        if (auto i = dyn_cast<IntegerType>(ty)) {
            return (i.getWidth() + 7) / 8;
        }
        if (auto f = dyn_cast<FloatType>(ty)) {
            return (f.getWidth() + 7) / 8;
        }
        if (isa<IndexType>(ty)) {
            return 8;
        }
        if (auto c = dyn_cast<ComplexType>(ty)) {
            int64_t inner = primitiveByteSize(c.getElementType());
            return inner < 0 ? -1 : 2 * inner;
        }
        return -1;
    }
};

struct DefineCallbackOpPattern : public OpConversionPattern<CallbackOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CallbackOp op, CallbackOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Only match with ops without an entry block
        if (!op.empty()) {
            return failure();
        }

        Block *entry;
        rewriter.modifyOpInPlace(op, [&] { entry = op.addEntryBlock(); });
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entry);

        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        auto idAttr = op.getIdAttr();
        auto constantId = LLVM::ConstantOp::create(rewriter, loc, idAttr);
        auto argcAttr = op.getArgcAttr();
        auto constantArgc = LLVM::ConstantOp::create(rewriter, loc, argcAttr);
        auto rescAttr = op.getRescAttr();
        auto constantResc = LLVM::ConstantOp::create(rewriter, loc, rescAttr);

        SmallVector<Value> callArgs = {constantId, constantArgc, constantResc};

        Type i64 = rewriter.getI64Type();
        Type voidType = LLVM::LLVMVoidType::get(ctx);
        bool isVarArg = true;
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        auto typeConverter = getTypeConverter();
        LLVM::LLVMFuncOp customCallFnOp =
            mlir::LLVM::lookupOrCreateFn(rewriter, mod, "__catalyst_inactive_callback",
                                         {/*args=*/i64, i64, i64},
                                         /*ret_type=*/voidType, isVarArg)
                .value();
        SmallVector<Attribute> passthroughs;
        auto keyAttr = StringAttr::get(ctx, "nofree");
        passthroughs.push_back(keyAttr);
        customCallFnOp.setPassthroughAttr(ArrayAttr::get(ctx, passthroughs));
        // TODO: remove redundant alloca+store since ultimately we'll receive struct*
        for (auto arg : op.getArguments()) {
            Type structTy = typeConverter->convertType(arg.getType());
            auto structVal =
                UnrealizedConversionCastOp::create(rewriter, loc, structTy, arg).getResult(0);
            Value ptr = getStaticAlloca(loc, rewriter, structTy, 1);
            LLVM::StoreOp::create(rewriter, loc, structVal, ptr);
            callArgs.push_back(ptr);
        }
        LLVM::CallOp::create(rewriter, loc, customCallFnOp, callArgs);
        func::ReturnOp::create(rewriter, loc, TypeRange{}, ValueRange{});
        return success();
    }
};

struct ReplaceCallbackOpWithFuncOp : public OpConversionPattern<CallbackOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CallbackOp op, CallbackOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Only match with ops with an entry block
        if (op.empty()) {
            return failure();
        }

        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                                               op.getFunctionType());
        func.setPrivate();
        auto noinline = rewriter.getStringAttr("noinline");
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        SmallVector<Attribute> passthrough = {noinline};
        auto ctx = rewriter.getContext();
        func->setAttr("passthrough", ArrayAttr::get(ctx, passthrough));
        auto typeConverter = getTypeConverter();
        gradient::wrapMemRefArgsFunc(func, typeConverter, rewriter, op.getLoc());
        rewriter.eraseOp(op);
        return success();
    }
};

struct CallbackCallOpPattern : public OpConversionPattern<CallbackCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CallbackCallOp op, CallbackCallOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Just change the calling convention from scalar replacement of aggregates
        // to pointer to struct.
        auto loc = op.getLoc();
        SmallVector<Value> callArgs;
        for (auto structVal : adaptor.getInputs()) {
            // allocate a memref descriptor on the stack
            Value ptr = getStaticAlloca(loc, rewriter, structVal.getType(), 1);
            // store the memref descriptor on the pointer
            LLVM::StoreOp::create(rewriter, loc, structVal, ptr);
            // add the ptr to the arguments
            callArgs.push_back(ptr);
        }
        func::CallOp::create(rewriter, loc, adaptor.getCallee(), TypeRange{}, callArgs);
        rewriter.eraseOp(op);
        return success();
    }
};

struct CustomGradOpPattern : public OpConversionPattern<gradient::CustomGradOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(gradient::CustomGradOp op, gradient::CustomGradOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // only match after all three are func.func
        auto callee = op.getCalleeAttr();
        auto forward = op.getForwardAttr();
        auto reverse = op.getReverseAttr();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        auto calleeOp = mod.lookupSymbol<func::FuncOp>(callee);
        auto forwardOp = mod.lookupSymbol<func::FuncOp>(forward);
        auto reverseOp = mod.lookupSymbol<func::FuncOp>(reverse);
        auto ready = calleeOp && forwardOp && reverseOp;
        if (!ready) {
            return failure();
        }

        auto loc = op.getLoc();
        gradient::insertEnzymeCustomGradient(rewriter, mod, loc, calleeOp, forwardOp, reverseOp);
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
        patterns.add<RemoteCustomCallOpPattern>(typeConverter, context);
        patterns.add<RemoteOpenOpPattern>(typeConverter, context);
        patterns.add<RemoteSendBinaryOpPattern>(typeConverter, context);
        patterns.add<RemoteLibCallOpPattern>(typeConverter, context);
        patterns.add<PrintOpPattern>(typeConverter, context);
        patterns.add<AssertionOpPattern>(typeConverter, context);
        patterns.add<DefineCallbackOpPattern>(typeConverter, context);
        patterns.add<ReplaceCallbackOpWithFuncOp>(typeConverter, context);
        patterns.add<CallbackCallOpPattern>(typeConverter, context);
        patterns.add<CustomGradOpPattern>(typeConverter, context);

        LLVMConversionTarget target(*context);
        target.addLegalDialect<func::FuncDialect>();
        target.addIllegalDialect<CatalystDialect>();
        target.addIllegalDialect<catalyst::gradient::GradientDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace catalyst
