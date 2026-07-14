// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/Support/Path.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "Catalyst/Utils/StaticAllocas.h"

#include "Executor/IR/ExecutorOps.h"
#include "Executor/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace executor {

#define GEN_PASS_DEF_CONVERTEXECUTORTOLLVMPASS
#include "Executor/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Local conversion helpers
//===----------------------------------------------------------------------===//

Value getGlobalString(Location loc, OpBuilder &rewriter, StringRef key, StringRef value,
                      ModuleOp mod)
{
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size());
    LLVM::GlobalOp glb = mod.lookupSymbol<LLVM::GlobalOp>(key);
    if (!glb) {
        OpBuilder::InsertionGuard guard(rewriter);
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

/// Byte size of a primitive element, used for the byte-buffer marshalling path.
int64_t primitiveByteSize(Type ty)
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

int64_t memrefElemSizeBytes(MemRefType ty) { return primitiveByteSize(ty.getElementType()); }

//===----------------------------------------------------------------------===//
// executor.open  ->  __catalyst__remote__open(addr)
//===----------------------------------------------------------------------===//

struct OpenOpLowering : public OpConversionPattern<executor::OpenOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(executor::OpenOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i64Ty = rewriter.getI64Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        Type openSig = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy});
        LLVM::LLVMFuncOp openFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__open", openSig);

        Value addrPtr =
            getGlobalString(loc, rewriter, "remote_setup_addr", op.getAddress().str() + '\0', mod);

        LLVM::CallOp::create(rewriter, loc, openFn, ValueRange{addrPtr});
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// executor.send_binary  ->  __catalyst__remote__send_binary(addr, path, format)
//===----------------------------------------------------------------------===//

struct SendBinaryOpLowering : public OpConversionPattern<executor::SendBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(executor::SendBinaryOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = rewriter.getI32Type();
        Type i64Ty = rewriter.getI64Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        Type sendBinSig = LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, ptrTy, i32Ty});
        LLVM::LLVMFuncOp sendBinFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__send_binary", sendBinSig);

        std::string tag = llvm::sys::path::stem(op.getBinaryPath()).str();
        Value addrPtr =
            getGlobalString(loc, rewriter, "remote_addr_" + tag, op.getAddress().str() + '\0', mod);
        Value pathPtr = getGlobalString(loc, rewriter, "remote_path_" + tag,
                                        op.getBinaryPath().str() + '\0', mod);
        Value formatTag =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(op.getFormat()));

        LLVM::CallOp::create(rewriter, loc, sendBinFn, ValueRange{addrPtr, pathPtr, formatTag});
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// executor.launch  ->  __catalyst__remote__launch(addr, sym,
//                                               num_in,  in_descs,  in_ranks,  in_sizes,
//                                               num_out, out_descs, out_ranks, out_sizes)
//===----------------------------------------------------------------------===//

struct LaunchOpLowering : public OpConversionPattern<executor::LaunchOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(executor::LaunchOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i64Ty = rewriter.getI64Type();
        Type voidTy = LLVM::LLVMVoidType::get(ctx);
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        // parameters:
        // - addr: the address of the remote executor
        // - symbol: the symbol to invoke
        // - num_inputs: the number of input arguments
        // - input_descs: the input descriptor array
        // - input_ranks: the input rank array
        // - input_sizes: the input size array
        // - num_outputs: the number of output arguments
        // - output_descs: the output descriptor array
        // - output_ranks: the output rank array
        Type launchSig = LLVM::LLVMFunctionType::get(
            voidTy, {ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy});
        LLVM::LLVMFuncOp launchFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__launch", launchSig);

        std::string callee = op.getKernelCallee().str();
        Value addrPtr = getGlobalString(loc, rewriter, "remote_addr_" + callee,
                                        op.getAddress().str() + '\0', mod);

        std::string symbolName = "_catalyst_pyface_" + callee;
        Value symbolPtr =
            getGlobalString(loc, rewriter, "remote_sym_" + callee, symbolName + '\0', mod);

        SmallVector<Value> inputDescPtrs;
        SmallVector<int64_t> inputRanks, inputElemSizes;
        for (auto [origInput, llvmInput] : llvm::zip(op.getInputs(), adaptor.getInputs())) {
            auto memrefTy = cast<MemRefType>(origInput.getType());
            int64_t elemSz = memrefElemSizeBytes(memrefTy);
            if (elemSz < 0) {
                return op->emitOpError("unsupported memref element type for executor.launch: ")
                       << memrefTy.getElementType();
            }
            inputRanks.push_back(memrefTy.getRank());
            inputElemSizes.push_back(elemSz);
            Value alloca = getStaticAlloca(loc, rewriter, llvmInput.getType(), 1);
            LLVM::StoreOp::create(rewriter, loc, llvmInput, alloca);
            inputDescPtrs.push_back(alloca);
        }

        SmallVector<Value> outputDescPtrs;
        SmallVector<int64_t> outputRanks, outputElemSizes;
        for (Type resultTy : op.getResultTypes()) {
            auto memrefTy = cast<MemRefType>(resultTy);
            int64_t elemSz = memrefElemSizeBytes(memrefTy);
            if (elemSz < 0) {
                return op->emitOpError("unsupported memref element type for executor.launch: ")
                       << memrefTy.getElementType();
            }
            outputRanks.push_back(memrefTy.getRank());
            outputElemSizes.push_back(elemSz);
            Type llvmDescTy = getTypeConverter()->convertType(resultTy);
            Value alloca = getStaticAlloca(loc, rewriter, llvmDescTy, 1);
            outputDescPtrs.push_back(alloca);
        }

        Value inputDescsArr = buildStackPtrArray(loc, rewriter, inputDescPtrs);
        Value outputDescsArr = buildStackPtrArray(loc, rewriter, outputDescPtrs);

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

//===----------------------------------------------------------------------===//
// executor.call  ->  __catalyst__remote__call_wrapper(addr, sym,
//                                                   args_buf, args_size,
//                                                   &out_buf, &out_size)
//===----------------------------------------------------------------------===//

struct CallOpLowering : public OpConversionPattern<executor::CallOp> {
    using OpConversionPattern::OpConversionPattern;

    /// Static-shape memref byte count, or -1 on unsupported types.
    static int64_t memrefBufferBytes(Type ty, Operation *op)
    {
        auto memrefTy = dyn_cast<MemRefType>(ty);
        if (!memrefTy) {
            op->emitOpError("executor.call requires memref-typed operands; got ") << ty;
            return -1;
        }
        if (!memrefTy.hasStaticShape()) {
            op->emitOpError("executor.call requires static-shape memref args; got ") << memrefTy;
            return -1;
        }
        int64_t elemSz = primitiveByteSize(memrefTy.getElementType());
        if (elemSz < 0) {
            op->emitOpError("unsupported memref element type for executor.call: ")
                << memrefTy.getElementType();
            return -1;
        }
        return memrefTy.getNumElements() * elemSz;
    }

    LogicalResult matchAndRewrite(executor::CallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = rewriter.getI32Type();
        Type i64Ty = rewriter.getI64Type();
        Type voidTy = LLVM::LLVMVoidType::get(ctx);
        Type i8Ty = rewriter.getI8Type();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        // parameters:
        // - addr: the address of the remote executor
        // - symbol: the symbol to invoke
        // - args_buf: the input buffer
        // - args_size: the size of the input buffer
        // - out_buf: the output buffer
        // - out_size: the size of the output buffer
        Type callSig =
            LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy});
        LLVM::LLVMFuncOp callFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__call_wrapper", callSig);
        Type freeSig = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
        LLVM::LLVMFuncOp freeFn = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__remote__free_result", freeSig);

        unsigned numInputs =
            op.getNumInputArgs().value_or(static_cast<int32_t>(op.getInputs().size()));
        if (numInputs > op.getInputs().size()) {
            return op->emitOpError("num_input_args exceeds operand count");
        }

        SmallVector<int64_t> inputOffsets;
        int64_t totalInputBytes = 0;
        for (unsigned i = 0; i < numInputs; ++i) {
            int64_t argSize = memrefBufferBytes(op.getInputs()[i].getType(), op);
            if (argSize < 0) {
                return failure();
            }
            inputOffsets.push_back(totalInputBytes);
            totalInputBytes += argSize;
        }
        for (unsigned i = numInputs; i < op.getInputs().size(); ++i) {
            if (memrefBufferBytes(op.getInputs()[i].getType(), op) < 0) {
                return failure();
            }
        }

        Type bufTy = LLVM::LLVMArrayType::get(i8Ty, totalInputBytes > 0 ? totalInputBytes : 1);
        std::string sym = op.getSymbol().str();

        Value addrPtr = getGlobalString(loc, rewriter, "remote_lib_addr_" + sym,
                                        op.getAddress().str() + '\0', mod);
        Value symPtr = getGlobalString(loc, rewriter, "remote_lib_sym_" + sym, sym + '\0', mod);

        Value argsBuf = getStaticAlloca(loc, rewriter, bufTy, 1);
        for (unsigned i = 0; i < numInputs; ++i) {
            auto memrefTy = cast<MemRefType>(op.getInputs()[i].getType());
            int64_t numBytes =
                memrefTy.getNumElements() * primitiveByteSize(memrefTy.getElementType());
            Value src = MemRefDescriptor(adaptor.getInputs()[i]).alignedPtr(rewriter, loc);
            Value slot = LLVM::GEPOp::create(
                rewriter, loc, ptrTy, bufTy, argsBuf,
                ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(inputOffsets[i])},
                LLVM::GEPNoWrapFlags::inbounds);
            Value sizeVal =
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(numBytes));
            LLVM::MemcpyOp::create(rewriter, loc, slot, src, sizeVal, /*isVolatile=*/false);
        }
        Value argsSize =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(totalInputBytes));

        Value outBufSlot = getStaticAlloca(loc, rewriter, ptrTy, 1);
        Value outSizeSlot = getStaticAlloca(loc, rewriter, i64Ty, 1);

        LLVM::CallOp::create(
            rewriter, loc, callFn,
            ValueRange{addrPtr, symPtr, argsBuf, argsSize, outBufSlot, outSizeSlot});

        Value outBuf = LLVM::LoadOp::create(rewriter, loc, ptrTy, outBufSlot);
        int64_t outOffset = 0;
        for (unsigned i = numInputs; i < op.getInputs().size(); ++i) {
            auto memrefTy = cast<MemRefType>(op.getInputs()[i].getType());
            int64_t numBytes =
                memrefTy.getNumElements() * primitiveByteSize(memrefTy.getElementType());
            Value destPtr = MemRefDescriptor(adaptor.getInputs()[i]).alignedPtr(rewriter, loc);
            Value src = LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, outBuf,
                                            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(outOffset)},
                                            LLVM::GEPNoWrapFlags::inbounds);
            Value sizeVal =
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(numBytes));
            LLVM::MemcpyOp::create(rewriter, loc, destPtr, src, sizeVal, /*isVolatile=*/false);
            outOffset += numBytes;
        }

        LLVM::CallOp::create(rewriter, loc, freeFn, ValueRange{outBuf});
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertExecutorToLLVMPass : impl::ConvertExecutorToLLVMPassBase<ConvertExecutorToLLVMPass> {
    using ConvertExecutorToLLVMPassBase::ConvertExecutorToLLVMPassBase;

    void runOnOperation() final
    {
        MLIRContext *ctx = &getContext();
        ModuleOp mod = getOperation();

        LLVMTypeConverter typeConverter(ctx);

        RewritePatternSet patterns(ctx);
        patterns.add<OpenOpLowering, SendBinaryOpLowering, LaunchOpLowering, CallOpLowering>(
            typeConverter, ctx);

        LLVMConversionTarget target(*ctx);
        target.addLegalOp<ModuleOp>();
        target.addIllegalDialect<executor::ExecutorDialect>();

        if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

} // namespace executor
} // namespace catalyst
